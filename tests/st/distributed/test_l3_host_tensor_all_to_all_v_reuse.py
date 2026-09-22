# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: repeated HOST-rail ``pld.tensor.all_to_all_v`` with skew.

Exercises the two-round credit barrier across REPEATED invocations that share
one set of comm-domain windows — the shape the receive window lifecycle and
the counts lifetime actually depend on (review feedback on the counts redesign):

  * three STRAIGHT-LINE collective calls on the SAME ``stage`` / ``data`` /
    ``signal`` / ``counts`` / ``recv`` windows (the HOST rail rejects calls
    nested in ``for``/``while`` loops, so the calls are unrolled);
  * the input window is RESTAGED with different values before each call, and
    the per-destination counts CHANGE between calls (1 row per destination,
    then the full MAX_RECV, then a mixed/over-capacity matrix), so a stale
    counts read or a receive-window overwrite changes the observed values;
  * before consume #0 and #1, each rank burns ``AAV_REUSE_SPIN * (nr-1-rank)``
    loop iterations — rank 0 lags, the last rank races ahead. The fast rank
    therefore reaches invocation k+1's payload push while the slow rank is
    still consuming invocation k's data, which is exactly the interleaving the
    two barriers must serialize (A: nothing is pushed until every rank
    consumed the previous window; B: nobody restages counts until every peer
    read them).

Correctness is asserted per invocation on both counts: ``recv_counts[src]``
must equal the clamped counts the source staged for that call, and every valid
row must carry the source's per-call payload. The tail rows of each capacity
slot are deliberately NOT asserted: after call 0 they may hold leftovers of an
earlier invocation, which is the documented "trim to recv_counts" contract.

P=2 and P=4 (skips when fewer devices are available). Tune the skew with
``AAV_REUSE_SPIN`` (default 300000).
"""

import os
import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64
MAX_RECV = 4
N_CALLS = 3
SCRATCH_ROWS = 64

# Outer iterations of the per-rank busy loop at the top of consume #0 and #1.
# rank r burns SPIN * (nr - 1 - r) of them, i.e. the last rank does not spin.
SPIN = int(os.environ.get("AAV_REUSE_SPIN", "300000"))

# Skew multiplier per invocation: consume #2 is the last one, nothing pushes
# after it, so it needs no skew.
_SKEW_PER_CALL = (SPIN, SPIN, 0)


def _effective_rows(count: int, max_recv: int) -> int:
    """Rows actually transferred: ``clamp(count, 0, MAX_RECV)``."""
    return max(0, min(count, max_recv))


def _counts_for_call(call: int, nr: int, mr: int) -> list[list[int]]:
    """Per-(sender, dest) counts for one invocation.

    Deliberately extreme between calls so an overwrite is visible: call 1
    pushes ``MAX_RECV`` rows into every destination (rows a slow rank has
    already read as valid after call 0), call 2 mixes zero, small and
    over-capacity values (over-capacity exercises the clamp).
    """
    if call == 0:
        return [[1] * nr for _ in range(nr)]
    if call == 1:
        return [[mr] * nr for _ in range(nr)]
    return [[(2 * r + d) % (mr + 2) for d in range(nr)] for r in range(nr)]


def _build_reuse_program(n_ranks: int, max_recv: int):
    """Build the N-rank reuse program: three straight-line exchanges."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr

    @pl.program
    class HostAllToAllVReuse:
        """Three back-to-back exchanges on one set of windows, with skew."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            for row in pl.range(total):
                chunk = pl.load(inp, [row, 0], [1, SIZE])
                stage = pl.store(chunk, [row, 0], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            self.stage_step(inp, stage)

        @pl.function(type=pl.FunctionType.InCore)
        def fill_counts_step(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            for d in pl.range(nr):
                v = pl.read(counts_row, [d, 0])
                pl.write(counts, [d, 0], v)

        @pl.function(type=pl.FunctionType.Orchestration)
        def fill_counts_orch(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            self.fill_counts_step(counts_row, counts)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            n_spin: pl.Scalar[pl.INT32],
            scratch: pl.Out[pl.Tensor[[SCRATCH_ROWS + 1, 1], pl.INT32]],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            # Rank-skewed slow consumer: burn the GM-store loop BEFORE reading
            # the window, so a fast rank genuinely overlaps this consumption.
            # The trailing marker row keeps `n_spin` itself observable even if
            # the compiler collapses the loop (the loop writes rows
            # [0, SCRATCH_ROWS) only).
            pl.write(scratch, [SCRATCH_ROWS, 0], pl.cast(n_spin, pl.INT32))
            spin_n = pl.cast(n_spin, pl.INDEX)
            for i in pl.range(spin_n):
                for j in pl.range(SCRATCH_ROWS):
                    pl.write(scratch, [j, 0], pl.cast(i, pl.INT32))

            # Mirror the whole window (valid rows checked against golden, tail
            # rows only to keep `out` fully written — a pl.Out tensor is
            # write-only on the device).
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            n_spin: pl.Scalar[pl.INT32],
            scratch: pl.Out[pl.Tensor[[SCRATCH_ROWS + 1, 1], pl.INT32]],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            return self.consume_step(data, recv_counts, n_spin, scratch, out, recv_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[N_CALLS, nr, total, SIZE], pl.FP32],
            send_counts_in: pl.Tensor[[N_CALLS, nr, nr, 1], pl.INT32],
            scratch: pl.Out[pl.Tensor[[N_CALLS, nr, SCRATCH_ROWS + 1, 1], pl.INT32]],
            outputs: pl.Out[pl.Tensor[[N_CALLS, nr, total, SIZE], pl.FP32]],
            recv_outputs: pl.Out[pl.Tensor[[N_CALLS, nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[N_CALLS, nr, total, SIZE], pl.FP32], pl.Tensor[[N_CALLS, nr, nr, 1], pl.INT32]]:
            input_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            # Windows are RE-BOUND per use (`stage = pld.window(...)` again over
            # the SAME buffer): the InOut-use discipline kills a variable once a
            # user-function call consumes it as Out, and every dispatch passes
            # its window by variable. Fresh views over the same allocation keep
            # the memory shared while each call gets a clean binding — the same
            # idiom the single-shot HOST ST uses inside its dispatch loops.
            #
            # All three invocations share one set of buffers: the signal credits
            # accumulate across them, counts are restaged in place, and the data
            # window is overwritten — the exact window lifecycle the two
            # barriers protect.

            # --- invocation 0: one row per destination -------------------
            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[0, r], stage, device=r)
            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts_in[0, r], counts, device=r)
            stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv)
            for r in pl.range(pld.world_size()):
                spin = (pld.world_size() - 1 - r) * _SKEW_PER_CALL[0]
                self.consume_orch(
                    data, recv, spin, scratch[0, r], outputs[0, r], recv_outputs[0, r], device=r
                )

            # --- invocation 1: full MAX_RECV push ------------------------
            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[1, r], stage, device=r)
            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts_in[1, r], counts, device=r)
            stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv)
            for r in pl.range(pld.world_size()):
                spin = (pld.world_size() - 1 - r) * _SKEW_PER_CALL[1]
                self.consume_orch(
                    data, recv, spin, scratch[1, r], outputs[1, r], recv_outputs[1, r], device=r
                )

            # --- invocation 2: mixed / over-capacity counts --------------
            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[2, r], stage, device=r)
            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts_in[2, r], counts, device=r)
            stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv)
            for r in pl.range(pld.world_size()):
                spin = (pld.world_size() - 1 - r) * _SKEW_PER_CALL[2]
                self.consume_orch(
                    data, recv, spin, scratch[2, r], outputs[2, r], recv_outputs[2, r], device=r
                )

            return outputs, recv_outputs

    return HostAllToAllVReuse


class TestL3HostTensorAllToAllVReuse:
    """Repeated-exchange coverage: credits, counts lifetime, window reuse."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_repeated_calls_with_skewed_consumers(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all_to_all_v reuse P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        compiled = ir.compile(
            _build_reuse_program(nr, mr),
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0),
        )

        # Per-call payloads: distinct constant offsets keep every (call, sender,
        # destination, row) value unique, so a value written by the WRONG
        # invocation cannot alias a correct one.
        inputs = torch.zeros((N_CALLS, nr, total, SIZE), dtype=torch.float32)
        send_counts_in = torch.zeros((N_CALLS, nr, nr, 1), dtype=torch.int32)
        counts = [_counts_for_call(k, nr, mr) for k in range(N_CALLS)]
        for k in range(N_CALLS):
            for r in range(nr):
                for d in range(nr):
                    send_counts_in[k, r, d, 0] = counts[k][r][d]
                    base = d * mr
                    for row in range(mr):
                        for j in range(SIZE):
                            inputs[k, r, base + row, j] = float(
                                (k + 1) * 100000 + r * 1000 + d * 100 + row * 10 + j % 10
                            )

        scratch = torch.zeros((N_CALLS, nr, SCRATCH_ROWS + 1, 1), dtype=torch.int32)
        outputs = torch.zeros((N_CALLS, nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((N_CALLS, nr, nr, 1), dtype=torch.int32)
        compiled(inputs, send_counts_in, scratch, outputs, recv_outputs)

        for k in range(N_CALLS):
            for rank in range(nr):
                for src in range(nr):
                    n_rows = _effective_rows(counts[k][src][rank], mr)

                    got_count = int(recv_outputs[k, rank, src, 0].item())
                    assert got_count == n_rows, (
                        f"P={nr} call={k} rank={rank} src={src}: recv_counts={got_count} "
                        f"!= clamped({counts[k][src][rank]}) = {n_rows} — a stale counts read or a "
                        f"counts restage that raced a peer's pull"
                    )

                    base = src * mr
                    for row in range(n_rows):
                        expected_row = inputs[k, src, rank * mr + row, :]
                        got_row = outputs[k, rank, base + row, :]
                        assert torch.allclose(got_row, expected_row, atol=1e-5), (
                            f"P={nr} call={k} rank={rank} src={src} row={row}: max diff = "
                            f"{(got_row - expected_row).abs().max().item()} — a faster rank's later "
                            f"invocation likely overwrote this rank's receive window mid-consumption"
                        )

        # The skew fixture itself must have run: consume #1 (the last spinning
        # one) leaves, in ITS OWN scratch slice, the marker row holding
        # SPIN*(nr-1-r) and the loop rows holding the last counter minus one;
        # consume #2 does not spin at all. If these read 0 the skew never
        # happened and the payload checks above proved nothing about skewed
        # progress.
        for r in range(nr):
            expected_spin = SPIN * (nr - 1 - r)
            if expected_spin == 0:
                continue
            got_marker = int(scratch[1, r, SCRATCH_ROWS, 0].item())
            assert got_marker == expected_spin, (
                f"P={nr} rank={r}: skew scalar did not arrive (marker={got_marker}, expected {expected_spin})"
            )
            got_counter = int(scratch[1, r, 3, 0].item())
            assert got_counter == expected_spin - 1, (
                f"P={nr} rank={r}: skew loop did not run (counter={got_counter}, "
                f"expected {expected_spin - 1})"
            )
            assert int(scratch[2, r, SCRATCH_ROWS, 0].item()) == 0, (
                f"P={nr} rank={r}: the last consume must not spin"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
