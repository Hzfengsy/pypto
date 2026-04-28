# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AutoTileMatmulL0 pass.

The pass walks Mat-resident ``tile.matmul`` calls, queries
``utils::ChooseL0Tile`` against the active backend's L0 capacities, and rewrites
each call into a peeled first matmul plus a K-loop of ``tile.matmul_acc`` over
Mat-resident slices.  The K-loop is marked ``ForKind.Pipeline`` with
``pipeline_stages=2`` whenever the tiled K dimension produces at least three
sub-iterations (so the loop body has at least two iterations for ping-pong).

The conftest configures the Ascend950 backend, which advertises L0a/L0b = 64KB
and L0c = 256KB.  Tests rely on those capacities to predict the chooser's
output.

Tests use property-based assertions (number/kind of statements, attribute
presence) rather than full ``assert_structural_equal`` because the DSL surface
for emitting the exact peeled matmul + iter-arg ForStmt block is verbose; the
properties checked here are what downstream passes care about.
"""

from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes


def _find_calls_by_op(stmts: list[Any], op_name: str) -> list[Any]:
    """Recursively walk a list of statements and return all Calls whose op
    name matches ``op_name``."""
    found: list[Any] = []

    def visit(s: Any) -> None:
        if isinstance(s, ir.AssignStmt):
            v = s.value
            if isinstance(v, ir.Call) and v.op.name == op_name:
                found.append(v)
        # Recurse into common structural containers.
        for attr in ("body", "then_body", "else_body"):
            child = getattr(s, attr, None)
            if child is not None:
                visit(child)
        for attr in ("stmts",):
            children = getattr(s, attr, None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    visit(c)

    for s in stmts:
        visit(s)
    return found


def _find_for_stmts(root: Any) -> list[Any]:
    """Recursively walk an IR root and return all ForStmts."""
    found: list[Any] = []

    def visit(s: Any) -> None:
        if isinstance(s, ir.ForStmt):
            found.append(s)
        for attr in ("body", "then_body", "else_body"):
            child = getattr(s, attr, None)
            if child is not None:
                visit(child)
        for attr in ("stmts",):
            children = getattr(s, attr, None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    visit(c)

    visit(root)
    return found


def _kernel_body(program: ir.Program, name: str = "kernel") -> Any:
    func = next(f for gv, f in program.functions.items() if gv.name == name)
    return func.body


def _flatten_stmts(body: Any) -> list[Any]:
    """Return body's top-level statement list (1 stmt or N stmts in SeqStmts)."""
    if isinstance(body, ir.SeqStmts):
        return list(body.stmts)
    return [body]


class TestAutoTileMatmulL0KOnly:
    """K-tiling rewrites for Mat-resident tile.matmul."""

    def test_skinny_gemm_pipelined(self):
        """16x64 @ 2048 BF16 → ChooseL0Tile picks (m=16, n=64, k=256).

        K=2048 → 8 K-iterations after peeling → 7 loop iters → Pipeline
        marker with pipeline_stages=2.  The pass should emit:

            slice_a_0 = tile.slice(lhs_mat, [16, 256], [0, 0])
            slice_b_0 = tile.slice(rhs_mat, [256, 64], [0, 0])
            c_init   = tile.matmul(slice_a_0, slice_b_0)
            for ko in pl.pipeline(256, 2048, 256, init_values=(c_init,), stage=2):
                slice_a = tile.slice(lhs_mat, [16, 256], [0, ko])
                slice_b = tile.slice(rhs_mat, [256, 64], [ko, 0])
                c_new   = tile.matmul_acc(c, slice_a, slice_b)
                yield c_new
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)

        body_stmts = _flatten_stmts(_kernel_body(After))
        # Expected stmt count: lhs_mat, rhs_mat, slice_a_0, slice_b_0, c_init, for, store, return
        assert len(body_stmts) == 8, (
            f"Expected 8 top-level statements after rewrite, got {len(body_stmts)}: "
            f"{[type(s).__name__ for s in body_stmts]}"
        )

        # No raw tile.matmul on the original Mat operands at the top level.
        # After the rewrite, the only top-level tile.matmul should be the peeled
        # first sub-matmul.
        matmul_calls = _find_calls_by_op(body_stmts, "tile.matmul")
        assert len(matmul_calls) == 1, f"Expected exactly one peeled tile.matmul, got {len(matmul_calls)}"

        # Find the K-loop and verify its kind / attrs.
        for_stmts = _find_for_stmts(_kernel_body(After))
        assert len(for_stmts) == 1
        k_loop = for_stmts[0]
        assert k_loop.kind == ir.ForKind.Pipeline, f"Expected Pipeline kind, got {k_loop.kind}"
        attrs = dict(k_loop.attrs)
        assert "pipeline_stages" in attrs, f"Missing pipeline_stages attr; got {attrs}"
        assert attrs["pipeline_stages"] == 2, f"Expected pipeline_stages=2, got {attrs['pipeline_stages']}"
        # Loop has one iter_arg (the accumulator) and one return var (out_acc).
        assert len(k_loop.iter_args) == 1
        assert len(k_loop.return_vars) == 1
        # Body contains a tile.matmul_acc.
        body_stmts_inner = _flatten_stmts(k_loop.body)
        matmul_acc_calls = _find_calls_by_op(body_stmts_inner, "tile.matmul_acc")
        assert len(matmul_acc_calls) == 1

    def test_already_l0_sized_skipped(self):
        """64×64×64 BF16 → fits in L0 capacity after double-buffering →
        ChooseL0Tile returns (M, N, K) → pass leaves the matmul untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[64, 64], pl.BF16],
                rhs: pl.Tensor[[64, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                lhs_mat: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_pass_idempotent(self):
        """Running the pass twice produces the same result as running it once.

        After the rewrite, the only ``tile.matmul`` left in the program is the
        peeled first iteration, whose operands are ``tile.slice`` results (not
        original Mat-loaded operands).  The second run shouldn't match again."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        once = passes.auto_tile_matmul_l0()(Before)
        twice = passes.auto_tile_matmul_l0()(once)
        ir.assert_structural_equal(twice, once)


class TestAutoTileMatmulL0Skips:
    """Cases where the pass intentionally leaves the matmul untouched."""

    def test_matmul_acc_left_untouched(self):
        """``tile.matmul_acc`` is out of scope for v1 — emit perfhint, skip."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                acc_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 2048], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 2048], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[2048, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [2048, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc_init, lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)

    def test_non_mat_operands_left_untouched(self):
        """Operands not in ``MemorySpace::Mat`` (e.g. Vec or Acc) are out of
        scope; the pass shouldn't try to tile them."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 2048], pl.BF16],
                rhs: pl.Tensor[[2048, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                # Default tile.load lands in Vec, not Mat.
                lhs_vec: pl.Tile[[16, 2048], pl.BF16] = pl.tile.load(lhs, [0, 0], [16, 2048])
                rhs_vec: pl.Tile[[2048, 64], pl.BF16] = pl.tile.load(rhs, [0, 0], [2048, 64])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_vec, rhs_vec)
                out = pl.store(c, [0, 0], out)
                return out

        After = passes.auto_tile_matmul_l0()(Before)
        ir.assert_structural_equal(After, Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
