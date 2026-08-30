# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the CubeTileFractalValid property verifier.

pto-isa keeps two independent notions of size for a cube operand:

* the **logical** extent, which ``pto.tmatmul`` reads from the operands' valid
  region and which A2A3 bounds only at ``[1, 4095]`` — ``m = 1`` is a documented
  case, which is why ``pto.mad`` carries a ``disable_gemv`` clause at all; and
* the **physical** extent, which must be a whole NZ fractal box: ``M0 = 16``
  rows by ``C0 / sizeof(dtype)`` cols for ``C0 = 32`` bytes.

Only the second is constrained. Hardware addresses a narrower valid region
inside a full-size tile natively ("compact mode is automatic in matmul when
valid region < physical tile size"), so padding the declared tile and narrowing
it with ``valid_shape`` is always available and costs no DMA.

A sub-fractal *physical* tile has no such fallback. Before this verifier, an
``M = 1`` matmul reached codegen as ``pto.tmatmul`` over a ``rows=1`` L0A/L0C
tile: AutoTileMatmulL0 declined to tile it (reporting a perf hint), ptoas
accepted the shape, and the kernel named a tile the cube cannot address.

The property is in ``GetStructuralProperties()``, so it is verified at pipeline
input for a tile-level program, and re-verified after
``ConvertTensorToTileOps`` — which is where a tensor-level ``pl.matmul`` first
acquires cube operands.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    backend.reset_for_testing()


def _verify(prog, backend_type=BackendType.Ascend910B):
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.CubeTileFractalValid)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def _matmul_program(m, k, n, dtype=pl.FP16, valid_m=None):
    """A tile-level ``[m, k] @ [k, n]``, optionally narrowed to ``valid_m`` rows."""
    valid = [valid_m if valid_m is not None else m, k]

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[m, k], dtype],
            b: pl.Tensor[[k, n], dtype],
            out: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ):
            am: pl.Tile[[m, k], dtype, pl.Mem.Mat] = pl.tile.load(
                a, [0, 0], [m, k], valid_shape=valid, target_memory=pl.Mem.Mat
            )
            bm: pl.Tile[[k, n], dtype, pl.Mem.Mat] = pl.tile.load(b, [0, 0], [k, n], target_memory=pl.Mem.Mat)
            out = pl.store(pl.tile.matmul(am, bm), [0, 0], out)
            return out

    return Prog


def test_sub_fractal_m_rejected():
    """M = 1 is the GEMV/decode shape that reached device as a rows=1 L0A tile."""
    diags = _verify(_matmul_program(1, 128, 64))
    assert len(diags) == 1
    assert diags[0].rule_name == "CubeTileFractalValid"
    assert "1 physical rows" in diags[0].message
    assert "16" in diags[0].message
    # M == 1 has a dedicated instruction; the message must say so rather than
    # only offering the 16x-waste padding.
    assert "pl.tile.gemv" in diags[0].message


def test_sub_fractal_m_suggests_rounded_up_extent():
    """The actionable number is the extent to declare, not the granularity."""
    diags = _verify(_matmul_program(24, 128, 64))
    assert len(diags) == 1
    assert "24 physical rows" in diags[0].message
    assert "Declare 32 physical rows" in diags[0].message
    # 24 rows is not a single-row operand, so GEMV is not the advice here.
    assert "pl.tile.gemv" not in diags[0].message


def test_k_and_n_axes_not_yet_checked():
    """Scope guard, not an endorsement: the K and N granularities are left
    unenforced because the manual's ``K0 = N0 = C0 / E`` disagrees with the
    512-byte ``CUBE_BLOCK_SIZE`` box for any dtype that is not 2 bytes wide, so
    the per-dtype rule for those axes is ambiguous as documented. ``K = 24`` is
    very likely illegal on device and is deliberately not rejected here yet;
    tighten this test together with the check once the ISA owners confirm the
    rule."""
    assert _verify(_matmul_program(64, 24, 64)) == []


def test_sub_fractal_accumulator_rejected():
    """``matmul_acc`` carries a caller-supplied accumulator, so its M axis is
    checked in its own right rather than only through the lhs."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[8, 128], pl.FP16],
            b: pl.Tensor[[128, 64], pl.FP16],
            acc: pl.Tile[[8, 64], pl.FP32, pl.Mem.Acc],
            out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
        ):
            am: pl.Tile[[8, 128], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                a, [0, 0], [8, 128], target_memory=pl.Mem.Mat
            )
            bm: pl.Tile[[128, 64], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                b, [0, 0], [128, 64], target_memory=pl.Mem.Mat
            )
            out = pl.store(pl.tile.matmul_acc(acc, am, bm), [0, 0], out)
            return out

    diags = _verify(Prog)
    assert {d.rule_name for d in diags} == {"CubeTileFractalValid"}
    assert any("accumulator" in d.message and "8 physical rows" in d.message for d in diags)


def test_fractal_shape_accepted():
    assert _verify(_matmul_program(64, 128, 64)) == []


def test_narrowed_valid_shape_accepted():
    """The whole point of the rule: a padded physical tile carrying a 1-row
    logical extent is legal, and is the fix the diagnostic recommends."""
    assert _verify(_matmul_program(16, 128, 64, valid_m=1)) == []


def test_m_granularity_is_dtype_independent():
    """``M0 = 16`` is fixed across element widths -- unlike the K/N box width,
    which the manual derives from ``C0 / sizeof(dtype)``. An int8 kernel gets the
    same row rule as an fp16 one."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[8, 64], pl.INT8],
            b: pl.Tensor[[64, 64], pl.INT8],
            out: pl.Out[pl.Tensor[[8, 64], pl.INT32]],
        ):
            am: pl.Tile[[8, 64], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                a, [0, 0], [8, 64], target_memory=pl.Mem.Mat
            )
            bm: pl.Tile[[64, 64], pl.INT8, pl.Mem.Mat] = pl.tile.load(
                b, [0, 0], [64, 64], target_memory=pl.Mem.Mat
            )
            out = pl.store(pl.tile.matmul(am, bm), [0, 0], out)
            return out

    diags = _verify(Prog)
    assert len(diags) == 1
    assert "8 physical rows" in diags[0].message
    assert "granularity 16" in diags[0].message


def test_gemv_single_row_lhs_accepted():
    """GEMV's one-row left operand is the required A-vector organization, not a
    sub-fractal tile, so its row axis is exempt."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP16],
            b: pl.Tensor[[128, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            am: pl.Tile[[1, 128], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                a, [0, 0], [1, 128], target_memory=pl.Mem.Mat
            )
            bm: pl.Tile[[128, 64], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                b, [0, 0], [128, 64], target_memory=pl.Mem.Mat
            )
            out = pl.store(pl.tile.gemv(am, bm), [0, 0], out)
            return out

    assert _verify(Prog) == []


def test_pipeline_rejects_tile_level_sub_fractal_matmul():
    """Structural, so PassPipeline catches it at ``pipeline_input`` — before any
    lowering, with the user's own span."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with pytest.raises(pypto.Error, match="CubeTileFractalValid"):
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_matmul_program(1, 128, 64))


def test_pipeline_rejects_tensor_level_sub_fractal_matmul():
    """A tensor-level ``pl.matmul`` has no cube operand at pipeline input, so the
    check is re-run after ``ConvertTensorToTileOps`` synthesizes one. Without
    that re-verification this shape compiled all the way to a ``rows=1``
    ``pto.tmatmul``."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP16],
            b: pl.Tensor[[128, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            return pl.tensor.assemble(out, pl.matmul(a, b), [0, 0])

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with pytest.raises(pypto.Error, match="CubeTileFractalValid"):
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)


def test_pipeline_accepts_fractal_matmul():
    """Regression guard against an over-strict check."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_matmul_program(64, 128, 64))


def test_sub_fractal_loop_carried_accumulator_rejected():
    """A carry is only as legal as the value that seeded it.

    The sub-fractal ``tile.create`` here is never itself a matmul operand -- it
    is the loop's ``init_values`` -- so the accumulator reaches ``matmul_acc``
    only through the ``IterArg``. Resolving the carry to its initializer is what
    keeps that path covered.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[8, 512], pl.FP16],
            b: pl.Tensor[[512, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[8, 64], pl.FP32]],
        ):
            acc0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                [8, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for k0, (acc,) in pl.range(0, 512, 128, init_values=(acc0,)):
                am: pl.Tile[[8, 128], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                    a, [0, k0], [8, 128], target_memory=pl.Mem.Mat
                )
                bm: pl.Tile[[128, 64], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                    b, [k0, 0], [128, 64], target_memory=pl.Mem.Mat
                )
                nxt = pl.tile.matmul_acc(acc, am, bm)
                acc_out = pl.yield_(nxt)
            out = pl.store(acc_out, [0, 0], out)
            return out

    diags = _verify(Prog)
    messages = [d.message for d in diags]
    assert any("accumulator" in m and "8 physical rows" in m for m in messages), messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
