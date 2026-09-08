# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for a ``pl.reshape`` that raises a tile to rank 3.

PTO's ``tile_buf`` is 2D. ``FlattenTileNdTo2D`` collapses every InCore tile to
``[product(leading), last]``, but a ``pl.reshape`` reads its rank off a literal
shape operand rather than off an operand's type, so the pass used to rebuild the
call with the same ND tuple and leave the rank-3 result standing. PTO codegen
then typed the tile from ``shape_[0]`` / ``shape_[1]`` alone
(``ExtractTileTypeInfo``) and dropped the rest: ``[2, 8, 128]`` was emitted as
``rows=2, cols=8`` — 16 elements instead of 2048.

The two planners failed differently, which is why both are covered here:

- ``PTOAS`` skips ``AllocateMemoryAddr``, so the reshape stayed a real
  ``pto.treshape`` between the truncated type and the true 2D one, and ptoas
  rejected the file with ``'pto.treshape' op expects src and dst to have the
  same total byte size``.
- ``PYPTO`` folded the reshape away and emitted ``pto.alloc_tile ... rows=16,
  cols=1`` for a ``[16, 1, 128]`` tile — a truncated allocation, silently.

Random per-element data is what makes the second failure observable: with a
constant fill, a kernel that computes only the first 16 elements still produces
the expected output everywhere.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner

ROWS = 16
COLS = 128
_PLANNERS = [MemoryPlanner.PYPTO, MemoryPlanner.PTOAS]


def _planner_tag(planner: MemoryPlanner) -> str:
    return {MemoryPlanner.PYPTO: "pypto", MemoryPlanner.PTOAS: "ptoas"}[planner]


@pl.jit
def scale_via_3d_reshape(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Scale by 2 with the compute expressed on a rank-3 view of the tile.

    ``[16, 128] -> [2, 8, 128]`` collapses back to ``[16, 128]``, so the flatten
    leaves an identity reshape that ``FoldNoOpReshape`` removes outright.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [ROWS, COLS])
        r = pl.reshape(t, [2, 8, COLS])
        s = pl.tile.mul(r, 2.0)
        b = pl.reshape(s, [ROWS, COLS])
        pl.store(b, [0, 0], out)
    return out


@pl.jit
def scale_via_reshaped_2d(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """The same kernel through a rank-3 shape that is a real 2D reshape.

    ``[16, 128] -> [16, 8, 16]`` collapses to ``[128, 16]``, not to the
    identity, so the ``pto.treshape`` survives ``FoldNoOpReshape`` and the
    emitted type is checked against a tile that actually moves.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [ROWS, COLS])
        r = pl.reshape(t, [ROWS, 8, 16])
        s = pl.tile.mul(r, 2.0)
        b = pl.reshape(s, [ROWS, COLS])
        pl.store(b, [0, 0], out)
    return out


def _source() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(ROWS, COLS, dtype=torch.float32).contiguous()


def _scale_cases():
    for kernel, label in (
        (scale_via_3d_reshape, "identity_collapse"),
        (scale_via_reshaped_2d, "shape_changing_collapse"),
    ):
        for planner in _PLANNERS:
            x = _source()
            out = torch.zeros((ROWS, COLS), dtype=torch.float32)
            yield st.case(
                kernel,
                x,
                out,
                name=f"rank_raising_reshape_{label}_{_planner_tag(planner)}",
                memory_planner=planner,
                golden=lambda _tensors, src=x: src * 2.0,
            )


@st.cases(*_scale_cases())
def test_rank_raising_reshape_scales_every_element(case_run):
    """Every element is scaled — not just the first ``rows * cols`` of a truncated tile."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
