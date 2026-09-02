# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixture for the aliased-import dep case.

A ``@pl.jit.incore`` kernel that lives in its *own module*, so a caller can
reach it through ``from _alias_dep_fixture import copy_incore as kern`` — the
shape that made specialization fail with a misleading "missing inferred tensor
metadata" error. Its parameters are named unlike any caller variable, so
nothing can resolve by name coincidence.
"""

import pypto.language as pl
from pypto.jit.decorator import jit


@jit.incore
def copy_incore(src: pl.Tensor, dst: pl.Out[pl.Tensor]) -> pl.Tensor:
    tile = pl.load(src, [0, 0], [64, 64])
    pl.store(tile, [0, 0], dst)
    return dst
