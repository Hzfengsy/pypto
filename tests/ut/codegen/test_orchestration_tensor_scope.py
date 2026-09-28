# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor use validation across explicit AUTO runtime scopes."""

import pypto.language as pl
import pytest
from _orchestration_codegen_common import _generate_orch_full_pipeline, _out_of_scope_tensor_refs


@pytest.mark.parametrize("inout", [False, True])
@pytest.mark.parametrize("local_buffer", [False, True])
@pytest.mark.parametrize("use_view", [False, True])
def test_caller_allocated_output_scope(local_buffer, use_view, inout):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def fill(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.tile.full([16], pl.FP32, 1.0), [0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def update(self, out: pl.InOut[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.tile.add(pl.load(out, [0], [16]), 1.0), [0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def consume(
            self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.load(x, [0], [16]), [0], out)

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self, a: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32]
        ) -> pl.Tensor[[16], pl.FP32]:
            with pl.scope():
                if local_buffer:
                    scratch = pl.create_tensor([16], dtype=pl.FP32)
                else:
                    scratch = a
                result = self.fill(scratch)
                if inout:
                    result = self.update(result)
            with pl.scope():
                if use_view:
                    view = pl.reshape(result, [4, 4])
                    out = self.consume(pl.reshape(view, [16]), out)
                else:
                    out = self.consume(result, out)
            return out

    if local_buffer:
        with pytest.raises(ValueError, match="used after its AUTO runtime scope has closed"):
            _generate_orch_full_pipeline(Program)
    else:
        code = _generate_orch_full_pipeline(Program)
        assert not _out_of_scope_tensor_refs(code), code


def test_loop_yield_cannot_read_closed_scope_tensor():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            for i in pl.range(2):
                n = pl.min(i + 1, 2)
                with pl.scope():
                    scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                    for bi in pl.spmd(n):
                        scratch[0:16, 0:16] = pl.add(a, 1.0)
                    a = scratch
            return a

    with pytest.raises(ValueError, match="used after its AUTO runtime scope has closed"):
        _generate_orch_full_pipeline(Program)


def test_sibling_scopes_can_reuse_tensor_source_name():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.scope():
                scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                for bi in pl.spmd(1):
                    scratch[0:16, 0:16] = pl.add(a, 1.0)
                for bj in pl.spmd(1):
                    a[0:16, 0:16] = pl.add(scratch, 1.0)
            with pl.scope():
                scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                for bk in pl.spmd(1):
                    scratch[0:16, 0:16] = pl.add(a, 1.0)
                for bl in pl.spmd(1):
                    a[0:16, 0:16] = pl.add(scratch, 1.0)
            return a

    code = _generate_orch_full_pipeline(Program)
    assert code.count("alloc_tensors(") == 2, code
    assert not _out_of_scope_tensor_refs(code), code
