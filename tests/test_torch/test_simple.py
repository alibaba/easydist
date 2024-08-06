# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
import torch
import functorch
from functorch.compile import aot_function
import rich

from easydist.utils.testing.mock import TorchMockDeviceMesh
from easydist.torch import EDTorchShardingAnn, set_device_mesh
from easydist.torch.passes import fix_addmm_bias, eliminate_detach
from easydist import easydist_setup


def fn_1(x, y):
    return torch.concat([x, y], dim=1)


def fn_2(x, y):
    return torch.mm(torch.exp(torch.tanh(x)), y)


@functorch.compile.make_boxed_compiler
def compiler_fn(fx_module: torch.fx.GraphModule, inps):
    fx_module = fix_addmm_bias(fx_module)
    fx_module = eliminate_detach(fx_module)
    fx_module.recompile()
    print(fx_module.graph)

    sharding_interpreter = EDTorchShardingAnn(fx_module)
    sharding_info, fwd_shape_info = sharding_interpreter.run(*inps)
    rich.print("sharding_info:\n", sharding_info)
    rich.print("fwd_shape_info:\n", fwd_shape_info)

    return fx_module

@pytest.mark.skip
@pytest.mark.parametrize("fn", [fn_1, fn_2])
def test_simple_case(fn):
    easydist_setup("torch")

    mock_mesh = TorchMockDeviceMesh(1, 2, debug_only=True)
    set_device_mesh(mock_mesh)

    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)
    aot_print_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
    res = aot_print_fn(x, y)

    grad_res = torch.ones_like(res)
    res.backward(grad_res)


if __name__ == '__main__':
    test_simple_case()
