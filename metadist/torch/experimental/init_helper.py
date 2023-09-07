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

from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensor
from torch.utils._mode_utils import no_dispatch


def fake_to_real(e):
    with no_dispatch():
        if isinstance(e, FakeTensor):
            out = torch.zeros_like(e, device=e.fake_device)
            return out
        return e


@torch.no_grad()
def materialize_zero(tensor, materialization_device):
    tensor = fake_to_real(tensor)
    return torch.zeros_like(tensor, device=materialization_device)


@torch.no_grad()
def materialize_random(tensor, materialization_device):
    if tensor.dtype == torch.bool:
        return torch.rand(tensor.size(), dtype=torch.float, device=materialization_device) > 0.5
    elif torch.is_floating_point(tensor):
        return torch.rand(tensor.size(), dtype=tensor.dtype, device=materialization_device)
    else:
        return torch.randint(high=8,
                             size=tensor.size(),
                             dtype=tensor.dtype,
                             device=materialization_device)


@torch.no_grad()
def materialize_module_from_cpu(tensor, param_buf_key, cpu_model: torch.nn.Module,
                                materialization_device):
    cpu_params_buffers = {
        **dict(cpu_model.named_parameters()),
        **dict(cpu_model.named_buffers()),
    }
    if param_buf_key in cpu_params_buffers:
        return cpu_params_buffers[param_buf_key].to(device=materialization_device)

    return materialize_zero(tensor, materialization_device)


@torch.no_grad()
def materialize_module_from_set_parameter(tensor, param_buf_key, module: torch.nn.Module,
                                          materialization_device):
    tensor = fake_to_real(tensor)

    submodule_name_list = param_buf_key.split(".")

    if "reset_parameters" in module.__dir__():
        if param_buf_key in dict(module.named_parameters()):
            ori_parameters = module._parameters
            module._parameters = pytree.tree_map(fake_to_real, module._parameters)
            module.to_empty(device=materialization_device)
            module.reset_parameters()
            para = dict(module.named_parameters()).get(param_buf_key, None)
            module._parameters = ori_parameters
            return para

    if "reset_buffers" in module.__dir__():
        if param_buf_key in dict(module.named_buffers()):
            ori_buffers = module._buffers
            module._buffers = pytree.tree_map(fake_to_real, module._buffers)
            module.reset_buffers()
            buf = dict(module.named_buffers()).get(param_buf_key, None)
            module._buffers = ori_buffers
            return buf

    if len(submodule_name_list) == 1:
        return materialize_zero(tensor, materialization_device)

    submodule_name, sub_param_buf_key = submodule_name_list[0], ".".join(submodule_name_list[1:])

    if submodule_name in module._modules:
        return materialize_module_from_set_parameter(tensor, sub_param_buf_key,
                                                     module._modules[submodule_name],
                                                     materialization_device)

    return materialize_zero(tensor, materialization_device)


class InitHelper:

    def __init__(self) -> None:
        pass

    def get_materialize_fn(self):
        pass


class SetParaInitHelper:

    def __init__(self, module=None) -> None:
        self.module = module

    def get_materialize_fn(self):
        return partial(materialize_module_from_set_parameter, module=self.module)


class CpuModuleInitHelper:

    def __init__(self, cpu_module) -> None:
        self.cpu_module = cpu_module

    def get_materialize_fn(self):
        return partial(materialize_module_from_cpu, cpu_model=self.cpu_module)


def init_contiguous_buf(params, params_strategy, device_mesh):

    params = list(params.values())
    dtype = params[0].dtype
    device = params[0].device
    if not all(p.dtype == dtype for p in params):
        raise ValueError("All parameters must be of the same dtype.")
    if not all(p.device == device for p in params):
        raise ValueError("All parameters must be on the same device.")
    contiguous_buf_size = 0
    assert len(params_strategy) == len(params), "mismatch length of params_strategy and params"
    for p_s, p in zip(params_strategy, params):
        tensor_size = p.numel()
        for mesh_dim_idx, placement in enumerate(p_s):
            if placement.is_shard():
                tensor_size = (tensor_size // device_mesh.mesh.shape[mesh_dim_idx]) + 1

        contiguous_buf_size += tensor_size

    return torch.empty(contiguous_buf_size, dtype=dtype, device=device)
