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

import os

import torch
from torch._functorch import config
from torch.distributed._tensor.op_schema import OpSchema

from .api import easydist_compile
from .bridge import torch2meta_graph
from .passes.sharding import sharding_transform
from .device_mesh import set_device_mesh, get_device_mesh
from .sharding_interpreter import EDTorchShardingAnn
from .spmd_prop_rule import *
from easydist.torch.init_meta_allocator import init_meta_allocator

# disable with torch <= 2.0.1
if hasattr(config, "use_fake_tensor"):
    config.use_fake_tensor = False

__all__ = [
    'EDTorchShardingAnn', 'sharding_transform', 'set_device_mesh', 'get_device_mesh',
    'torch2meta_graph', 'easydist_compile'
]


def easydist_setup_torch(device, allow_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # NCCL_ASYNC_ERROR_HANDLING is deprecated, use TORCH_NCCL_ASYNC_ERROR_HANDLING from torch 2.2.0
    if torch.__version__ <= (2, 2):
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    else:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

    # this env var enforces the order of kernel execution on GPU as the kernel queuing order from host.
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # initialize profiling allocator
    init_meta_allocator()


# hotfix for PyTorch 2.1.0
if "2.1.0" in torch.__version__:
    # ========================================================================
    def hot_fix_opschema_hash(self):
        # NOTE: we turn kwargs_schema into a frozenset to hash as it would not be nested dict
        frozen_set_kwargs_schema = frozenset(self.kwargs_schema.items())
        return hash((
            self.func_schema.name,
            tuple(self.func_schema.arguments),
            tuple(self.func_schema.returns),
            self.func_schema.name,
            tuple(tuple(e) if isinstance(e, list) else e for e in self.args_schema),
            frozen_set_kwargs_schema,
        ))

    OpSchema.__hash__ = hot_fix_opschema_hash

    # ========================================================================
    # Fix: AttributeError: 'DTensorSpec' object has no attribute 'local_shape'
    # In spmd_prop_rule.py::_prop_native_group_norm

    from torch.distributed._tensor.placement_types import DTensorSpec
    from typing import Tuple

    # Borrow from https://github.com/pytorch/pytorch/blob/e9ebda29d87ce0916ab08c06ab26fd3766a870e5/torch/distributed/_tensor/placement_types.py#L367
    # which was removed in 2.1.0
    def local_shape(self) -> Tuple[int, ...]:
        """
        Compute the shape of a local shard of the given DTensor on its current
        coordinate of the mesh.
        """
        assert self.shape is not None, "DTensorSpec does not contain global shape."
        local_shape = list(self.shape)  # start with global shape
        for idx, placement in enumerate(self.placements):
            mesh_dim_size = self.mesh.size(idx)
            my_coordinate = self.mesh.get_coordinate_on_dim(idx)
            assert my_coordinate is not None, "Rank not part of mesh!"
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert (
                    shard_dim
                    < self.ndim), f"Sharding dim {shard_dim} greater than tensor ndim {self.ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate)
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size
        return tuple(local_shape)

    assert not hasattr(DTensorSpec, "local_shape")
    setattr(DTensorSpec, "local_shape", property(local_shape))

