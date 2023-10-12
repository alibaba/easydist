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

from .api import compile, enable_transform, get_input_strategy, easydist_shard
from .bridge import shard_module, torch2meta_graph
from .passes.sharding import sharding_transform
from .device_mesh import set_device_mesh, get_device_mesh
from .sharding_interpreter import EDTorchShardingAnn
from .spmd_prop_rule import *

# disable with torch <= 2.0.1
if hasattr(config, "use_fake_tensor"):
    config.use_fake_tensor = False

__all__ = [
    'EDTorchShardingAnn', 'sharding_transform', 'set_device_mesh', 'get_device_mesh',
    'torch2meta_graph', 'compile', 'easydist_shard', 'enable_transform', 'shard_module',
    'get_input_strategy'
]


def easydist_setup_torch(device, allow_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # this environment used for cuda graph
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"


# hotfix for PyTorch 2.1.0
if "2.1.0" in torch.__version__:
    def hot_fix_opschema_hash(self):
        # NOTE: we turn kwargs_schema into a frozenset to hash as it would not be nested dict
        frozen_set_kwargs_schema = frozenset(self.kwargs_schema.items())
        return hash(
            (
                self.func_schema.name,
                tuple(self.func_schema.arguments),
                tuple(self.func_schema.returns),
                self.func_schema.name,
                tuple(tuple(e) if isinstance(e, list) else e for e in self.args_schema),
                frozen_set_kwargs_schema,
            )
        )

    OpSchema.__hash__ = hot_fix_opschema_hash
