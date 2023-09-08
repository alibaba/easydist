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
