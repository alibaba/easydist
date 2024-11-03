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

import hashlib
from contextlib import contextmanager
import copy
import numpy as np
import random
from typing import Any, Dict
from enum import Enum
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor
import torch.distributed.distributed_c10d as c10d
if torch.__version__ >= (2, 4):
    import torch.distributed.tensor as spmd
    from torch.distributed.tensor._utils import compute_local_shape
else:
    import torch.distributed._tensor as spmd
    from torch.distributed._tensor.ops.view_ops import compute_local_shape
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.utils._mode_utils import no_dispatch
import torch.utils._pytree as pytree
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._guards import detect_fake_mode

from easydist.metashard.combination import ReduceOp
from easydist.metashard import metair
from easydist.metashard.metair import NodeSPMDStrategy
from easydist.torch.device_mesh import get_device_mesh


def to_meta(node_output):
    if isinstance(node_output, FakeTensor):
        with no_dispatch():
            return torch.zeros_like(node_output, device="meta")
    if type(node_output) is torch.Tensor:
        return node_output.detach().to(device="meta").contiguous()
    elif type(node_output) is torch.nn.parameter.Parameter:
        return node_output.data.detach().to(device="meta").contiguous()
    else:
        return node_output


def create_meta_from_node(node):
    fake_args = pytree.tree_map_only(torch.fx.Node, lambda n: n.meta['val'], node.args)
    fake_mode = detect_fake_mode(fake_args)
    if fake_mode is not None:
        fake_args = pytree.tree_map_only(torch.Tensor, lambda n: fake_mode.from_tensor(n), fake_args)
    fake_val = node.target(*fake_args, **node.kwargs)
    if isinstance(fake_val, list) or isinstance(fake_val, tuple):
        return {'val': fake_val}
    return {'val': fake_val, 'tensor_meta': _extract_tensor_metadata(fake_val)}


def to_torch_spmd(meta_spmd: metair.SPMD):
    if meta_spmd.is_shard():
        return spmd.Shard(dim=meta_spmd.args["dim"])
    elif meta_spmd.is_partial():
        mapping_ops = {
            ReduceOp.SUM: c10d.ReduceOp.RedOpType.SUM,
            ReduceOp.MAX: c10d.ReduceOp.RedOpType.MAX,
            ReduceOp.MIN: c10d.ReduceOp.RedOpType.MIN,
            ReduceOp.AVG: c10d.ReduceOp.RedOpType.AVG,
        }
        return spmd.placement_types._Partial(reduce_op=mapping_ops[meta_spmd.args["ops"]])
    elif meta_spmd.is_replicate():
        return spmd.Replicate()


class EDNodeType(Enum):
    COMMUNICATION = 1
    COMPUTATION = 2
    AUXILIARY = 3


@dataclass
class EDInfo:
    node_type: EDNodeType = None
    ori_meta: Dict = None
    spmd_annotation: Any = None
    strategy: NodeSPMDStrategy = None
    runtime_ms: float = 0.0
    tiled_runtime_ms: Dict = None
    normalized_int_runtime_ms: int = 1
    comm_meta = None  # {comm_vol, comm_shape}

    def is_communication(self):
        return self.node_type == EDNodeType.COMMUNICATION

    def is_computation(self):
        return self.node_type == EDNodeType.COMPUTATION

    def get_sharded_meta(self):
        sharded_meta = copy.copy(self.ori_meta)

        if self.strategy is None:
            return sharded_meta

        spmd_mesh = get_device_mesh('spmd')

        if isinstance(sharded_meta['val'], torch.Tensor):
            global_out_shape = sharded_meta['val'].shape
            shard_out = [to_torch_spmd(i) for i in self.strategy.get_outvar_strtg(0)]
            local_out_shape = compute_local_shape(list(global_out_shape), spmd_mesh, shard_out)
            sharded_meta['val'] = torch.ops.aten.new_empty.default(sharded_meta['val'],
                                                                  local_out_shape)
            if 'tensor_meta' in sharded_meta:
                sharded_meta['tensor_meta'] = _extract_tensor_metadata(self.ori_meta['val'])

        if isinstance(sharded_meta['val'], tuple) or isinstance(sharded_meta['val'], list):
            sharded_meta['val'] = list(sharded_meta['val'])
            for idx in range(len(sharded_meta['val'])):
                if sharded_meta['val'][idx] is None:
                    continue
                global_out_shape = sharded_meta['val'][idx].shape
                shard_out = [to_torch_spmd(i) for i in self.strategy.get_outvar_strtg(idx)]
                local_out_shape = compute_local_shape(list(global_out_shape), spmd_mesh,
                                                      shard_out)
                sharded_meta['val'][idx] = torch.ops.aten.new_empty.default(
                    sharded_meta['val'][idx], local_out_shape)
            sharded_meta['val'] = tuple(sharded_meta['val'])

        return sharded_meta


@contextmanager
def _sharding_ann_env():

    ori_cudnn_enabled = torch.backends.cudnn.enabled
    ori_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    ori_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

    # (TODO) device failed when convolution_backward in sharding_discovery
    torch.backends.cudnn.enabled = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        yield
    finally:
        torch.backends.cudnn.enabled = ori_cudnn_enabled
        torch.backends.cuda.matmul.allow_tf32 = ori_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = ori_cudnn_allow_tf32


@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, torch.nn.Parameter],
):
    assert opt is not None

    # update opt.state with proxy tensors
    orig_states: Dict[str, Any] = copy.copy(opt.state)
    for n in named_states:
        # opt.state's key type is string, but optimizer uses Parameter as keys
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state.update(orig_states)


@contextmanager
def _enable_compile():
    # The return value of torch._utils.is_compiling changes optimizer behavior.
    # We need that function to return True to include optimizer in the graph.
    # See: https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true():
        return True

    enabled_funcs = [
        torch._utils.is_compiling,
        torch._dynamo.is_compiling,
    ]

    orig_func_codes = [f.__code__ for f in enabled_funcs]
    for f in enabled_funcs:
        f.__code__ = f_true.__code__

    try:
        yield
    finally:
        for f, orig_code in zip(enabled_funcs, orig_func_codes):
            f.__code__ = orig_code


def get_input_signature(*args, **kwargs):

    input_meta_str = pytree.tree_map(to_meta, [args, kwargs])
    return hashlib.sha256(repr(input_meta_str).encode("utf-8")).hexdigest()


def get_dtensor_spec(mesh, placements, shape=None, ndim=None):
    # In PyTorch < 2.1.0: DTensorSpec(mesh, placements, shape=None, ndim=None)
    # In PyTorch >= 2.1.0: DTensorSpec(mesh, placements)
    if "shape" in list(DTensorSpec.__dataclass_fields__.keys()):
        return DTensorSpec(mesh=mesh, placements=placements, shape=shape, ndim=ndim)
    return DTensorSpec(mesh=mesh, placements=tuple(placements))

def extract_tensor_meta_info(tensor: torch.Tensor):
    metadata = torch._utils.get_tensor_metadata(tensor)
    meta_info = (
        f"  metadata: {metadata}\n"
        f"  size: {tensor.size()}\n"
        f"  dtype: {tensor.dtype}\n"
        f"  dev: {tensor.device}\n"
        f"  requires grad: {tensor.requires_grad}\n"
        f"  is_cuda: {tensor.is_cuda}\n"
        f"  numel: {tensor.numel()}\n"
        f"  grad: {tensor.grad}\n"
        f"  storage offset: {tensor.storage_offset()}\n"
        f"  stride: {tensor.stride()}\n"
        f"  is_leaf: {tensor.is_leaf}\n"
        f"  is_contiguous: {tensor.is_contiguous()}\n"
        f"  is_sparse: {tensor.is_sparse}\n"
        f"  layout: {tensor.layout}\n"
    )

    return meta_info

def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for built-in Python
    random.seed(seed)
    # Set(seed) for each of the random number generators in python:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
