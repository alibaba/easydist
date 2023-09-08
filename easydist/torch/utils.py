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
from copy import copy
from typing import Any, Dict

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._mode_utils import no_dispatch
import torch.utils._pytree as pytree


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
    orig_states: Dict[str, Any] = copy(opt.state)
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

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


def get_input_signature(*args, **kwargs):

    input_meta_str = pytree.tree_map(to_meta, [args, kwargs])
    return hashlib.sha256(repr(input_meta_str).encode("utf-8")).hexdigest()
