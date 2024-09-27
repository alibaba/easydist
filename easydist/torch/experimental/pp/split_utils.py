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

from contextlib import nullcontext
from typing import List, Union, Tuple, Any, Sequence, Optional, cast

import torch
import torch._custom_ops

from easydist.torch.utils import _rematerialize_optimizer
import torch.utils._pytree as pytree
from torch.fx._symbolic_trace import _Patcher
from torch.nn.utils import stateless
from easydist.torch.split_utils import (
    list_before_split,
    list_after_split,
    _before_split,
    _after_split,
)

'''
The valid parameters types are: 
dict_keys([
    <class 'torch.Tensor'>, 
    typing.Optional[torch.Tensor], 
    typing.Sequence[torch.Tensor], 
    typing.Sequence[typing.Optional[torch.Tensor]],
    <class 'int'>, 
    typing.Optional[int], 
    typing.Sequence[int], 
    typing.Optional[typing.Sequence[int]], 
    <class 'float'>, 
    typing.Optional[float], 
    typing.Sequence[float], 
    typing.Optional[typing.Sequence[float]], 
    <class 'bool'>, 
    typing.Optional[bool], 
    typing.Sequence[bool], 
    typing.Optional[typing.Sequence[bool]], 
    <class 'str'>, 
    typing.Optional[str], 
    typing.Union[int, float, bool], 
    typing.Union[int, float, bool, NoneType], 
    typing.Sequence[typing.Union[int, float, bool]],
    <class 'torch.dtype'>, 
    typing.Optional[torch.dtype], 
    <class 'torch.device'>, 
    typing.Optional[torch.device]]
    )
'''
op_param_type = Sequence[Optional[torch.Tensor]]
op_ret_type = List[torch.Tensor]


@torch._custom_ops.custom_op("easydist::fw_bw_split")
def fw_bw_split(args: op_param_type) -> op_ret_type:
    ...


@torch._custom_ops.impl_abstract("easydist::fw_bw_split")
def fw_bw_split_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


@torch._custom_ops.impl("easydist::fw_bw_split")
def fw_bw_split_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class FWBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):  # tensors must be passed as args
        tensor_list = list(tensor_tuple)  # custom op requires list
        return tuple(torch.ops.easydist.fw_bw_split(
            tensor_list))  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, *grads):
        return tuple(torch.ops.easydist.fw_bw_split(grads))


def split_func_with_bw(tensor_list: op_param_type) -> op_ret_type:
    assert not get_backward_flag() or all(
        t.requires_grad for t in tensor_list
    ), "Only split on tensors that need grad, otherwise backward pass won't be tracked"
    return FWBWSplitFunc.apply(*tensor_list)


class BeforeBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):  # tensors must be passed as args
        tensor_list = list(tensor_tuple)  # custom op requires list
        return tuple(torch.ops.easydist.fw_bw_split(
            tensor_list))  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, grad):
        return grad  # return must be tensor of tuple of tensors


def split_func_without_bw(tensor_list: op_param_type) -> op_ret_type:
    assert not get_backward_flag() or all(
        t.requires_grad for t in tensor_list
    ), "Only split on tensors that need grad, otherwise backward pass won't be tracked"
    return BeforeBWSplitFunc.apply(*tensor_list)


@torch._custom_ops.custom_op("easydist::step_split")
def step_split(args: op_param_type) -> op_ret_type:
    ...


@torch._custom_ops.impl_abstract("easydist::step_split")
def step_split_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


@torch._custom_ops.impl("easydist::step_split")
def step_split_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class StepSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):  # tensors must be passed as args
        tensor_list = list(tensor_tuple)
        return tuple(torch.ops.easydist.step_split(
            tensor_list))  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, *grads):
        return grads  # return must be tensor of tuple of tensors


def split_func_optimizier_step(tensor_list: op_param_type) -> op_ret_type:
    ret = StepSplit.apply(*tensor_list)
    return ret


ANNOTATION_OPS = [torch.ops.easydist.fw_bw_split.default, torch.ops.easydist.step_split.default]

__updated_params_states = None, None
__backward_flag = False
__step_flag = False


def get_updated_params_states():
    global __updated_params_states
    return __updated_params_states


def set_updated_params_states(updated_params, updated_states):
    global __updated_params_states
    __updated_params_states = updated_params, updated_states


def get_backward_flag():
    global __backward_flag
    return __backward_flag


def set_backward_flag(flag):
    global __backward_flag
    __backward_flag = flag


def get_step_flag():
    global __step_flag
    return __step_flag


def set_step_flag(flag):
    global __step_flag
    __step_flag = flag


def clear_pp_compile_states():
    set_backward_flag(False)
    set_updated_params_states(None, None)
    set_step_flag(False)


def get_registered_by_mro(registered, cls_begin: type) -> type:
    for cls in cls_begin.mro():
        if cls in registered:
            return registered[cls]
    raise RuntimeError(f"no registered split function for {cls_begin}")


def split(ret):
    cls_ret = type(ret)
    ctx = {}
    tensor_tuple: Tuple[torch.Tensor] = get_registered_by_mro(_before_split, cls_ret)(ctx, ret)
    tensor_tuple_after_split: Tuple[torch.Tensor] = split_func_with_bw(tensor_tuple)
    ret = get_registered_by_mro(_after_split, cls_ret)(ctx, tensor_tuple_after_split)
    return ret

class SplitPatcher(_Patcher):

    def __init__(self, module: torch.nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.module = module
        self.optimizer = optimizer

    def __enter__(self):
        patcher = super().__enter__()

        orig_backward = torch.Tensor.backward

        def backward_wrapper(self,
                             gradient=None,
                             retain_graph: bool = None,
                             create_graph: bool = False,
                             inputs: Optional[Sequence[torch.Tensor]] = None):
            tensor_list = [self] + (inputs or [])
            tensor_list = split_func_without_bw(tensor_list)
            self, inputs = tensor_list[0], tensor_list[1:]
            if len(inputs) == 0:
                inputs = None
            orig_backward(self, gradient, retain_graph, create_graph, inputs)
            set_backward_flag(True)

        patcher.patch_method(torch.Tensor, 'backward', backward_wrapper, deduplicate=False)

        if self.module:
            mod_cls = type(self.module)
            orig_forward = mod_cls.forward

            def forward_wrapper(module, *args, **kwargs):
                ret = orig_forward(module, *args, **kwargs)
                return ret

            patcher.patch_method(mod_cls, 'forward', forward_wrapper, deduplicate=False)

        if self.optimizer:
            opt_cls = type(self.optimizer)
            orig_step = opt_cls.step

            def step_wrapper(optimizer, *args, **kwargs):
                params = dict(self.module.named_parameters()) if self.module else {}
                grads = {n: p.grad for n, p in params.items() if p.grad is not None}
                named_states = {}
                for n, p in params.items():
                    if p in self.optimizer.state:
                        named_states[n] = self.optimizer.state[p]

                states, spec = pytree.tree_flatten((params, grads, named_states))

                ctx = {}
                states = list_before_split(ctx, states)
                states = split_func_optimizier_step(states)
                states = list_after_split(ctx, states)
                params, split_grads, named_states = pytree.tree_unflatten(states, spec)

                for n, p in params.items():  # need to split on grads
                    p.grad = split_grads[n]

                with stateless._reparametrize_module(
                        cast(torch.nn.Module, self.module), params, tie_weights=True) if self.module else nullcontext(), _rematerialize_optimizer(
                            optimizer, named_states, params) if optimizer else nullcontext():
                    orig_step(optimizer, *args, **kwargs)

                set_updated_params_states(params, named_states)
                set_step_flag(True)

            patcher.patch_method(opt_cls, 'step', step_wrapper, deduplicate=False)

        return patcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

if __name__ == '__main__':
    a = torch.rand(10, 10, requires_grad=True)
    b = torch.rand(3, 10, 10, requires_grad=True)
    res = split_func_with_bw([a, b])
    print(res[0].requires_grad, res[1].requires_grad)
    (res[0].mean() + res[1].mean()).backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = split_func_without_bw(a)
    print(res.requires_grad)
    res.mean().backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = split_func_optimizier_step([a])[0]
    print(res.requires_grad)
    res.mean().backward()
