# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import torch
import torch._custom_ops
from easydist.torch.experimental.pp.compile_pipeline import get_backward_flag
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
op_param_type = typing.Sequence[typing.Optional[torch.Tensor]]
op_ret_type = typing.List[torch.Tensor]


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


def fw_bw_split_func(tensor_list: op_param_type) -> op_ret_type:
    assert not get_backward_flag() or all(
        t.requires_grad for t in tensor_list
    ), "Only split on tensors that need grad, otherwise backward pass won't be tracked"
    return FWBWSplitFunc.apply(*tensor_list)


class BeforeBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):  # tensors must be passed as args
        return torch.ops.easydist.fw_bw_split(
            [tensor])[0]  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, grad):
        return grad  # return must be tensor of tuple of tensors


def before_bw_split_func(tensor: torch.Tensor) -> torch.Tensor:
    assert not get_backward_flag(
    ) or tensor.requires_grad, "Only split on tensors that need grad, otherwise backward pass won't be tracked"
    ret = BeforeBWSplitFunc.apply(tensor)
    return ret


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


def step_split_func(tensor_list: op_param_type) -> op_ret_type:
    ret = StepSplit.apply(*tensor_list)
    return ret


if __name__ == '__main__':
    a = torch.rand(10, 10, requires_grad=True)
    b = torch.rand(3, 10, 10, requires_grad=True)
    res = fw_bw_split_func([a, b])
    print(res[0].requires_grad, res[1].requires_grad)
    (res[0].mean() + res[1].mean()).backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = before_bw_split_func(a)
    print(res.requires_grad)
    res.mean().backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = step_split_func([a])[0]
    print(res.requires_grad)
    res.mean().backward()
