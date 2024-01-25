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

import torch
import torch._custom_ops
import typing

param_type = typing.Sequence[typing.Optional[torch.Tensor]]
ret_type = typing.List[torch.Tensor]


@torch._custom_ops.custom_op("easydist::fw_bw_split")
def fw_bw_split(args: param_type) -> ret_type:
    ...


@torch._custom_ops.impl_abstract("easydist::fw_bw_split")
def bias_gelu_back_impl_abstract(args: param_type) -> ret_type:
    need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
    args = [arg.clone() if need_clone(arg) else arg for arg in args]
    return args


@torch._custom_ops.impl("easydist::fw_bw_split")
def fw_bw_split_impl(args: param_type) -> ret_type:
    need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
    args = [arg.clone() if need_clone(arg) else arg for arg in args]
    return args


class FWBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            *args):  # TODO @botbw: support kwargs: https://github.com/pytorch/pytorch/issues/96337
        args = list(args)
        return tuple(torch.ops.easydist.fw_bw_split(args))

    @staticmethod
    def backward(ctx, *grad_output):
        return tuple(torch.ops.easydist.fw_bw_split(grad_output))


def fw_bw_split_func(*args, **kwargs):
    if len(kwargs):
        raise TypeError(
            "fw_bw_split_func() got an unexpected keyword argument '%s', autograd.Function haven't support kwargs yet, try SplitPoint.END to solve this"
            % list(kwargs.keys()))
    return FWBWSplitFunc.apply(*args)


if __name__ == '__main__':
    a = torch.rand(10, 10, requires_grad=True)
    b = torch.rand(3, 10, 10, requires_grad=True)
    res = fw_bw_split_func(a, b)
    (res[0].mean() + res[1].mean()).backward()
    print(res)
