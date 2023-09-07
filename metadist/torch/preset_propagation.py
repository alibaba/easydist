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

import functools

import torch
from torch.fx.node import _get_qualified_name

from metadist.metashard import view_propagation
from metadist.metashard.annotation import ShardAnnotation, ShardDim
from metadist.metashard.combination import CombinationFunc, ReduceOp

from .device_mesh import device_mesh_world_size

aten = torch.ops.aten


class PresetMetaSPMD:
    op_rules = {}


def register_meta_spmd(op):

    def wrapper(impl):
        op_list = op if isinstance(op, list) else [op]
        for op_i in op_list:
            if isinstance(op_i, str):
                op_name = op_i
            else:
                op_name = _get_qualified_name(op_i)
            PresetMetaSPMD.op_rules[op_name] = impl
        return impl

    return wrapper


def preset_meta_spmd(meta_op, input_args=None):
    if isinstance(meta_op, str):
        args, kwargs = input_args
        return PresetMetaSPMD.op_rules[meta_op](args, kwargs)
    elif meta_op.name in PresetMetaSPMD.op_rules:
        if input_args is None:
            args, kwargs = meta_op.input_args
        else:
            args, kwargs = input_args
        return PresetMetaSPMD.op_rules[meta_op.name](args, kwargs)
    return None, None


@register_meta_spmd([
    "placeholder",
    "get_attr",
])
def meta_spmd_placeholder(args, kwargs):
    input_shape = args.shape
    output_shape = args.shape

    view_ann = view_propagation(input_shape, output_shape, world_size=device_mesh_world_size())
    return view_ann['sharding_ann'], view_ann['combination_ann']


@register_meta_spmd([
    aten.view.default,
    aten._unsafe_view.default,
])
def meta_spmd_view(args, kwargs):
    input_shape = args[0].shape
    output_shape = list(args[1])

    view_ann = view_propagation(input_shape, output_shape, world_size=device_mesh_world_size())
    return view_ann['sharding_ann'], view_ann['combination_ann']


@register_meta_spmd(aten._reshape_alias.default)
def meta_spmd_reshape(args, kwargs):
    input_shape = args[0].shape
    output_shape = list(args[1])
    stride = list(args[2])

    view_ann = view_propagation(input_shape, output_shape, world_size=device_mesh_world_size())
    return view_ann['sharding_ann'], view_ann['combination_ann']


# (TODO) here just replicate all optimizer computation
# need to generate some shard strategy here
@register_meta_spmd([
    aten._foreach_add.Scalar,
    aten._foreach_mul.Scalar,
    aten._foreach_sub.Scalar,
    aten._foreach_div.Scalar,
    aten._foreach_neg.default,
    aten._foreach_reciprocal.default,
    aten._foreach_sqrt.default,
])
def meta_foreach_one_tensor_list(args, kwargs):
    tensor_list = args[0]
    sharding_ann = ShardAnnotation([[[ShardDim.get_noshard_dim()] * len(tensor.shape)
                                     for tensor in tensor_list]])
    return sharding_ann, {}


@register_meta_spmd([
    aten._foreach_add.List,
    aten._foreach_mul.List,
    aten._foreach_sub.List,
    aten._foreach_div.List,
])
def meta_foreach_two_tensor_list(args, kwargs):
    tensor_1_list, tensor_2_list = args[0], args[1]
    sharding_ann = ShardAnnotation([[[ShardDim.get_noshard_dim()] * len(tensor.shape)
                                     for tensor in tensor_1_list],
                                    [[ShardDim.get_noshard_dim()] * len(tensor.shape)
                                     for tensor in tensor_2_list]])
    return sharding_ann, {}


@register_meta_spmd([
    aten._foreach_addcdiv.Scalar,
    aten._foreach_addcmul.Scalar,
])
def meta_foreach_three_tensor_list(args, kwargs):
    tensor_1_list, tensor_2_list, tensor_3_list = args[0], args[1], args[2]
    sharding_ann = ShardAnnotation([[[ShardDim.get_noshard_dim()] * len(tensor.shape)
                                     for tensor in tensor_1_list],
                                    [[ShardDim.get_noshard_dim()] * len(tensor.shape)
                                     for tensor in tensor_2_list],
                                    [[ShardDim.get_noshard_dim()] * len(tensor.shape)
                                     for tensor in tensor_3_list]])
    return sharding_ann, {}


@register_meta_spmd([aten.empty.memory_format, aten.zeros.default])
def meta_create_op(args, kwargs):
    tensor_shape = args[0]

    shard_idx = 1
    combine_ann = {}

    sharding_ann = ShardAnnotation([[ShardDim.get_noshard_dim() for _ in tensor_shape]])

    world_size = device_mesh_world_size()

    for dim_idx, dim_shape in enumerate(tensor_shape):
        if world_size <= dim_shape:
            sharding_ann[0][dim_idx] = ShardDim(shard_idx)
            combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=dim_idx)
            shard_idx += 1

    return sharding_ann, combine_ann


@register_meta_spmd(aten.cat.default)
def meta_cat(args, kwargs):
    tensorlist, dim = args[0:2]
    tensor_shape = tensorlist[0].shape

    shard_idx = 1
    sharding_ann = [ShardDim.get_noshard_dim() for _ in tensor_shape]
    combine_ann = {}

    world_size = device_mesh_world_size()

    for idx in range(len(tensor_shape)):
        if world_size <= tensor_shape[idx] and idx != dim:
            sharding_ann[idx] = ShardDim(shard_idx)
            combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=idx)
            shard_idx += 1

    sharding_ann = ShardAnnotation([sharding_ann for _ in tensorlist])

    return sharding_ann, combine_ann


@register_meta_spmd(aten._log_softmax_backward_data.default)
def meta_log_softmax_backward_data(args, kwargs):
    grad_output, output, dim = args[0:3]
    tensor_shape = grad_output.shape

    shard_idx = 1
    sharding_ann = [ShardDim.get_noshard_dim() for _ in tensor_shape]
    combine_ann = {}

    world_size = device_mesh_world_size()

    for idx in range(len(tensor_shape)):
        if world_size <= tensor_shape[idx] and idx != dim:
            sharding_ann[idx] = ShardDim(shard_idx)
            combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=idx)
            shard_idx += 1

    sharding_ann = ShardAnnotation([sharding_ann] * 2)

    return sharding_ann, combine_ann


@register_meta_spmd(aten.mm.default)
def meta_mm(args, kwargs):
    mat1, mat2 = args[0:2]
    shape_n, shape_m = mat1.shape
    shape_m, shape_p = mat2.shape

    sharding_ann = ShardAnnotation([[ShardDim.get_noshard_dim() for _ in range(2)]
                                    for _ in range(2)])
    shard_idx = 1
    combine_ann = {}

    world_size = device_mesh_world_size()

    if world_size <= shape_n:
        sharding_ann[0][0] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=0)
        shard_idx += 1

    if world_size <= shape_m:
        sharding_ann[0][1] = ShardDim(shard_idx)
        sharding_ann[1][0] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.reduce, ops=ReduceOp.SUM)
        shard_idx += 1

    if world_size <= shape_p:
        sharding_ann[1][1] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=1)
        shard_idx += 1

    return sharding_ann, combine_ann


@register_meta_spmd(aten.bmm.default)
def meta_bmm(args, kwargs):
    mat1, mat2 = args[0:2]
    shape_b_, shape_n, shape_m = mat1.shape
    shape_b, shape_m, shape_p = mat2.shape
    assert shape_b_ == shape_b

    sharding_ann = ShardAnnotation([[ShardDim.get_noshard_dim() for _ in range(3)]
                                    for _ in range(2)])
    shard_idx = 1
    combine_ann = {}

    world_size = device_mesh_world_size()

    if world_size <= shape_b:
        sharding_ann[0][0] = ShardDim(shard_idx)
        sharding_ann[1][0] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=0)
        shard_idx += 1

    if world_size <= shape_n:
        sharding_ann[0][1] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=1)
        shard_idx += 1

    if world_size <= shape_m:
        sharding_ann[0][2] = ShardDim(shard_idx)
        sharding_ann[1][1] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.reduce, ops=ReduceOp.SUM)
        shard_idx += 1

    if world_size <= shape_p:
        sharding_ann[1][2] = ShardDim(shard_idx)
        combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=2)
        shard_idx += 1

    return sharding_ann, combine_ann


@register_meta_spmd(aten.empty_like.default)
def meta_empty_like(args, kwargs):
    tensor_shape = args[0].shape

    world_size = device_mesh_world_size()

    sharding_ann = ShardAnnotation([[ShardDim.get_noshard_dim()] * len(tensor_shape)])
    shard_idx = 1
    combine_ann = {}
    for idx, shape in enumerate(tensor_shape):
        if world_size <= shape:
            sharding_ann[0][idx] = ShardDim(shard_idx)
            combine_ann[shard_idx] = functools.partial(CombinationFunc.gather, dim=idx)
            shard_idx += 1

    return sharding_ann, combine_ann


@register_meta_spmd(aten.native_group_norm.default)
def meta_group_norm(args, kwargs):
    tensor_shape, w_shape, b_shape = args[0].shape, args[1].shape, args[2].shape

    world_size = device_mesh_world_size()
    sharding_ann = ShardAnnotation([[ShardDim.get_noshard_dim()] * len(tensor_shape),
                                    [ShardDim.get_noshard_dim()] * len(w_shape),
                                    [ShardDim.get_noshard_dim()] * len(b_shape)])

    shard_idx = 1
    combine_ann = {}

    if world_size <= tensor_shape[0]:
        sharding_ann[0][0] = ShardDim(shard_idx)
        combine_ann[shard_idx] = [
            functools.partial(CombinationFunc.gather, dim=0),
            functools.partial(CombinationFunc.gather, dim=0),
            functools.partial(CombinationFunc.gather, dim=0)
        ]

    return sharding_ann, combine_ann
