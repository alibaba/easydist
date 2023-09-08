# Register sharding propagation rules for some ops. These rules are missed in community pytorch.
# refer to https://github.com/pytorch/pytorch/tree/main/torch/distributed/_tensor/ops.

import copy
import functools
import math
import operator
from functools import reduce
from typing import cast

import torch
from torch._meta_registrations import calc_conv_nd_return_shape
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (DTensorSpec, Replicate, Shard, _Partial)

aten = torch.ops.aten

extra_pointwise_op = [
    aten.silu.default, aten.leaky_relu.default, aten.leaky_relu_backward.default, aten.elu.default,
    aten.elu_backward.default, aten.fill.Scalar
]

for op in extra_pointwise_op:
    register_prop_rule(op)(pointwise_rule)


# from torch/distributed/_spmd/experimental_ops.py on pytorch main branch
# copy it because it is not ready in released pytorch
@register_prop_rule(aten.native_layer_norm.default)  # pyre-ignore
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    input, normalized_shape, weight, bias, eps = op_schema.args_schema
    assert isinstance(input, DTensorSpec)
    assert isinstance(normalized_shape, (tuple, list))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(p, Replicate) for p in weight.placements)
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(p, Replicate) for p in bias.placements)
    # only the left-most (non-normalized) dimensions of the input can be sharded
    batch_ndim = len(input.shape) - len(normalized_shape)
    assert all(
        isinstance(p, Replicate) or (isinstance(p, Shard) and p.dim < batch_ndim, )
        for p in input.placements)
    stats_spec = DTensorSpec(
        mesh=input.mesh,
        placements=input.placements,
        ndim=input.ndim,
        shape=torch.Size(s for (i, s) in enumerate(input.shape)),
    )
    return OutputSharding(output_spec=(input, stats_spec, stats_spec))


# from torch/distributed/_spmd/experimental_ops.py on pytorch main branch
# copy it because it is not ready in released pytorch
@register_prop_rule(aten.native_layer_norm_backward.default)  # pyre-ignore
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    (
        grad,
        input,
        normalized_shape,
        result1,
        result2,
        weight,
        bias,
        grad_input_mask,
    ) = op_schema.args_schema
    assert isinstance(grad, DTensorSpec)
    assert isinstance(grad_input_mask, (list, tuple))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in weight.placements)
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in bias.placements)

    if all(isinstance(s, Replicate) for s in grad.placements):
        weight_grad = (DTensorSpec(
            mesh=weight.mesh,
            placements=[Replicate()] * weight.mesh.ndim,
            shape=weight.shape,
        ) if weight else None)
        bias_grad = (DTensorSpec(
            mesh=bias.mesh,
            placements=[Replicate()] * bias.mesh.ndim,
            shape=bias.shape,
        ) if bias else None)
        return OutputSharding(
            # NOTE: type errors below are legit. This is because DTensor currently
            # doesn't support Optional return values. Need to be fixed in DTensor repo.
            output_spec=(
                grad if grad_input_mask[0] else None,
                weight_grad if grad_input_mask[1] else None,
                bias_grad if grad_input_mask[2] else None,
            ), )

    batch_ndim = len(input.shape) - len(normalized_shape)
    assert any(isinstance(s, Shard) and s.dim < batch_ndim
               for s in grad.placements), f"Got {grad.placements}"
    weight_grad = (DTensorSpec(
        mesh=weight.mesh,
        placements=[_Partial()] * weight.mesh.ndim,
        shape=weight.shape,
    ) if weight else None)
    bias_grad = (DTensorSpec(
        mesh=bias.mesh,
        placements=[_Partial()] * bias.mesh.ndim,
        shape=bias.shape,
    ) if bias else None)
    return OutputSharding(
        # NOTE: type errors below are legit. This is because DTensor currently
        # doesn't support Optional return values. Need to be fixed in DTensor repo.
        output_spec=(
            grad if grad_input_mask[0] else None,
            weight_grad if grad_input_mask[1] else None,
            bias_grad if grad_input_mask[2] else None,
        ), )


@register_prop_rule(aten.convolution.default)
def _prop_convolution_default(op_schema: OpSchema) -> OutputSharding:
    input, weight = op_schema.args_schema[0:2]
    stride, padding, dialation, transposed, output_padding, groups = op_schema.args_schema[-6:]
    assert isinstance(input, DTensorSpec)
    assert isinstance(weight, DTensorSpec)
    output_placements = [Replicate()] * input.mesh.ndim

    if len(output_padding) == 0 or all(v == 0 for v in output_padding):
        output_padding = None
    output_shape = calc_conv_nd_return_shape(input, weight, stride, padding, dialation, transposed,
                                             groups, output_padding)

    input_shape = input.shape
    for idx, s in enumerate(input.placements):
        # if shard on batch, height, witdh
        if isinstance(s, Shard) and s.dim in [0] + list(range(2, len(input_shape))):
            output_placements[idx] = Shard(dim=s.dim)

    for idx, s in enumerate(weight.placements):
        if isinstance(s, Shard) and s.dim == 0:
            output_placements[idx] = Shard(dim=1)
        s_input = input.placements[idx]
        if isinstance(s, Shard) and s.dim == 1 and isinstance(s_input, Shard) and s_input.dim == 1:
            output_placements[idx] = _Partial()

    output_spec = DTensorSpec(mesh=input.mesh,
                              placements=output_placements,
                              shape=torch.Size(output_shape))

    return OutputSharding(output_spec=output_spec)


@register_prop_rule(aten.convolution_backward.default)
def _prop_convolution_backward_default(op_schema: OpSchema) -> OutputSharding:
    (grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding,
     groups, output_mask) = op_schema.args_schema

    weight_placement = [Replicate()] * weight.mesh.ndim
    bias_placement = [Replicate()] * weight.mesh.ndim

    for idx, s in enumerate(input.placements):
        if isinstance(s, Shard) and s.dim == 0:
            weight_placement[idx] = _Partial()
            bias_placement[idx] = _Partial()
        else:
            weight_placement[idx] = weight.placements[idx]
            if isinstance(weight_placement[idx], Shard) and weight_placement[idx].dim == 0:
                bias_placement[idx] = Shard(dim=0)

    weight_grad = DTensorSpec(mesh=weight.mesh, placements=weight_placement, shape=weight.shape)

    bias_grad = DTensorSpec(mesh=weight.mesh,
                            placements=bias_placement,
                            shape=torch.Size(bias_sizes))

    return OutputSharding(output_spec=(
        input if output_mask[0] else None,
        weight_grad if output_mask[1] else None,
        bias_grad if output_mask[2] else None,
    ), )


@register_prop_rule(aten.native_batch_norm.default)
def _prop_batch_norm_default(op_schema: OpSchema) -> OutputSharding:
    input, weight, bias = op_schema.args_schema[0:3]
    running_mean, running_var = op_schema.args_schema[3:5]

    return OutputSharding(output_spec=(input, running_mean, running_var))


@register_prop_rule(aten.cudnn_batch_norm.default)
def _prop_batch_norm_default(op_schema: OpSchema) -> OutputSharding:
    input, weight, bias = op_schema.args_schema[0:3]
    running_mean, running_var = op_schema.args_schema[3:5]

    reserve_placement = [Replicate()] * weight.mesh.ndim

    reserve = DTensorSpec(mesh=weight.mesh, placements=reserve_placement, shape=torch.Size([0]))

    return OutputSharding(output_spec=(input, running_mean, running_var, reserve), )


@register_prop_rule(aten._native_batch_norm_legit_functional.default)
def _prop_batch_norm_legit_functional_default(op_schema: OpSchema) -> OutputSharding:

    input, weight, bias = op_schema.args_schema[0:3]
    running_mean, running_var = op_schema.args_schema[3:5]

    return OutputSharding(output_spec=(input, running_mean, running_var, running_mean,
                                       running_var), )


@register_prop_rule(aten.native_group_norm.default)
def _prop_native_group_norm(op_schema: OpSchema) -> OutputSharding:

    input, weight, bias, N, C, HxW, group, eps = op_schema.args_schema

    local_shape = input.local_shape
    flatten_inner_shape = functools.reduce((lambda x, y: x * y), local_shape[2:])

    op_schema.args_schema = input, weight, bias, local_shape[0], local_shape[
        1], flatten_inner_shape, group, eps

    assert isinstance(input, DTensorSpec)
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(p, Replicate) for p in weight.placements)
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(p, Replicate) for p in bias.placements)
    # only the batch (1st) dimensions of the input can be sharded
    batch_ndim = 1
    assert all(
        isinstance(p, Replicate) or (isinstance(p, Shard) and p.dim < batch_ndim, )
        for p in input.placements)

    stats_placement = [Replicate(), Replicate()]
    for idx, p in enumerate(input.placements):
        if isinstance(p, Shard) and p.dim < batch_ndim:
            stats_placement[idx] = copy.deepcopy(p)
    stats_spec = DTensorSpec(
        mesh=input.mesh,
        placements=stats_placement,
        ndim=input.mesh.ndim,
        shape=torch.Size((input.shape[0], group)),
    )
    return OutputSharding(output_spec=(input, stats_spec, stats_spec))


@register_prop_rule(aten.native_group_norm_backward.default)
def _prop_native_group_norm_backward(op_schema: OpSchema) -> OutputSharding:
    grad_output, input, mean, rstd, gamma, N, C, HxW, group, output_mask = op_schema.args_schema

    local_shape = input.local_shape

    op_schema.args_schema = grad_output, input, mean, rstd, gamma, local_shape[0], local_shape[
        1], reduce(operator.mul, local_shape[2:]), group, output_mask

    spec = Replicate()
    if isinstance(input.placements[0], Shard):
        spec = _Partial()
    d_gamma = DTensorSpec(mesh=gamma.mesh, placements=[spec] * gamma.mesh.ndim, shape=gamma.shape)

    d_bias = DTensorSpec(mesh=gamma.mesh, placements=[spec] * gamma.mesh.ndim, shape=gamma.shape)

    return OutputSharding(
        # NOTE: type errors below are legit. This is because DTensor currently
        # doesn't support Optional return values. Need to be fixed in DTensor repo.
        output_spec=(
            input if output_mask[0] else None,
            d_gamma if output_mask[1] else None,
            d_bias if output_mask[2] else None,
        ), )


@register_prop_rule(
    [aten.cudnn_batch_norm_backward.default, aten.native_batch_norm_backward.default])
def _prop_batch_norm_backward_default(op_schema: OpSchema) -> OutputSharding:
    input, grad_output, weight, running_mean, running_var = op_schema.args_schema[0:5]

    weight_placement = [Replicate()] * weight.mesh.ndim

    for idx, s in enumerate(weight.placements):
        grad_place = grad_output.placements[idx]
        if isinstance(s, Replicate) and isinstance(grad_place, Shard) and grad_place.dim == 0:
            weight_placement[idx] = _Partial()
        if isinstance(s, Shard):
            weight_placement[idx] = weight.placements[idx]

    weight_grad = DTensorSpec(mesh=weight.mesh, placements=weight_placement, shape=weight.shape)
    bias_grad = DTensorSpec(mesh=weight.mesh, placements=weight_placement, shape=weight.shape)

    return OutputSharding(
        # NOTE: type errors below are legit. This is because DTensor currently
        # doesn't support Optional return values. Need to be fixed in DTensor repo.
        output_spec=(
            input,
            weight_grad,
            bias_grad,
        ), )


@register_prop_rule(aten.max_pool2d_with_indices.default)
def _prop_max_pool2d_with_indices(op_schema: OpSchema) -> OutputSharding:
    input, kernel, stride = op_schema.args_schema[:3]
    if len(op_schema.args_schema) >= 4:
        padding = op_schema.args_schema[3]
    else:
        padding = [0, 0]

    output_shape = list(input.shape)

    for inner_dim in range(2):
        output_shape[2 + inner_dim] = int(
            math.floor(input.shape[2 + inner_dim] + 2 * padding[inner_dim] -
                       (kernel[inner_dim] - 1) - 1) / stride[inner_dim] + 1)

    output_spec = DTensorSpec(mesh=input.mesh,
                              placements=input.placements,
                              shape=torch.Size(output_shape))

    return OutputSharding(output_spec=(output_spec, input))


@register_prop_rule([
    aten.max_pool2d_with_indices_backward.default,
    aten.avg_pool3d_backward.default,
])
def _prop_max_pool2d_with_indices_backward(op_schema: OpSchema) -> OutputSharding:
    grad_output, input = op_schema.args_schema[0:2]

    return OutputSharding(output_spec=DTensorSpec(
        mesh=grad_output.mesh, placements=grad_output.placements, shape=input.shape))


@register_prop_rule(aten.constant_pad_nd.default)
def _prop_constant_pad_nd(op_schema: OpSchema) -> OutputSharding:
    input, pad, value = op_schema.args_schema

    input_sizes = input.shape
    l_inp = len(input_sizes)

    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    new_shape = list(input_sizes[:l_diff])

    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        new_shape.append(new_dim)

    output_spec = DTensorSpec(mesh=input.mesh,
                              placements=input.placements,
                              shape=torch.Size(new_shape))

    return OutputSharding(output_spec=output_spec)


@register_prop_rule(aten.embedding.default)
def _prop_embedding(op_schema: OpSchema) -> OutputSharding:
    weight_spec, inp_spec = op_schema.args_spec
    if any(placement.is_shard(0) for placement in weight_spec.placements):
        raise NotImplementedError(
            "DTensor does not support row-wise sharded embedding operation yet!")

    out_shape = inp_spec.shape + weight_spec.shape[1:]

    if all(placement.is_replicate()
           for placement in weight_spec.placements) and inp_spec.placements == [Shard(0)]:
        # Embedding table is replicated, input ids are sharded along batch
        # dimension. Output lookups should match input sharding spec in this case.
        return OutputSharding(output_spec=DTensorSpec(
            mesh=inp_spec.mesh, placements=inp_spec.placements, shape=out_shape))

    if all(placement.is_replicate() for placement in inp_spec.placements):
        weight_dim_map = weight_spec.dim_map
        output_dim_map = inp_spec.dim_map
        output_dim_map.append(weight_dim_map[1])
        return OutputSharding(output_spec=DTensorSpec.from_dim_map(
            inp_spec.mesh, output_dim_map, [], shape=out_shape))

    return OutputSharding(
        output_spec=None,
        schema_suggestions=[
            OpSchema(
                func_schema=op_schema.func_schema,
                args_schema=(
                    weight_spec,
                    DTensorSpec(mesh=inp_spec.mesh,
                                placements=[Replicate()] * len(inp_spec.placements),
                                shape=inp_spec.shape,
                                ndim=inp_spec.ndim),
                ),
                kwargs_schema=op_schema.kwargs_schema,
            )
        ],
    )


# from torch/distributed/_spmd/experimental_ops.py on pytorch main branch
# copy it because it is not ready in released pytorch
@register_prop_rule(aten.select.int)
def _prop_select(op_schema: OpSchema) -> OutputSharding:

    tensor, dim = op_schema.args_schema[:2]
    assert isinstance(tensor, DTensorSpec)
    assert isinstance(dim, int)
    placements = tensor.placements
    assert all(not p.is_shard(dim)
               for p in placements), "DTensor does not support select on sharded dimension."

    # select will remove one dimension, decrement dim of Shard placements by 1
    # if they are larger than dim.
    new_placements = []
    for p in placements:
        # Using isinstance instead of is_shard so that mypy won't complain
        # about accessing dim attribute.
        if isinstance(p, Shard) and p.dim > dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)

    new_shape = tensor.shape[:dim] + tensor.shape[dim + 1:]

    return OutputSharding(output_spec=DTensorSpec(
        mesh=tensor.mesh, placements=new_placements, shape=new_shape, ndim=tensor.ndim - 1))


@register_prop_rule(aten.slice_backward.default)
def _prop_slice_backward(op_schema: OpSchema) -> OutputSharding:
    grad, input_sizes = op_schema.args_schema[0:2]

    return OutputSharding(output_spec=DTensorSpec(
        mesh=grad.mesh, placements=grad.placements, shape=input_sizes, ndim=grad.ndim))


@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_add.Scalar,
        aten._foreach_div.Scalar,
        aten._foreach_mul.Scalar,
        aten._foreach_sub.Scalar,
    ])
def _prop__foreach_binop_scalar(op_schema: OpSchema) -> OutputSharding:
    self, scalar = op_schema.args_schema
    assert isinstance(self, list) and all(isinstance(s, DTensorSpec) for s in self)
    assert not isinstance(scalar, list)
    return OutputSharding(output_spec=self)


@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_add.List,
        aten._foreach_div.List,
        aten._foreach_mul.List,
    ])
def _prop__foreach_binop_list(op_schema: OpSchema) -> OutputSharding:
    self, other = op_schema.args_schema[:2]
    scalar = None if len(op_schema.args_schema) < 3 else op_schema.args_schema[2]
    assert isinstance(self, list) and all(
        isinstance(s, DTensorSpec) for s in self), f"Expect a List[DTensorSpec] but got {self}"
    assert isinstance(other, list) and all(
        isinstance(o, DTensorSpec) for o in other), f"Expect a List[DTensorSpec] but got {other}"
    assert len(self) == len(other), ("Two tensor lists must match in length, "
                                     f"but got {len(self)} and {len(other)}")

    # if any(s != o for s, o in zip(self, other)):
    #     # If DTensorSpec for the two operand do not match, suggest using
    #     # self's DTensorSpec. This will trigger allreduce if other is partial
    #     # and self is replicated.
    #     return OutputSharding(
    #         output_spec=None,
    #         schema_suggestions=[
    #             OpSchema(
    #                 func_schema=op_schema.func_schema,
    #                 args_schema=(self, self, scalar) if scalar else (self, self),
    #                 kwargs_schema=op_schema.kwargs_schema,
    #                 is_inplace=op_schema.is_inplace,
    #                 is_out_variant=op_schema.is_out_variant,
    #             )
    #         ],
    #     )
    # else:
    return OutputSharding(output_spec=self)


@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_addcdiv.Scalar,
        aten._foreach_addcmul.Scalar,
    ])
def _prop__foreach_addcop_scalar(op_schema: OpSchema):
    self, tensor1, tensor2 = op_schema.args_schema[:3]
    scalar = None if len(op_schema.args_schema) < 4 else op_schema.args_schema[3]
    assert isinstance(self, list) and all(isinstance(s, DTensorSpec) for s in self)
    assert isinstance(tensor1, list) and all(isinstance(s, DTensorSpec) for s in self)
    assert isinstance(tensor2, list) and all(isinstance(s, DTensorSpec) for s in self)
    if any(s != t1 or s != t2 for s, t1, t2 in zip(self, tensor1, tensor2)):
        # If DTensorSpec for the two operand do not match, suggest using
        # self's DTensorSpec. This will trigger allreduce if other is partial
        # and self is replicated.
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(self, self, self, scalar) if scalar else (self, self, self),
                    kwargs_schema=op_schema.kwargs_schema,
                    is_inplace=op_schema.is_inplace,
                    is_out_variant=op_schema.is_out_variant,
                )
            ],
        )
    else:
        return OutputSharding(output_spec=self)


@register_prop_rule(  # pyre-ignore
    [
        aten._foreach_neg.default,
        aten._foreach_reciprocal.default,
        aten._foreach_sqrt.default,
    ])
def _prop__foreach_unaop(op_schema: OpSchema) -> OutputSharding:
    self = op_schema.args_schema[0]
    assert isinstance(self, list) and all(isinstance(s, DTensorSpec) for s in self)
    # FIXME(@mrshenli): for sqrt, this is only mathematically correct for
    # Replicate and Shard tensor.
    return OutputSharding(output_spec=self)


@register_prop_rule(aten.upsample_nearest2d.default)
def _prop_upsample_nearest2d(op_schema: OpSchema) -> OutputSharding:
    input, output_size = op_schema.args_schema[0:2]
    batch, channel = input.shape[:2]
    full_ouput_size = (batch, channel, *output_size)
    return OutputSharding(output_spec=DTensorSpec(
        mesh=input.mesh, placements=input.placements, shape=torch.Size(full_ouput_size)))


@register_prop_rule(aten.upsample_nearest2d_backward.default)
def _prop_upsample_nearest2d_backward(op_schema: OpSchema) -> OutputSharding:
    grad_output, output_size, input_size = op_schema.args_schema[0:3]
    return OutputSharding(output_spec=DTensorSpec(
        mesh=grad_output.mesh, placements=grad_output.placements, shape=torch.Size(input_size)))


@register_prop_rule(aten.tril.default)
def _prop_tril(op_schema: OpSchema) -> OutputSharding:
    self = op_schema.args_schema[0]

    assert isinstance(self, DTensorSpec)
    assert all(isinstance(p, Replicate) for p in self.placements)

    return OutputSharding(output_spec=self)


# (TODO) merge with _tensor/ops/math_ops.py aten._softmax.default
@register_prop_rule(aten._log_softmax.default)
def _prop_log_softmax_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec, softmax_dim, _ = op_schema.args_schema
    input_spec = cast(DTensorSpec, input_spec)
    softmax_dim = cast(int, softmax_dim)
    dim_map = input_spec.dim_map
    if softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0:
        raise RuntimeError("Cannot run softmax on sharding dimension!")
    return OutputSharding(input_spec)


# (TODO) merge with _tensor/ops/math_ops.py aten._softmax_backward_data.default
@register_prop_rule(aten._log_softmax_backward_data.default)
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    grad_out_spec, out_spec, softmax_dim, _ = op_schema.args_schema
    grad_out_spec = cast(DTensorSpec, grad_out_spec)
    out_spec = cast(DTensorSpec, out_spec)
    softmax_dim = cast(int, softmax_dim)
    grad_out_dim_map = grad_out_spec.dim_map
    out_dim_map = out_spec.dim_map
    if softmax_dim < len(grad_out_dim_map) and (grad_out_dim_map[softmax_dim] >= 0
                                                or out_dim_map[softmax_dim] >= 0):
        raise RuntimeError("Cannot run _softmax_backward_data on sharding dimension!")
    return pointwise_rule(op_schema)


@register_prop_rule(aten.nll_loss_forward.default)  # pyre-ignore
def _prop_nll_loss_forward(op_schema: OpSchema) -> OutputSharding:
    self, target = op_schema.args_schema[:2]
    assert isinstance(self, DTensorSpec)
    assert isinstance(target, DTensorSpec)

    out_placement = [Replicate()] * self.mesh.ndim

    for idx, s in enumerate(self.placements):
        if isinstance(s, Shard) and s.dim == 0:
            out_placement[idx] = _Partial()
        else:
            out_placement[idx] = self.placements[idx]

    return OutputSharding(output_spec=(
        DTensorSpec(mesh=self.mesh, placements=out_placement, shape=torch.Size([])),
        DTensorSpec(
            mesh=self.mesh, placements=[Replicate()] * self.mesh.ndim, shape=torch.Size([])),
    ))


@register_prop_rule(aten.nll_loss_backward.default)  # pyre-ignore
def _prop_nll_loss_backward(op_schema: OpSchema) -> OutputSharding:
    grad_output, self = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(self, DTensorSpec)
    return OutputSharding(output_spec=self)


@register_prop_rule(aten.avg_pool3d.default)
def _prop_avg_pool3d(op_schema: OpSchema) -> OutputSharding:
    input, kernel_size, stride = op_schema.args_schema[:3]

    padding = [0, 0, 0]
    if len(op_schema.args_schema) > 3:
        padding = op_schema.args_schema[3]

    output_shape = list(input.shape)

    for inner_dim in range(3):
        output_shape[2 + inner_dim] = int(
            math.floor(input.shape[2 + inner_dim] + 2 * padding[inner_dim] -
                       (kernel_size[inner_dim] - 1) - 1) / stride[inner_dim] + 1)

    return OutputSharding(output_spec=DTensorSpec(
        mesh=input.mesh, placements=input.placements, shape=torch.Size(output_shape)))


# from torch/distributed/_tensor/ops/random_ops.py on pytorch main branch
# copy it because it is not ready in released pytorch
@register_prop_rule(aten.native_dropout.default)
def dropout_rule(op_schema: OpSchema) -> OutputSharding:
    self_spec = cast(DTensorSpec, op_schema.args_schema[0])

    # NOTE: dropout on a partial tensor should be similar to the case of a replicate tensor
    partial = False
    for placement in self_spec.placements:
        if isinstance(placement, _Partial):
            partial = True
            break

    if partial:
        return OutputSharding(
            None,
            failed_reason="aten.native_dropout.default with _Partial is not supported yet!",
        )
    else:
        return OutputSharding(output_spec=(self_spec, self_spec))


@register_prop_rule([aten.max.dim, aten.min.dim])
def reduce_dim_rule(op_schema: OpSchema) -> OutputSharding:
    self, dim = op_schema.args_schema[:2]
    keepdim = False
    if len(op_schema.args_schema) == 3:
        keepdim = op_schema.args_schema[2]

    self_shape, self_placements = list(self.shape), copy.deepcopy(self.placements)
    if keepdim:
        self_shape[dim] = 1
    else:
        del self_shape[dim]

    out_spec = DTensorSpec(mesh=self.mesh,
                           placements=self_placements,
                           shape=self_shape,
                           ndim=self.ndim - 1)

    return OutputSharding(output_spec=(out_spec, out_spec))


@register_prop_rule([
    aten.index_put.default,
    aten.index_put_.default,
])
def reduce_dim_rule(op_schema: OpSchema) -> OutputSharding:
    self = op_schema.args_schema[0]
    return OutputSharding(output_spec=self)


@register_prop_rule(aten._scaled_dot_product_efficient_attention.default)
def scaled_dot_product_efficient_attention_rule(op_schema: OpSchema) -> OutputSharding:
    query, key, value = op_schema.args_schema[0:3]

    out_placements = query.placements
    dim_size = len(value.shape)
    for idx, s in enumerate(value.placements):
        if isinstance(s, Shard) and s.dim == dim_size - 1:
            out_placements[idx] = Shard(s.dim)

    out_spec = DTensorSpec(mesh=query.mesh,
                           placements=out_placements,
                           shape=torch.Size(list(query.shape[:-1]) + list(value.shape[-1:])),
                           ndim=query.ndim)

    state_spec = DTensorSpec(mesh=query.mesh,
                             placements=[Replicate()] * query.mesh.ndim,
                             shape=torch.Size(list(query.shape[:-2]) + [0]),
                             ndim=query.ndim - 1)

    return OutputSharding(output_spec=(out_spec, state_spec))


@register_prop_rule(aten.masked_fill_.Scalar)
def masked_fill_rule(op_schema: OpSchema) -> OutputSharding:
    tensor = op_schema.args_schema[0]
    return OutputSharding(output_spec=tensor)


# from torch/distributed/_tensor/ops/embedding_ops.py on pytorch main branch
# copy it because it is not ready in released pytorch
@register_prop_rule(aten.embedding_dense_backward.default)
def embedding_dense_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output, indices, num_weights = op_schema.args_schema[:3]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(indices, DTensorSpec)

    output_shape = (num_weights, ) + grad_output.shape[len(indices.shape):]
    output_placements = []

    for idx in range(len(grad_output.placements)):
        if grad_output.placements[idx] == indices.placements[idx]:
            if isinstance(grad_output.placements[idx], Replicate):
                output_placements.append(Replicate())
            else:
                output_placements.append(_Partial())
        elif grad_output.placements == [_Partial()] and indices.placements == [Replicate()]:
            # The embedding table is replicated and the indices is also replicated
            # (local is a more precise term). This is postional embedding. In this
            # case, gradients for the embmedding table should be Partial.
            output_placements.append(_Partial())
        else:
            raise NotImplementedError("Unsupported embedding dense backward schema:\n"
                                      f"grad_output - {grad_output}\n"
                                      f"indices - {indices}")

    return OutputSharding(output_spec=DTensorSpec(
        mesh=indices.mesh, shape=output_shape, placements=output_placements))


# from torch/distributed/_spmd/experimental_ops.py on pytorch main branch
# copy it because it is not ready in released pytorch
@register_prop_rule(aten.stack.default)
def _prop_stack(op_schema: OpSchema) -> OutputSharding:
    tensors = op_schema.args_schema[0]
    dim = 0 if len(op_schema.args_schema) == 1 else cast(int, op_schema.args_schema[1])
    assert (isinstance(tensors, list) and len(tensors) > 0), "expect at least one tensor to stack"
    assert all(isinstance(t, DTensorSpec)
               for t in tensors), f"expect a list of DTensorSpecs, but got {tensors}"
    assert all(t.shape == tensors[0].shape
               for t in tensors), f"expect all tensors to have the same shape, but got {tensors}."
    # TODO: provide schema_suggestions when placements do not match
    assert all(
        t.placements == tensors[0].placements
        for t in tensors), f"expect all tensors to have the same placements, but got {tensors}."
    assert all(
        not p.is_shard(dim)
        for p in tensors[0].placements), "DTensor does not support stack on sharded dimension."

    concat_placements = copy.deepcopy(tensors[0].placements)

    for idx, s in enumerate(concat_placements):
        if isinstance(s, Shard) and s.dim >= dim:
            concat_placements[idx] = Shard(s.dim + 1)

    output_shape = list(tensors[0].shape)
    output_shape.insert(dim, len(tensors))

    return OutputSharding(output_spec=DTensorSpec(
        mesh=tensors[0].mesh, shape=output_shape, placements=tensors[0].placements))
