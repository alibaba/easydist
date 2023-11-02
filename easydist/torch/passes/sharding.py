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

import operator
import logging
from typing import List

import torch
import torch.utils._pytree as pytree
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed._tensor import mesh_resources
from torch.distributed._functional_collectives import _expand_group
from torch.fx.node import Node, _get_qualified_name, map_arg
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.distributed._tensor.ops.view_ops import (view_groups, normalize_sizes, expand,
                                                    propagate_shape_and_sharding,
                                                    compute_local_shape)

from easydist.torch.device_mesh import device_mesh_rank, get_device_mesh
from easydist.torch.utils import to_torch_spmd, EDInfo, EDNodeType
from easydist.utils.testing import MockDeviceMesh
from easydist.metashard.metair import VarSPMDStrategy, SPMD
from easydist.metashard.combination import ReduceOp

logger = logging.getLogger(__name__)

reduce_map = {
    ReduceOp.SUM: "sum",
    ReduceOp.MAX: "max",
    ReduceOp.MIN: "min",
    ReduceOp.AVG: "avg",
}

CREATE_ATEN_OP = [
    torch.ops.aten.empty.memory_format, torch.ops.aten.zeros.default, torch.ops.aten.ones.default,
    torch.ops.aten.scalar_tensor.default, torch.ops.aten.arange.default,
    torch.ops.aten.zeros.default
]


def all_reduce_start(self: torch.Tensor, reduceOp: str, group: List[int], tag: str = ""):
    tag, rankset, group_size = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_reduce(self, reduceOp, tag, rankset,
                                                  group_size)  # type: ignore[attr-defined]
    return tensor


def all_reduce_end(self: torch.Tensor, reduceOp: str, group: List[int], tag: str = ""):
    return torch.ops.c10d_functional.wait_tensor(self)


def all_gather_start(self: torch.Tensor, gather_dim: int, group: List[int], tag: str = ""):
    if not self.is_contiguous():
        self = self.contiguous()
    tag, rankset, group_size = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_gather_into_tensor(
        self, tag, rankset, group_size)  # type: ignore[attr-defined]
    return tensor


def all_gather_end(self: torch.Tensor, gather_dim: int, group: List[int], tag: str = ""):
    self = torch.ops.c10d_functional.wait_tensor(self)
    tag, rankset, group_size = _expand_group(group, tag)
    if gather_dim != 0:
        self = torch.cat(torch.chunk(self, group_size, dim=0), dim=gather_dim)
    return self


def scatter_wrapper(tensor, num_chunks, dim, indice):
    return torch.ops.aten.chunk(tensor, num_chunks, dim)[indice]


def reduce_scatter_start(self: torch.Tensor,
                         reduceOp: str,
                         scatter_dim: int,
                         group: List[int],
                         tag: str = ""):
    tag, rankset, group_size = _expand_group(group, tag)
    assert (self.size(scatter_dim) % group_size == 0
            ), f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size}"
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops.c10d_functional.reduce_scatter_tensor(
        self, reduceOp, tag, rankset, group_size)  # type: ignore[attr-defined]
    return tensor


def reduce_scatter_end(self: torch.Tensor,
                       reduceOp: str,
                       scatter_dim: int,
                       group: List[int],
                       tag: str = ""):
    return torch.ops.c10d_functional.wait_tensor(self)


def all_to_all_start(tensor, gather_dim, scatter_dim, num_chunks, indice, ranks, tag: str = ""):
    # (TODO) call all_gather for all_to_all, need to use all_to_all for less comm size
    gathered_tensor = all_gather_start(tensor, gather_dim, ranks, tag)
    return gathered_tensor


def all_to_all_end(tensor, gather_dim, scatter_dim, num_chunks, indice, ranks, tag: str = ""):
    tensor = all_gather_end(tensor, gather_dim, ranks, tag)
    return scatter_wrapper(tensor, num_chunks, scatter_dim, indice)


COMM_FUNCS = [all_reduce_start, all_gather_start, reduce_scatter_start, all_to_all_start]
COMM_SYNC_FUNCS = [all_reduce_end, all_gather_end, reduce_scatter_end, all_to_all_end]
CUSTOM_FUNCS = COMM_FUNCS + COMM_SYNC_FUNCS + [scatter_wrapper]


def materialize(x, device):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bool:
            return torch.rand(x.size(), dtype=torch.float, device=device) > 0.5
        elif torch.is_floating_point(x):
            return torch.rand(x.size(), dtype=x.dtype, device=device)
        else:
            return torch.randint(high=8, size=x.size(), dtype=x.dtype, device=device)
    return x


def redist_tensor_func(input_tensor, specs):
    if isinstance(input_tensor, DTensor) and input_tensor.size() != torch.Size([0]):
        device_mesh = get_device_mesh()
        if specs != input_tensor._spec.placements:
            return input_tensor.redistribute(device_mesh, specs).contiguous()
    return input_tensor.contiguous()


def insert_spmd_head(body):
    return ["from torch.distributed._tensor import Shard, Replicate\n", *body]


@torch.no_grad()
def to_dtensor(input_tensor, placements=None, global_size=None):
    if isinstance(input_tensor, torch.Tensor) and not isinstance(input_tensor, DTensor):
        device_mesh = get_device_mesh()
        if placements is None:
            placements = [Replicate()] * device_mesh.ndim
        if global_size is None:
            return DTensor.from_local(input_tensor, device_mesh, placements)
        else:
            return DTensor(input_tensor, device_mesh, placements, size=global_size)
    return input_tensor


def fix_in_gragh_tensor(fx_module: torch.fx.GraphModule, sharding_strategy):

    for node in fx_module.graph.nodes:
        if node.op == 'get_attr':
            with fx_module.graph.inserting_after(node):
                to_dtensor_node = fx_module.graph.call_function(to_dtensor, args=(node, ))

                node.replace_all_uses_with(to_dtensor_node)

                to_dtensor_node.update_arg(0, node)

        create_ops = [
            "torch.ops.aten.scalar_tensor.", "torch.ops.aten.ones.", "torch.ops.aten.empty.",
            "torch.ops.aten.arange.", "torch.ops.aten.zeros."
        ]

        if node.op == 'call_function':
            if node.target in CREATE_ATEN_OP:
                strategy, global_shape = None, None
                if node.name in sharding_strategy and len(sharding_strategy[node.name]) > 0:
                    strategy = sharding_strategy[node.name][0]
                    # modify the arg of creat ops
                    # (CAUTION!!) device_mesh maybe mock now,
                    # but the local shape need to be consistent of runtime
                    global_shape = node.args[0]
                    local_shape = list(global_shape)
                    device_mesh = get_device_mesh()
                    if isinstance(device_mesh, MockDeviceMesh):
                        logger.warning("maybe wrong shape in MockDeviceMesh for create ops.")
                    for idx, placement in enumerate(strategy):
                        if placement.is_shard():
                            shard_dim = placement.dim
                            split_size, pad_idx = divmod(global_shape[shard_dim],
                                                         device_mesh.size(idx))
                            local_shape[shard_dim] = split_size
                            if device_mesh_rank(device_mesh, idx) < pad_idx:
                                local_shape[shard_dim] = split_size + 1

                    node.update_arg(0, local_shape)

                with fx_module.graph.inserting_after(node):
                    to_dtensor_node = fx_module.graph.call_function(to_dtensor,
                                                                    args=(node, strategy,
                                                                          global_shape))

                    node.replace_all_uses_with(to_dtensor_node)

                    to_dtensor_node.update_arg(0, node)

    fx_module.recompile()

    return fx_module


def replace_subsequence_use(node, arg_, redist_node):
    users_node = list(node.users.keys())
    node_next = node.next

    def maybe_replace_node(n: Node) -> Node:
        if n == arg_:
            return redist_node
        else:
            return n

    while node_next.name != "":
        if node_next in users_node:
            new_args = map_arg(node_next.args, maybe_replace_node)
            new_kwargs = map_arg(node_next.kwargs, maybe_replace_node)
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            node_next.args = new_args
            node_next.kwargs = new_kwargs
        node_next = node_next.next


def sharding_transform_dtensor(fx_module: torch.fx.GraphModule, sharding_strategy):
    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if node.name in sharding_strategy:
                node_strategy = sharding_strategy[node.name]

                # skip for create ops
                if node.target in CREATE_ATEN_OP:
                    continue

                node_args_flatten = pytree.tree_flatten(node.args)[0]
                invars = [arg for arg in node_args_flatten if isinstance(arg, Node)]
                assert (len(invars) == len(node_strategy))

                for arg_, arg_strategy in zip(invars, node_strategy):
                    with fx_module.graph.inserting_before(node):
                        redist_node = fx_module.graph.call_function(redist_tensor_func,
                                                                    args=(arg_, arg_strategy))

                        # FIXME: may introduce redundancy communication, update_arg for all subsequence use
                        node.replace_input_with(arg_, redist_node)
                        #replace_subsequence_use(node, arg_, redist_node)

    fx_module.graph.on_generate_code(lambda _: insert_spmd_head)

    fx_module.recompile()

    # (fix) %_tensor_constant0 : [#users=1] = get_attr[target=_tensor_constant0]
    fx_module = fix_in_gragh_tensor(fx_module, sharding_strategy)

    return fx_module


def insert_comm_node(fx_module: torch.fx.GraphModule, node, var_, sorted_placements):

    device_mesh = get_device_mesh()

    for i, (current, target) in sorted_placements:
        my_coordinate = device_mesh.get_coordinate()
        num_chunks = device_mesh.size(dim=i)

        if current == target:
            continue

        if target.is_shard():
            if current.is_replicate():
                # insert split_tensor here
                with fx_module.graph.inserting_before(node):
                    scatter_node = fx_module.graph.call_function(scatter_wrapper,
                                                                 args=(var_, num_chunks,
                                                                       target.args["dim"],
                                                                       my_coordinate[i]))

                    node.replace_input_with(var_, scatter_node)
            elif current.is_shard():
                # all_to_all
                with fx_module.graph.inserting_before(node):
                    submesh = mesh_resources.create_child_mesh(device_mesh, i)
                    ranks = submesh.mesh.flatten().tolist()
                    all_to_all_start_node = fx_module.graph.call_function(
                        all_to_all_start,
                        args=(var_, current.args["dim"], target.args["dim"], num_chunks,
                              my_coordinate[i], ranks))
                    all_to_all_end_node = fx_module.graph.call_function(
                        all_to_all_end,
                        args=(all_to_all_start_node, current.args["dim"], target.args["dim"],
                              num_chunks, my_coordinate[i], ranks))

                    node.replace_input_with(var_, all_to_all_end_node)
            elif current.is_partial():
                # reduce_scatter
                with fx_module.graph.inserting_before(node):
                    submesh = mesh_resources.create_child_mesh(device_mesh, i)
                    ranks = submesh.mesh.flatten().tolist()
                    reduceOp = reduce_map[current.args["ops"]]
                    # make sure contiguous
                    reduce_scatter_start_node = fx_module.graph.call_function(
                        reduce_scatter_start, args=(var_, reduceOp, target.args["dim"], ranks))

                    reduce_scatter_end_node = fx_module.graph.call_function(
                        reduce_scatter_end,
                        args=(reduce_scatter_start_node, reduceOp, target.args["dim"], ranks))

                    node.replace_input_with(var_, reduce_scatter_end_node)
        elif target.is_replicate():
            if current.is_shard():
                # insert all_gather here
                with fx_module.graph.inserting_before(node):
                    submesh = mesh_resources.create_child_mesh(device_mesh, i)
                    ranks = submesh.mesh.flatten().tolist()
                    # make sure contiguous
                    all_gather_start_node = fx_module.graph.call_function(
                        all_gather_start, args=(var_, current.args["dim"], ranks))
                    all_gather_end_node = fx_module.graph.call_function(
                        all_gather_end, args=(all_gather_start_node, current.args["dim"], ranks))

                    node.replace_input_with(var_, all_gather_end_node)
            elif current.is_partial():
                # insert all_reduce here
                with fx_module.graph.inserting_before(node):
                    submesh = mesh_resources.create_child_mesh(device_mesh, i)
                    ranks = submesh.mesh.flatten().tolist()
                    reduceOp = reduce_map[current.args["ops"]]
                    all_reduce_start_node = fx_module.graph.call_function(all_reduce_start,
                                                                          args=(var_, reduceOp,
                                                                                ranks))

                    all_reduce_end_node = fx_module.graph.call_function(
                        all_reduce_end, args=(all_reduce_start_node, reduceOp, ranks))

                    node.replace_input_with(var_, all_reduce_end_node)

    return fx_module


# some of this part from torch/distributed/_tensor/ops/view_ops.py
def override_args(node, invars_strategy):
    device_mesh = get_device_mesh()

    view_op_map = {
        torch.ops.aten.view.default:
        lambda input_shape, shape: view_groups(input_shape, shape),
        torch.ops.aten._unsafe_view.default:
        lambda input_shape, shape: view_groups(input_shape, shape),
        torch.ops.aten.reshape.default:
        lambda input_shape, shape: view_groups(input_shape, shape),
        torch.ops.aten.expand.default:
        lambda input_shape, sizes: expand(input_shape, normalize_sizes(sizes)),
    }

    if node.target in view_op_map:
        global_in_shape = node.args[0].meta['val'].shape
        shape_argnum = 1

        input_dtensor_spec = [to_torch_spmd(i) for i in invars_strategy[0]]
        rules = view_op_map[node.target](global_in_shape, node.args[shape_argnum])

        (
            global_out_shape,
            shard_out,
            shardable_dims,
        ) = propagate_shape_and_sharding(
            input_dtensor_spec,
            tuple(global_in_shape),
            rules,
            tuple(device_mesh.mesh.shape),
        )

        local_out_shape = compute_local_shape(list(global_out_shape), device_mesh, shard_out)

        node.update_arg(shape_argnum, local_out_shape)


def create_meta_from_node(node):
    fake_args = []
    for arg in node.args:
        if isinstance(arg, Node):
            fake_args.append(arg.meta['val'])
        else:
            fake_args.append(arg)
    fake_val = node.target(*fake_args, **node.kwargs)
    return {'val': fake_val, 'tensor_meta': _extract_tensor_metadata(fake_val)}


def sharding_transform(fx_module: torch.fx.GraphModule, opt_strategy, state_io_map):

    shard_env = {}

    # (TODO) move this part out of this pass
    for node in fx_module.graph.nodes:
        if node.op == 'call_function' and 'val' not in node.meta:
            node.meta = create_meta_from_node(node)

    # the last element in fx_module output is the return tensor for user function
    # we need to make it replicate before
    num_return_value = fx_module._out_spec.children_specs[-1].num_leaves

    for node in fx_module.graph.nodes:
        if node.op == 'placeholder':
            if node.name in opt_strategy:
                shard_env[node.name] = opt_strategy[node.name]['strategy'].out_strtg_group[0]
            else:
                # TODO: only support 2d device mesh here
                shard_env[node.name] = VarSPMDStrategy(SPMD(SPMD.REPLICATE), SPMD(SPMD.REPLICATE))
        elif node.op == 'call_function':

            # skip for create ops
            ops_name = _get_qualified_name(node.target)
            if node.target in CREATE_ATEN_OP:
                # TODO: only support 2d device mesh here
                shard_env[node.name] = VarSPMDStrategy(SPMD(SPMD.REPLICATE), SPMD(SPMD.REPLICATE))
                continue

            if ops_name == "_operator.getitem":
                shard_env[node.name] = shard_env[node.args[0].name][node.args[1]]
                continue

            node_args_flatten = pytree.tree_flatten(node.args)[0]
            invars = [arg for arg in node_args_flatten if isinstance(arg, Node)]

            invars_strategy = opt_strategy[node.name]['strategy'].in_strtg_group

            override_args(node, invars_strategy)

            assert len(invars) == len(invars_strategy)

            for var_, tgt_specs in zip(invars, invars_strategy):
                src_specs = shard_env[var_.name]
                if tgt_specs != src_specs:
                    # mismatch need to insert communication node
                    sorted_placements = list(enumerate(zip(src_specs, tgt_specs)))
                    fx_module = insert_comm_node(fx_module, node, var_, sorted_placements)

            shard_env[node.name] = opt_strategy[node.name]['strategy'].out_strtg_group
            if len(shard_env[node.name]) == 1:
                shard_env[node.name] = shard_env[node.name][0]

        if node.op == 'output':
            need_replicate_node = node.args[0][-1 * num_return_value:]
            for o_node in need_replicate_node:
                src_specs = shard_env[o_node.name]
                tgt_specs = [SPMD(SPMD.REPLICATE)] * len(src_specs)
                tgt_specs = VarSPMDStrategy(*tgt_specs)
                if tgt_specs != src_specs:
                    # mismatch need to insert communication node
                    sorted_placements = list(enumerate(zip(src_specs, tgt_specs)))
                    fx_module = insert_comm_node(fx_module, node, o_node, sorted_placements)

            for in_node, out_node in state_io_map.items():
                if in_node.name in shard_env:
                    src_specs = shard_env[out_node.name]
                    tgt_specs = shard_env[in_node.name]
                    o_node = None
                    for node_ in node.args[0]:
                        if node_ is not None and node_.name == out_node.name:
                            o_node = node_
                            break
                    assert o_node is not None
                    if tgt_specs != src_specs:
                        # mismatch need to insert communication node
                        sorted_placements = list(enumerate(zip(src_specs, tgt_specs)))
                        fx_module = insert_comm_node(fx_module, node, o_node, sorted_placements)

    fx_module.recompile()

    # (TODO) move this part out of this pass
    for node in fx_module.graph.nodes:
        if not hasattr(node, "ed_info"):
            node.ed_info = EDInfo(ori_meta=node.meta)

        if node.name in opt_strategy:
            node.ed_info.strategy = opt_strategy[node.name]['strategy']

        node.meta = node.ed_info.get_sharded_meta()

        if node.op == 'placeholder':
            node.ed_info.node_type = EDNodeType.AUXILIARY
        elif node.op == 'call_function':
            # create meta for custom function
            if node.target in CUSTOM_FUNCS:
                node.meta = create_meta_from_node(node)
            # annotate node type
            if node.target in COMM_FUNCS:
                node.ed_info.node_type = EDNodeType.COMMUNITAION
            elif node.target == operator.getitem:
                node.ed_info.node_type = EDNodeType.AUXILIARY
            else:
                node.ed_info.node_type = EDNodeType.COMPUTATION
        elif node.op == 'output':
            node.ed_info.node_type = EDNodeType.AUXILIARY

    return fx_module
