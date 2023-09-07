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
from torch.fx.node import Node, _get_qualified_name
import torch.distributed.distributed_c10d as c10d

import torch.distributed._tensor as spmd
from torch.distributed._tensor import distribute_tensor
import torch.utils._pytree as pytree

from metadist.metashard.combination import ReduceOp
from metadist.metashard import metair
from metadist.metashard.metair import MetaGraph, MetaNode, MetaVar
from metadist.utils import rsetattr, rgetattr
import metadist.config as mdconfig

from .passes.sharding import get_device_mesh

ABSTRACT_DTYPE = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.complex64: "complex64",
}


def to_torch_spmd(meta_spmd):
    if meta_spmd.state == metair.SPMD.SHARD:
        return spmd.Shard(dim=meta_spmd.args["dim"])
    elif meta_spmd.state == metair.SPMD.PARTIAL:
        mapping_ops = {
            ReduceOp.SUM: c10d.ReduceOp.RedOpType.SUM,
            ReduceOp.MAX: c10d.ReduceOp.RedOpType.MAX,
            ReduceOp.MIN: c10d.ReduceOp.RedOpType.MIN,
        }
        return spmd.placement_types._Partial(reduce_op=mapping_ops[meta_spmd.args["ops"]])
    elif meta_spmd.state == metair.SPMD.REPLICATE:
        return spmd.Replicate()


def materialize(x, device):
    if isinstance(x, torch.Tensor) and x.is_meta:
        if x.dtype == torch.bool:
            return torch.rand(x.size(), dtype=torch.float, device=device) > 0.5
        elif torch.is_floating_point(x):
            return torch.rand(x.size(), dtype=x.dtype, device=device)
        else:
            return torch.randint(high=8, size=x.size(), dtype=x.dtype, device=device)
    return x


def shard_module(model, input_, input_strategy, device="cuda"):
    mesh = get_device_mesh()

    input_strategy = [[to_torch_spmd(i) for i in var_strategy] for var_strategy in input_strategy]

    idx = 0
    for name in dict(model.named_parameters()):

        tensor_data = rgetattr(model, name).data
        tensor_data = materialize(tensor_data, device=device)

        rsetattr(
            model, name,
            torch.nn.parameter.Parameter(distribute_tensor(tensor_data, mesh,
                                                           input_strategy[idx])))

        with torch.no_grad():
            rsetattr(model, name + ".grad", torch.empty_like(rgetattr(model, name).data))
        idx += 1

    for name in dict(model.named_buffers()):

        tensor_data = rgetattr(model, name).data
        tensor_data = materialize(tensor_data, device=device)

        rsetattr(model, name, distribute_tensor(tensor_data, mesh, input_strategy[idx]))
        idx += 1

    shard_input = []
    for tensor in input_:
        tensor = materialize(tensor, device=device)
        shard_input.append(distribute_tensor(tensor, mesh, input_strategy[idx]))
        idx += 1

    return shard_input


def torch2meta_graph(fx_module: torch.fx.GraphModule, state_tensor_num, sharding_info,
                     meta_info) -> MetaGraph:
    meta_graph = MetaGraph(fx_module)
    meta_node_map = {}
    meta_var_map = {}
    MetaNode.clear_id_counter()
    MetaVar.clear_id_counter()
    output_names = []
    # 1. create MetaVar for each output of fx graph node except list or tuple
    #    and create MetaNode for each fx graph node except getitem

    for node in fx_module.graph.nodes:
        if node.op == "call_function":
            # 1.1. create MetaVar
            outvars = []
            compact_out_idx_tbl = []
            compact_out_idx = 0
            # NOTE: list and tuple are followed by getitem
            if isinstance(meta_info[node.name], list) or isinstance(meta_info[node.name], tuple):
                for idx, var_meta in enumerate(meta_info[node.name]):
                    if var_meta is not None and var_meta != {}:
                        compact_out_idx_tbl.append(compact_out_idx)
                        compact_out_idx = compact_out_idx + 1
                    else:
                        compact_out_idx_tbl.append(-1)

                # don't create MetaVar. Will fill it with getitem's MetaVar
                if compact_out_idx > 0:
                    outvars = [None] * compact_out_idx
            else:
                meta_var = MetaVar(name=node.name,
                                   shape=meta_info[node.name]["shape"],
                                   dtype=ABSTRACT_DTYPE[meta_info[node.name]["dtype"]])

                meta_var_map[node.name] = meta_var
                outvars.append(meta_var)

            op_name = _get_qualified_name(node.target)
            if op_name == "_operator.getitem":
                # not create MetaNode
                continue

            # 1.2. create MetaNode
            node_sharding_info = None
            if op_name in sharding_info:
                def _gen_meta(arg):
                    if isinstance(arg, Node):
                        return torch.empty(meta_info[arg.name]["shape"],
                                           dtype=meta_info[arg.name]["dtype"],
                                           device="meta")
                    else:
                        # primitive data type: int, float, etc
                        return arg

                args_meta = pytree.tree_map(_gen_meta, node.args)
                args_meta = str(tuple(args_meta)) + ' | ' + str(node.kwargs)
                if args_meta in sharding_info[op_name]:
                    node_sharding_info = sharding_info[op_name][args_meta]

            node_args_flatten = pytree.tree_flatten(node.args)[0]
            compact_in_idx_tbl = []
            compact_in_idx = 0
            for arg in node_args_flatten:
                if isinstance(arg, Node):
                    compact_in_idx_tbl.append(compact_in_idx)
                    compact_in_idx = compact_in_idx + 1
                else:
                    compact_in_idx_tbl.append(-1)
            invars = [None] * compact_in_idx

            meta_node = MetaNode(name=node.name,
                                 op_name=op_name,
                                 invars=invars,
                                 outvars=outvars,
                                 sharding_info=node_sharding_info)
            meta_node.compact_out_idx_tbl = compact_out_idx_tbl
            meta_node.compact_in_idx_tbl = compact_in_idx_tbl
            meta_node_map[node.name] = meta_node
            meta_graph.add_node(meta_node)
        elif node.op in ["placeholder", "get_attr"]:
            if meta_info[node.name] != {}:
                # 1.1. create MetaVar
                meta_var = MetaVar(name=node.name,
                                   shape=meta_info[node.name]["shape"],
                                   dtype=ABSTRACT_DTYPE[meta_info[node.name]["dtype"]])

                meta_var_map[node.name] = meta_var

                # 1.2. create MetaNode
                node_sharding_info = None
                if node.op in sharding_info:
                    arg_meta_tensor = torch.empty(meta_info[node.name]["shape"],
                                                  dtype=meta_info[node.name]["dtype"],
                                                  device="meta")
                    args_meta = str(arg_meta_tensor)
                    if args_meta in sharding_info[node.op]:
                        node_sharding_info = sharding_info[node.op][args_meta]

                meta_node = MetaNode(name=node.name,
                                     op_name=node.op,
                                     invars=[],
                                     outvars=[meta_var],
                                     sharding_info=node_sharding_info,
                                     is_placeholder=True)
                meta_node.compact_out_idx_tbl = [0]
                meta_node.compact_in_idx_tbl = []
                meta_node_map[node.name] = meta_node
                meta_graph.add_node(meta_node)

                meta_graph.add_input(meta_node)
        elif node.op == "output":
            output_names = [arg.name for arg in node.args[0] if arg is not None]

    # 2. update connection between MetaNode and MetaVar
    for node in fx_module.graph.nodes:
        if node.op == "call_function":
            op_name = _get_qualified_name(node.target)
            if op_name == "_operator.getitem":
                # connect MetaVar of this node to MetaNode of its up node
                arg, idx = node.args
                assert isinstance(arg, Node)
                assert isinstance(idx, int)
                arg_meta_node = meta_node_map[arg.name]  # find meta node of getitem's up node
                meta_var = meta_var_map[node.name]  # find getitem's output var
                arg_meta_node.set_out_var(meta_var, idx)
                continue

            node_args_flatten = pytree.tree_flatten(node.args)[0]
            for idx, arg in enumerate(node_args_flatten):
                if isinstance(arg, Node):
                    meta_node = meta_node_map[node.name]
                    in_var = meta_var_map[arg.name]
                    meta_node.set_in_var(in_var, idx)

    for ouput_n in output_names:
        meta_graph.add_output(meta_var_map[ouput_n])

    # add state io map for input/output sharding mismatch cost
    state_io_map = {}
    for i in range(state_tensor_num):
        state_io_map[meta_graph.input_list[i]] = meta_graph.output_list[i]
    meta_graph.state_io_map = state_io_map

    meta_graph.coarsen(coarsen_level=mdconfig.coarsen_level)

    return meta_graph


def get_torch_sharding_strategy(fx_module: torch.fx.GraphModule, opt_strategy):
    sharding_strategy = {}

    for node in fx_module.graph.nodes:
        if node.op == "call_function":
            op_name = _get_qualified_name(node.target)
            if op_name != "_operator.getitem":
                unique_key = node.name
                if unique_key in opt_strategy:
                    sharding_strategy[node.name] = [[
                        to_torch_spmd(ii) for ii in i
                    ] for i in opt_strategy[unique_key]['strategy'].in_strtg_group]
                else:
                    unique_key = f"{op_name}_[{node.name}]"
                    if unique_key in opt_strategy:
                        sharding_strategy[node.name] = [[
                            to_torch_spmd(ii) for ii in i
                        ] for i in opt_strategy[unique_key]['strategy'].in_strtg_group]

    return sharding_strategy
