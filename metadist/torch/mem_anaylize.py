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

from functools import reduce
from operator import mul
from typing import Dict, List

import torch
from torch.fx.node import Node, _get_qualified_name, map_arg

from .bridge import to_torch_spmd
from .device_mesh import device_mesh_world_size

DTYPE_BYTE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
    torch.bool: 0.125
}


def tensor_mem_size(shape_info, mem_ratio):
    mem_size = 0
    if isinstance(shape_info, tuple):
        for shape_info_, mem_ratio_ in zip(shape_info, mem_ratio):
            if 'shape' in shape_info_:
                if len(shape_info_['shape']) == 0:
                    mem_size += DTYPE_BYTE[shape_info_['dtype']]
                else:
                    mem_size += reduce(
                        mul, shape_info_['shape']) * DTYPE_BYTE[shape_info_['dtype']] * mem_ratio_
    else:
        if 'shape' in shape_info:
            if len(shape_info['shape']) == 0:
                mem_size += DTYPE_BYTE[shape_info['dtype']]
            else:
                mem_size = reduce(
                    mul, shape_info['shape']) * DTYPE_BYTE[shape_info['dtype']] * mem_ratio
    return mem_size


def _get_input_strategy(fx_graph, opt_strategy):
    input_strategy = {}

    input_name = []
    for node in fx_graph.graph.nodes:
        if node.op == 'placeholder':
            input_name.append(node.name)

    for node in reversed(fx_graph.graph.nodes):
        if node.op == "call_function":
            op_name = _get_qualified_name(node.target)
            if op_name != "_operator.getitem":
                unique_key = node.name
                if unique_key in opt_strategy:
                    invars = opt_strategy[unique_key]['node'].invars
                    for idx, var in enumerate(invars):
                        if var.name in input_name:
                            input_strategy[var.name] = [
                                to_torch_spmd(i)
                                for i in opt_strategy[unique_key]['strategy'].in_strtg_group[idx]
                            ]

    return input_strategy


def mem_anaylize(fx_graph: torch.fx.GraphModule, tensor_shape_info, opt_strategy):

    same_mem_funcs = [
        "torch.ops.aten.t", "torch.ops.aten.view", "torch.ops.aten._unsafe_view",
        "torch.ops.aten.slice", "torch.ops.aten.transpose", "torch.ops.aten.squeeze",
        "torch.ops.aten.unsqueeze", "torch.ops.aten.permute", "torch.ops.aten.slice",
        "_operator.getitem", "torch.ops.aten.expand"
    ]

    # Run through reverse nodes and record the first instance of a use
    # of a given node. This represents the *last* use of the node in the
    # execution order of the program, which we will use to free unused
    # values
    node_to_last_use: Dict[Node, Node] = {}
    user_to_last_uses: Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        if n not in node_to_last_use:
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

    for node in reversed(fx_graph.graph.nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    persistent_mem, activation_mem = {}, {}
    mem_ratio = {}
    same_memory_tensor_info = {}

    total_memory, peak_memory = 0, 0

    f = open("mem_info.txt", "w")

    world_size = device_mesh_world_size()

    input_strategy = _get_input_strategy(fx_graph, opt_strategy)

    for node in fx_graph.graph.nodes:
        # input and parametes will not release during inference
        if node.op == 'placeholder':

            mem_ratio[node.name] = 1
            if node.name in input_strategy:
                for place in input_strategy[node.name]:
                    if place.is_shard():
                        mem_ratio[node.name] = 1 / world_size

            persistent_mem[node.name] = tensor_mem_size(tensor_shape_info[node.name],
                                                        mem_ratio[node.name])
            total_memory += persistent_mem[node.name]
            if peak_memory < total_memory:
                peak_memory = total_memory

        if node.op == 'call_function':
            if _get_qualified_name(node.target) in same_mem_funcs:
                if node.args[0].name not in persistent_mem:
                    if node.args[0] not in same_memory_tensor_info:
                        same_memory_tensor_info[node.args[0].name] = [node.args[0].name]
                    same_memory_tensor_info[node.args[0].name].append(node.name)
            else:
                mem_ratio[node.name] = [1]
                if node.name in opt_strategy:
                    out_strategy = [[to_torch_spmd(i) for i in ii]
                                    for ii in opt_strategy[node.name]['strategy'].out_strtg_group]
                    mem_ratio[node.name] = [1] * len(out_strategy)
                    for idx, out_stgy in enumerate(out_strategy):
                        for place in out_stgy:
                            if place.is_shard():
                                mem_ratio[node.name][idx] = 1 / world_size

                if not isinstance(tensor_shape_info[node.name], tuple):
                    mem_ratio[node.name] = mem_ratio[node.name][0]

                activation_mem[node.name] = tensor_mem_size(tensor_shape_info[node.name],
                                                            mem_ratio[node.name])
                total_memory += activation_mem[node.name]
                if peak_memory < total_memory:
                    peak_memory = total_memory

        # release used tensor
        for to_delete in user_to_last_uses.get(node, []):

            for value in same_memory_tensor_info.values():
                while to_delete.name in value:
                    value.remove(to_delete.name)

            new_same_memory_tensor_info = {}
            for k, v in same_memory_tensor_info.items():
                if len(v) > 0:
                    new_same_memory_tensor_info[k] = v
                else:
                    if k in activation_mem:
                        total_memory -= tensor_mem_size(tensor_shape_info[k], mem_ratio[k])
                        del activation_mem[k]
            same_memory_tensor_info = new_same_memory_tensor_info

            if to_delete.name in activation_mem:
                if to_delete.name not in same_memory_tensor_info:
                    total_memory -= tensor_mem_size(tensor_shape_info[to_delete.name],
                                                    mem_ratio[to_delete.name])
                    del activation_mem[to_delete.name]

        gb_ratio = 1024 * 1024 * 1024
        f.write(
            f"Node | Total | Peak Memory: {node} | {total_memory / gb_ratio:.4f} GB | {peak_memory / gb_ratio:.4f} GB\n"
        )

    f.close()
