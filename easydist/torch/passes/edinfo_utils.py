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

import torch
from torch.fx.node import Node, _get_qualified_name
import torch.utils._pytree as pytree

from easydist.torch.utils import EDInfo, EDNodeType, create_meta_from_node
from easydist.torch.passes.sharding import CREATE_ATEN_OP, COMM_FUNCS


def create_edinfo(fx_module: torch.fx.GraphModule, sharding_info, shape_info) -> torch.fx.GraphModule:

    for node in fx_module.graph.nodes:

        if node.op == 'call_function' and 'val' not in node.meta:
            node.meta = create_meta_from_node(node)


        if not hasattr(node, "ed_info"):
            node.ed_info = EDInfo(ori_meta=node.meta)

        if node.op == "call_function":
            op_name = _get_qualified_name(node.target)

            node_sharding_info = None
            if op_name in sharding_info:

                def _gen_meta(arg: Node):
                    return torch.empty(shape_info[arg.name]["shape"],
                                       dtype=shape_info[arg.name]["dtype"],
                                       device="meta")

                args_meta = pytree.tree_map_only(Node, _gen_meta, node.args)
                args_meta = str(tuple(args_meta)) + ' | ' + str(node.kwargs)
                if args_meta in sharding_info[op_name]:
                    node_sharding_info = sharding_info[op_name][args_meta]

            node.ed_info.spmd_annotation = node_sharding_info

        elif node.op in ["placeholder", "get_attr"]:
            if hasattr(node, "meta") and 'val' in node.meta:
                node_sharding_info = None
                if node.op in sharding_info:
                    arg_meta_tensor = torch.empty(shape_info[node.name]["shape"],
                                                  dtype=shape_info[node.name]["dtype"],
                                                  device="meta")
                    args_meta = str(arg_meta_tensor)
                    if args_meta in sharding_info[node.op]:
                        node_sharding_info = sharding_info[node.op][args_meta]

                node.ed_info.spmd_annotation = node_sharding_info

    return fx_module


def annotation_edinfo(traced_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in traced_graph.graph.nodes:
        if not hasattr(node, "ed_info"):
            node.ed_info = EDInfo(ori_meta=node.meta)

        if node.op == 'placeholder':
            node.ed_info.node_type = EDNodeType.AUXILIARY
        elif node.op == 'call_function':
            # create meta for custom function
            if node.target not in CREATE_ATEN_OP:
                node.meta = create_meta_from_node(node)
            # annotate node type
            if node.target in COMM_FUNCS:
                node.ed_info.node_type = EDNodeType.COMMUNICATION
            # (TODO) hard code here to avoid to runtime profile torch.ops.aten._fused_adam.default
            elif node.target in [operator.getitem, torch.ops.aten._fused_adam.default]:
                node.ed_info.node_type = EDNodeType.AUXILIARY
            else:
                node.ed_info.node_type = EDNodeType.COMPUTATION
        elif node.op == 'output':
            node.ed_info.node_type = EDNodeType.AUXILIARY

    return traced_graph