# Copyright (c) 2024, Alibaba Group;
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
import inspect
import torch
import easydist.torch.scope_auto.scope_marker
from typing import Dict, Set, List, Union

class GraphInfo:

    def __init__(self, id: int):
        self.id: int = id
        self.submod_name = f"submod_{id}"
        self.orig_nodes: List[Union[torch.fx.Node, 'GraphInfo']] = []
        self.inputs: Set[str] = set()
        self.outputs: Set[str] = set()
        self.input_list: List[str] = []
        self.output_list: List[str] = []
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.name_to_new_node: Dict[str, torch.fx.Node] = {}
        self.targets: Dict[str, Any] = {}

    def record_orig_node(self, node_or_graphinfo: Union[torch.fx.Node, 'GraphInfo']):
        self.orig_nodes.append(node_or_graphinfo)

    def add_input(self, input: str):
        if input not in self.inputs:
            self.inputs.add(input)
            self.input_list.append(input)

    def add_output(self, output: str):
        if output not in self.outputs:
            self.outputs.add(output)
            self.output_list.append(output)

# extract nodes between marker's start and end from main graph
def extract_graphs(
    m: torch.fx.GraphModule
) -> Dict[torch.fx.Node, int]:

    def is_scope_start(n: torch.fx.Node) -> bool:
        if (n.op, n.target) == ("call_function", torch.ops.easydist.fw_scope_start.default) \
            or (n.op, n.target) == ("call_function", torch.ops.easydist.bw_scope_start.default):
            return True
        else:
            return False

    def is_scope_end(n: torch.fx.Node) -> bool:
        if (n.op, n.target) == ("call_function", torch.ops.easydist.fw_scope_end.default) \
            or (n.op, n.target) == ("call_function", torch.ops.easydist.bw_scope_end.default):
            return True
        else:
            return False

    with_scope = False
    for node in m.graph.nodes:
        if is_scope_start(node):
            with_scope = True

    if not with_scope:
        return None, None

    top_graph_info = GraphInfo(0)
    cur_graph_info = top_graph_info
    graph_ids = {}
    graph_infos: List[GraphInfo] = [top_graph_info]
    cur_subgraph_id = 0
    for node in m.graph.nodes:
        if node.op == "placeholder" or node.op == "get_attr":
            graph_ids[node.name] = 0
            top_graph_info.record_orig_node(node)
            continue

        if cur_subgraph_id > 0:
            # already in a subgraph
            if is_scope_start(node):
                raise RuntimeError(f"scope marker {node} is nested")
            graph_ids[node.name] = cur_subgraph_id
            cur_graph_info.record_orig_node(node)
            if is_scope_end(node):
                cur_subgraph_id = 0
                cur_graph_info = None
        else:
            # in top level
            if is_scope_start(node):
                cur_subgraph_id = len(graph_infos)
                cur_graph_info = GraphInfo(cur_subgraph_id)
                graph_ids[node.name] = cur_subgraph_id
                cur_graph_info.record_orig_node(node)
                graph_infos.append(cur_graph_info)
                top_graph_info.record_orig_node(cur_graph_info)
            else:
                graph_ids[node.name] = 0
                top_graph_info.record_orig_node(node)


    #for node in m.graph.nodes:
    #    if node.name in graph_ids:
    #        print(f"node {node}: subgraph id: {graph_ids[node.name]}")
    #    else:
    #        print(f"node {node}: subgraph id: -1")

    return graph_ids, graph_infos

def build_graph(
    top_mod: torch.fx.GraphModule,
    graph_ids: Dict[str, int],
    name_to_orig_node: Dict[str, torch.fx.Node],
    graph_info: GraphInfo
):
    graph = graph_info.graph
    name_to_new_node = graph_info.name_to_new_node
    targets = graph_info.targets

    # 1. convert inputs to placeholders
    if graph_info.input_list: # NOTE: it is empty if the graph is top level
        # build placeholder for sub graph
        for input in graph_info.input_list:
            orig_input_node = name_to_orig_node[input]
            placeholder = graph_info.graph.placeholder(input, type_expr=orig_input_node.type)
            placeholder.meta = orig_input_node.meta.copy()
            name_to_new_node[orig_input_node.name] = placeholder

    # 2. convert original nodes to new nodes
    for node in graph_info.orig_nodes:
        if isinstance(node, torch.fx.Node):
            if node.op == "placeholder": # top level placeholder
                default_value = (node.args[0] if len(node.args) > 0 else inspect.Signature.empty)
                ph = graph.placeholder(node.target,
                                       type_expr=node.type,
                                       default_value=default_value)

                ph.meta = node.meta.copy()
                name_to_new_node[node.name] = ph
            elif node.op == "get_attr": # top level get_attr
                attr = graph.get_attr(node.target)
                attr.meta = node.meta.copy()
                attr_val = top_mod

                for atom in node.target.split("."):
                    if not hasattr(attr_val, atom):
                        raise AttributeError(f"Node target {node.target} not found!")
                    attr_val = getattr(attr_val, atom)
                targets[node.target] = attr_val
                name_to_new_node[node.name] = attr
            elif node.op == "output":  # top level output
                graph.output(
                    torch.fx.graph.map_arg(node.args[0], lambda n: name_to_new_node[n.name]))
            else:
                new_args = torch.fx.graph.map_arg(node.args, lambda n: name_to_new_node[n.name])
                new_kwargs = torch.fx.graph.map_arg(node.kwargs, lambda n: name_to_new_node[n.name])

                if node.op not in ["call_module", "get_attr"]:
                    target = node.target
                else:
                    target_atoms = node.target.split(".")
                    target_attr = top_mod
                    for atom in target_atoms:
                        if not hasattr(target_attr, atom):
                            raise AttributeError(f"Operator target {node.target} not found!")
                        target_attr = getattr(target_attr, atom)
                    target = "_".join(target_atoms)
                    targets[target] = target_attr

                assert isinstance(new_args, tuple)
                assert isinstance(new_kwargs, dict)
                new_node = graph.create_node(
                    name=node.name,
                    op=node.op,
                    target=target,
                    args=new_args,
                    kwargs=new_kwargs,
                    type_expr=node.type,
                )
                new_node.meta = node.meta.copy()
                name_to_new_node[node.name] = new_node
        else:
            assert isinstance(node, GraphInfo)
            # build sub graph
            sub_graph_info = node
            sub_mod = build_graph(top_mod, graph_ids, name_to_orig_node, sub_graph_info)
            targets[sub_graph_info.submod_name] = sub_mod

            #print(f"sub module({sub_graph_info.submod_name}) before eliminate dead code:\n{sub_mod.graph}")
            sub_mod.graph.eliminate_dead_code()
            #print(f"sub module({sub_graph_info.submod_name}) after eliminate dead code:\n{sub_mod.graph}")

            # Emit call in current graph to the sub graph
            output_val = graph.create_node(op='call_module',
                                           target=sub_graph_info.submod_name,
                                           args=tuple(name_to_new_node[name]
                                                      for name in sub_graph_info.input_list),
                                           name=sub_graph_info.submod_name)

            sub_num_outputs = len(sub_graph_info.output_list)
            if sub_num_outputs > 1:
                # Unpack multiple return values from sub graph
                output_val_proxy = torch.fx.proxy.Proxy(output_val)
                for i, output_name in enumerate(sub_graph_info.output_list):
                    node = output_val_proxy[i].node
                    node.name = output_name
                    name_to_new_node[output_name] = node
            elif sub_num_outputs == 1:
                # Introduce one redudant node
                output_name = sub_graph_info.output_list[0]
                name_to_new_node[output_name] = graph.create_node(op='call_function',
                                                                  target=lambda x: x,
                                                                  args=(output_val, ),
                                                                  name=output_name)

    # 3. convert outputs to an output node
    if graph_info.output_list: # NOTE: it is empty if the graph is top level
        output_vals = tuple(name_to_new_node[name] for name in graph_info.output_list)
        num_output_vals = len(output_vals)
        if num_output_vals == 1:
            graph.output(output_vals[0])
        elif num_output_vals > 1:
            graph.output(output_vals)

    return torch.fx.graph_module.GraphModule(targets, graph)

def remove_marker_nodes(
    graph_info: GraphInfo
):
    for node in graph_info.orig_nodes:
        if not isinstance(node, torch.fx.Node):
            continue

        #if node.args:
        #    print(f"node {node}, node type: {type(node)}, args: {node.args}, arg type: {type(node.args)}, arg len: {len(node.args)}, arg elem type: {type(node.args[0])}")
        #else:
        #    print(f"node {node}, node type: {type(node)}, args: {node.args}, arg type: {type(node.args)}, arg len: {len(node.args)}")

        if (node.op, node.target) == ("call_function", torch.ops.easydist.fw_scope_start.default) \
           or (node.op, node.target) == ("call_function", torch.ops.easydist.fw_scope_end.default) \
           or (node.op, node.target) == ("call_function", torch.ops.easydist.bw_scope_start.default) \
           or (node.op, node.target) == ("call_function", torch.ops.easydist.bw_scope_end.default):
            # passthrough marker node and the following getitem node
            input_nodes = node.args[0]
            for user in node.users:
                assert user.target == operator.__getitem__
                _, idx = user.args

                src_node = input_nodes[idx]
                user.replace_all_uses_with(src_node)

def record_cross_graph(arg_node: torch.fx.Node,
                       node: torch.fx.Node,
                       graph_ids: Dict[str, int],
                       graph_infos: List[GraphInfo]):
    arg_graph_id = graph_ids[arg_node.name]
    node_graph_id = graph_ids[node.name]
    if arg_graph_id != node_graph_id:
        if node_graph_id > 0:
            graph_infos[node_graph_id].add_input(arg_node.name)

        if arg_graph_id > 0:
            graph_infos[arg_graph_id].add_output(arg_node.name)

def build_scope_modules(
    m: torch.fx.GraphModule
) -> torch.fx.GraphModule:
    graph_ids, graph_infos = extract_graphs(m)

    if not graph_infos:
        # only top level graph
        return m

    #print(f"before remove marker:\n{m.graph}")
    for graph_info in graph_infos:
        remove_marker_nodes(graph_info)
    #print(f"after remove marker:\n{m.graph}")

    assert len(graph_infos) > 1

    # collect input/output nodes
    name_to_orig_node: Dict[str, torch.fx.Node] = {}
    for node in m.graph.nodes:
        name_to_orig_node[node.name] = node
        torch.fx.graph.map_arg(node.args, lambda arg_node: record_cross_graph(arg_node, node, graph_ids, graph_infos))
        torch.fx.graph.map_arg(node.kwargs, lambda arg_node: record_cross_graph(arg_node, node, graph_ids, graph_infos))

    # add placeholders for empty sub graphs
    top_graph_info = graph_infos[0]

    new_top_mod = build_graph(m, graph_ids, name_to_orig_node, top_graph_info)

    #print(f"top module before eliminate dead code:\n{new_top_mod.graph}")
    new_top_mod.graph.eliminate_dead_code()
    #print(f"top module after eliminate dead code:\n{new_top_mod.graph}")

    new_top_mod.graph.set_codegen(m.graph._codegen)

    return new_top_mod


