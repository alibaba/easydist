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

from typing import Dict, List
import operator
from matplotlib.pyplot import step
import torch
import torch.fx as fx
from easydist.torch.experimental.pp.split_utils import ANNOTATION_OPS, get_backward_flag, get_step_flag
from easydist.torch.passes.comm_optimize import _link_nodes
from easydist.torch.passes.sharding import CUSTOM_FUNCS

def fix_sharding_node_order(fx_module: fx.GraphModule):
    nodes = list(fx_module.graph.nodes)
    for i in range(len(nodes)):
        node = nodes[i]
        if node.target in CUSTOM_FUNCS:
            ancessor_id = max(map(lambda n: nodes.index(n), node.all_input_nodes))
            if nodes[ancessor_id].op == 'placeholder':
                continue
            nodes.pop(i)
            nodes.insert(ancessor_id + 1, node)

    assert set(nodes) == set(fx_module.graph.nodes)
    _link_nodes(fx_module, nodes)
    fx_module.recompile()
    return fx_module

def bfs2bottom(root: fx.Node, vis: Dict[fx.Node, bool]):
    q = [root]
    res = []
    while q:
        cur = q.pop(0)
        res.append(cur)
        for user in cur.users:
            if vis[user]:
                continue
            vis[user] = True
            q.append(user)
    return set(res)

def bfs2top(root: fx.Node, vis: Dict[fx.Node, bool]):
    q = [root]
    res = []
    while q:
        cur = q.pop(0)
        res.append(cur)
        for input_node in cur.all_input_nodes:
            if vis[input_node]:
                continue
            vis[input_node] = True
            q.append(input_node)
    return set(res)

def fix_order(fx_module: fx.GraphModule):
    '''
                                            ┏ grad ┓
    Assume that the graph looks like fw -> bw -> step -> output
                                      ┣ act ┛               ┃
                                      ┗━━━━ fw output ━━━━━━┛
    '''
    # phs and output
    phs = []
    output = None
    # find bw split node and step split node
    fw_bw_split_nodes = []
    backward_split_node = step_split_node = None
    for node in fx_module.graph.nodes:
        if node.op == 'placeholder':
            phs.append(node)
        elif node.op == 'call_function':
            if node.target == torch.ops.easydist.fw_bw_split.default:
                fw_bw_split_nodes.append(node)
            elif node.target == torch.ops.easydist.step_split.default:
                assert step_split_node is None
                step_split_node = node
        elif node.op == 'output':
            output = node

    if get_backward_flag():
        assert len(fw_bw_split_nodes) % 2 == 1
        n = len(fw_bw_split_nodes)
        backward_split_node = fw_bw_split_nodes[n // 2]
    
    # partition graph nodes
    vis = {node: False for node in fx_module.graph.nodes}
    # ind = {node: len(node.all_input_nodes) for node in fx_module.graph.nodes}
    # outd = {node: len(node.users) for node in fx_module.graph.nodes}

    for ph in phs:
        vis[ph] = True
    vis[output] = True
    # step nodes, any toplogical order will do
    step_partition = set()
    if step_split_node:
        vis[step_split_node] = True
        step_partition = bfs2bottom(step_split_node, vis)

    # backward nodes, any toplogical order will do
    forward_partition = set()
    backward_partition = set()
    if backward_split_node:
        vis[backward_split_node] = True
        backward_partition = bfs2bottom(backward_split_node, vis)
        forward_partition = bfs2top(backward_split_node, vis)

    forward_nodes = []
    backward_nodes = []
    step_nodes = []
    # get topological order
    for node in fx_module.graph.nodes:
        if node.op == 'placeholder' or node.op == 'output':
            continue
        if node in forward_partition:
            forward_nodes.append(node)
        elif node in backward_partition:
            backward_nodes.append(node)
        elif node in step_partition:
            step_nodes.append(node)

    nodes = phs + forward_nodes + backward_nodes + step_nodes + [output]
    
    # jump nodes
    #   for grad and output (user is step_split_node) move to earlist point
    #   for activations, move to the latest point
    for node in fx_module.graph.nodes:
        if not vis[node] and (step_split_node in node.users or output in node.users):
                vis[node] = True
                accessor_id = max(map(lambda n: nodes.index(n), node.all_input_nodes))
                nodes.insert(accessor_id + 1, node)

    for node in reversed(list(fx_module.graph.nodes)):
        if not vis[node]:
            vis[node] = True
            succeesor_id = min(map(lambda n: nodes.index(n), node.users))
            nodes.insert(succeesor_id, node)

    assert set(nodes) == set(fx_module.graph.nodes)
    _link_nodes(fx_module, nodes)
    fx_module.recompile()
    return fx_module
            
    

# def fix_sharding_node_order(fx_module: fx.GraphModule):
#     outd = {node: len(node.users) for node in fx_module.graph.nodes}
#     nodes = list(fx_module.graph.nodes)
#     cur: Dict[fx.Node, None] = {ph: None for ph in fx_module.graph.nodes if ph.op == 'placeholder'}
#     for i, node in enumerate(nodes):
#         if node.op == 'call_function':
#             for input_node in node.all_input_nodes:
#                 outd[input_node] -= 1
#                 if outd[input_node] == 0:
#                     cur.pop(input_node)
#             if node.target is torch.ops.easydist.fw_bw_split.default:
#                 tensor_list = list(node.args[0])
#                 new_cur = {}
#                 for to_split in cur:
#                     tensor_list.append(to_split)
#                     new_node = fx_module.graph.create_node(
#                         op='call_function',
#                         target=operator.getitem,
#                         args=(node, len(tensor_list) - 1),
#                         name=to_split.name
#                     )
#                     to_split.replace_all_uses_with(new_node, delete_user_cb=lambda user: False)
#                     outd[new_node] = outd.pop(to_split)
#                     new_cur[new_node] = None
#                 cur = new_cur
#             cur[node] = None

#     # fx_module.recompile()
#     return fx_module
