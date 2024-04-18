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

from typing import Dict
from collections import Counter
import torch
import torch.fx as fx
from easydist.torch.experimental.pp.split_utils import get_backward_flag
from easydist.torch.passes.comm_optimize import _link_nodes
from easydist.torch.passes.sharding import CUSTOM_FUNCS

def bfs2bottom(root: fx.Node, vis: Dict[fx.Node, bool]):
    q = [root]
    res = []
    while q:
        cur = q.pop(0)
        for user in cur.users:
            if vis[user]:
                continue
            vis[user] = True
            res.append(user)
            q.append(user)
    return set(res)

def bfs2top(root: fx.Node, vis: Dict[fx.Node, bool]):
    q = [root]
    res = []
    while q:
        cur = q.pop(0)
        for input_node in cur.all_input_nodes:
            if vis[input_node]:
                continue
            vis[input_node] = True
            res.append(input_node)
            q.append(input_node)
    return set(res)

def fix_node_order(fx_module: fx.GraphModule):
    '''
                                    ┏ grad ┓
    Assume that the graph is fw -> bw -> step -> output
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

    for ph in phs:
        vis[ph] = True
    vis[output] = True
    # step nodes, any toplogical order will do
    step_partition = set()
    if step_split_node:
        vis[step_split_node] = True
        step_partition = bfs2bottom(step_split_node, vis)
        step_partition.add(step_split_node)

    # backward nodes, any toplogical order will do
    forward_partition = bfs2top(backward_split_node, vis)
    backward_partition = set()
    if backward_split_node:
        vis[backward_split_node] = True
        backward_partition = bfs2bottom(backward_split_node, vis)
        backward_partition.add(backward_split_node)

    # other comm nodes
    #   1. outputs move to earlist point
    #   2. activations nodes, move to the latest point (i.e. cooresponding backward submod)
    returns = bfs2top(output, vis)
    activations = set(n for n in vis if not vis[n])

    nodes = list(fx_module.graph.nodes)

    for i, node in enumerate(nodes):
        if node in (backward_partition | returns) and node.target in CUSTOM_FUNCS :
            ancessor_id = max(map(lambda n: nodes.index(n), node.all_input_nodes))
            nodes.pop(i)
            nodes.insert(ancessor_id + 1, node)

    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if node in activations:
            successor_id = min(map(lambda n: nodes.index(n), node.users))
            nodes.pop(i)
            nodes.insert(successor_id - 1, node)

    assert Counter(nodes) == Counter(fx_module.graph.nodes)

    _link_nodes(fx_module, nodes)
    fx_module.recompile()
    return fx_module


