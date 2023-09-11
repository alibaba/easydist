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

import logging
import operator

import torch
from sortedcontainers import SortedList
from torch.distributed._tensor.api import DTensor
from torch.fx.node import _get_qualified_name

from easydist.torch.device_mesh import get_device_mesh

logger = logging.getLogger(__name__)


def comm_optimize(fx_module: torch.fx.GraphModule, opt_strategy=None, grouping=False):
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    node_to_rank: dict[torch.fx.Node, int] = {}
    # processing
    rank = 0
    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            op_name = _get_qualified_name(node.target)
            if op_name == 'easydist.torch.passes.sharding.redist_tensor_func':
                continue
        node_to_rank[node] = rank
        rank += 1

    '''
        ppredecesor -> predecesor1 -> node1
            |    ...                    ^
            v                           |
    predecesorN -> nodeN           predecesor1'
    '''

    # comm_op expressed as (from_node, comm_node, to_node)

    comm_queue = SortedList(key=lambda op_tup: node_to_rank[op_tup[0]])

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            op_name = _get_qualified_name(node.target)

            if op_name != 'easydist.torch.passes.sharding.redist_tensor_func':
                input_nodes = node.all_input_nodes
                for predecesor in input_nodes:
                    if predecesor.op == 'call_function':
                        pre_op_name = _get_qualified_name(predecesor.target)
                        if pre_op_name == 'easydist.torch.passes.sharding.redist_tensor_func':
                            ppredecesor = predecesor.all_input_nodes
                            assert (len(ppredecesor) == 1)
                            from_node = ppredecesor[0]
                            comm_node = predecesor
                            to_node = node

                            # code needed to eliminate redundant comm node
                            if opt_strategy is not None and from_node.op == 'call_function':
                                target_node = from_node
                                idx = 0
                                if from_node.name.__contains__('getitem'):
                                    pppredecesor = from_node.all_input_nodes
                                    assert(len(pppredecesor) == 1)
                                    target_node = pppredecesor[0]
                                    idx = from_node.args[1]
                                to_node_strategy = opt_strategy[to_node.name]['strategy']
                                from_node_strategy = opt_strategy[target_node.name]['strategy']
                                if from_node_strategy is not None and \
                                    from_node_strategy.get_outvar_strtg(idx) == \
                                    to_node_strategy.get_invar_strtg(input_nodes.index(comm_node)):
                                    to_node.replace_input_with(comm_node, from_node)
                                    continue
                            comm_queue.add((from_node, comm_node, to_node))

    # node just computed -> commnications followed
    comm_map = {}

    # can be significantly improved
    for (from_node, comm_node, to_node) in comm_queue:
        if comm_map.get(from_node) is None:
            comm_map[from_node] = []
        comm_map[from_node].append((from_node, comm_node, to_node))

    def redist_tensor_func_transformed(input_tensors: list, input_specs: list):
        res = []
        device_mesh = get_device_mesh()
        res_cache: dict[DTensor, list] = {}
        for input_tensor, spec in zip(input_tensors, input_specs):
            if isinstance(input_tensor, DTensor) and input_tensor.size() != torch.Size([0]):
                if spec != input_tensor._spec.placements:
                    hist = res_cache.get(input_tensor)
                    current_res = None
                    if hist != None:
                        for hist_spec, idx in hist:
                            # (CAUTION) assuming no inplace operation in graph 
                            if hist_spec == spec:
                                current_res = res[idx]
                                break
                    else:
                        res_cache[input_tensor] = []
                    if current_res is None:
                        current_res = input_tensor.redistribute(
                            device_mesh, spec).contiguous()
                        res_cache[input_tensor].append((spec, len(res)))
                    res.append(current_res)
                    continue
            res.append(input_tensor.contiguous())
        return res

    # add new comm node after nodes that need comms after computation
    for node in comm_map:
        comm_list = comm_map[node]
        # redundancy remained
        input_nodes = [n for (n, _, _) in comm_list]
        input_specs = [n.args[1] for (_, n, _) in comm_list]
        with fx_module.graph.inserting_after(node):
            new_node = fx_module.graph.call_function(
                redist_tensor_func_transformed, args=(input_nodes, input_specs))
        for idx, (_, comm_node, to_node) in enumerate(comm_list):
            with fx_module.graph.inserting_after(new_node):
                idx_node = fx_module.graph.call_function(operator.getitem, args=(new_node, idx))
            to_node.replace_input_with(comm_node, idx_node)

    # at this point all old comm operators should be eliminated
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    logger.info("Communication Optimization: Done!")
    return fx_module
