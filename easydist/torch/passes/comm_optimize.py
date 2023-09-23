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
import time
import random

from functools import reduce

import torch
from sortedcontainers import SortedList
from torch.distributed._tensor.api import DTensor, Replicate
from torch.fx.node import _get_qualified_name

from easydist.torch.device_mesh import get_device_mesh
from easydist.torch.rcpsp import RCPSP
from easydist.torch.graph_profile import HyperPerfMeasure

logger = logging.getLogger(__name__)

def rcpsp_trial(fx_module: torch.fx.GraphModule, shape_info):
    perf = HyperPerfMeasure(fx_module)
    device_mesh = get_device_mesh()
    placements = [Replicate()] * device_mesh.ndim
    args = [DTensor.from_local(torch.randn(arg['shape'], dtype=arg['dtype']), device_mesh, placements) if arg else None for n, arg in shape_info.items() if n.__contains__('arg')]
    perf.run(*args)
    node_profile = perf.node_runtime()
    for node in fx_module.graph.nodes:
        if node.op == 'call_function' and \
            _get_qualified_name(node.target) == \
            'easydist.torch.passes.sharding.redist_tensor_func':
            if node.all_input_nodes[0].name.__contains__('to_dtensor'):
                node_profile[node.name] = 0.005
                continue
            input_info = shape_info[node.all_input_nodes[0].name]
            t = 0.0
            bandwidth = 4.0 * 1024 * 1024
            if isinstance(input_info, list):
                for itm in input_info:
                    t += reduce((lambda x, y: x * y), itm['shape']) / bandwidth
            elif input_info is not None:
                t = reduce((lambda x, y: x * y), input_info['shape'], 0.002 * bandwidth) / bandwidth
            node_profile[node.name] = t
    int_node_profile = {}
    for key in node_profile:
        int_node_profile[key] = int(node_profile[key] * 1000)
        assert(int_node_profile[key] > 0)

    print(int_node_profile)
    #print(perf.node_runtime())
    node_to_rank = {}
    rank_to_node = {}
    print(type(fx_module.graph.nodes))
    for rank, node in enumerate(fx_module.graph.nodes):
        node_to_rank[node] = rank
        rank_to_node[rank] = node

    jobs_data = []
    dependencies = []
    ttt = []
    for node in fx_module.graph.nodes:
        if node.op == 'call_function' and \
            _get_qualified_name(node.target) == \
            'easydist.torch.passes.sharding.redist_tensor_func':
            #jobs_data.append([(0, random.randint(5, 7))])
            jobs_data.append([(0, int_node_profile[node.name])])
        else:
            if node.name in int_node_profile:
                #jobs_data.append([(1, random.randint(1,5))])
                jobs_data.append([(1, int_node_profile[node.name])])
            else:
                jobs_data.append([(1, 20)])

        for pre in node.all_input_nodes:
            dependencies.append(((node_to_rank[pre], 0), (node_to_rank[node], 0)))
            ttt.append((pre, node))
    print('dependencies')
    print(ttt)
    assert(len(jobs_data) == len(fx_module.graph.nodes))
    logger.info('enter rcpsp')
    logger.info('job cnt:')
    logger.info(len(jobs_data))
    logger.info('dependency cnt:')
    logger.info(len(dependencies))
    start_t = time.perf_counter()
    raw_sche = RCPSP(jobs_data, dependencies)
    logger.info(f"[RCPSP.time]:\t {time.perf_counter() - start_t} s.")
    logger.info('exit rcpsp')

    sche = [(mode, rank_to_node[job]) for mode, job in raw_sche]
    print('sche')
    print(sche)
    assert(len(sche) == len(fx_module.graph.nodes))
    return sche

def comm_optimize(fx_module: torch.fx.GraphModule, shape_info=None, opt_strategy=None, grouping=False):
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()
    print(fx_module)
    print('root')
    print(fx_module.graph._root)

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

    comm_dest = {}
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
                            
                            comm_dest[comm_node] = to_node

                            # code needed to eliminate redundant comm node
                            if opt_strategy is not None and from_node.op == 'call_function':
                                target_node = from_node
                                idx = 0
                                if from_node.name.__contains__('getitem'):
                                    pppredecesor = from_node.all_input_nodes
                                    assert(len(pppredecesor) == 1)
                                    target_node = pppredecesor[0]
                                    idx = from_node.args[1]
                                if opt_strategy.get(to_node.name) is not None and \
                                    opt_strategy.get(target_node.name) is not None:
                                    to_node_strategy = opt_strategy[to_node.name]['strategy']
                                    from_node_strategy = opt_strategy[target_node.name]['strategy']
                                    if from_node_strategy is not None and \
                                        from_node_strategy.get_outvar_strtg(idx) == \
                                        to_node_strategy.get_invar_strtg(input_nodes.index(comm_node)):
                                        to_node.replace_input_with(comm_node, from_node)
                                        continue
                            comm_queue.add((from_node, comm_node, to_node))

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    # node just computed -> commnications followed
    print('fx_module')
    print(fx_module)
    comm_map = {}
    comm_strtg = 'rcpsp'
    if comm_strtg == 'eager':
        for (from_node, comm_node, to_node) in comm_queue:
            if comm_map.get(from_node) is None:
                comm_map[from_node] = []
            comm_map[from_node].append((from_node, comm_node, to_node))
    elif comm_strtg == 'rcpsp':
        sche = rcpsp_trial(fx_module, shape_info)
        
        print('root')
        print(fx_module.graph._root)
        fx_module.graph._root._next = sche[0][1]
        sche[0][1]._prev = fx_module.graph._root
        for idx, (mode, node) in enumerate(sche):
            if idx + 1 < len(sche):
                node._next = sche[idx + 1][1]
                sche[idx + 1][1]._prev = node
            else:
                node._next = fx_module.graph._root
                fx_module.graph._root._prev = node
                break
            if mode == 'comp' and sche[idx + 1][0] == 'comm':
                comm_map[node] = []
                for follower in sche[idx + 1 :]:
                    if follower[0] == 'comm':
                        comm_map[node].append((follower[1].all_input_nodes[0],
                                               follower[1],
                                               comm_dest[follower[1]]))
                    else:
                        break
                assert(len(comm_map[node]) > 0)
        for node in fx_module.graph.nodes:
            print(node)
        fx_module.graph.eliminate_dead_code()
        fx_module.recompile()

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
