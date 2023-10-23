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
import torch.utils._pytree as pytree

from functools import reduce

import torch
from sortedcontainers import SortedList
from torch.distributed._tensor.api import DTensor, Replicate
from torch.fx.node import _get_qualified_name
from torch.distributed._tensor import DeviceMesh

from easydist.torch.device_mesh import get_device_mesh, set_device_mesh
from easydist.torch.graph_profile import HyperPerfMeasure, to_meta
from easydist.torch.passes.rule_override import _transform_to_Placement
import easydist.torch.rcpsp as rcpsp
import easydist.config as mdconfig

logger = logging.getLogger(__name__)

'''
currently get maximum bandwidth through 
communicating a large tensor(4096 * 1024 * 16)
'''
def bandwidth_profile():
    iter_time = 10
    comm_v = iter_time * 2 * 4096 * 1024 * 16 * 4.0
    res_t = 0.0
    for _ in range(0, iter_time):
        with torch.device('cuda'):
            t = torch.randn(4096, 1024, 16)
        torch.distributed.barrier()
        start_t = time.perf_counter()
        torch.distributed.all_reduce(t)
        torch.distributed.barrier()
        res_t += time.perf_counter() - start_t
    return comm_v / res_t

def graph_profile(fx_module: torch.fx.GraphModule, shape_info):

    # profile nodes in a graph
    perf = HyperPerfMeasure(fx_module)
    device_mesh = get_device_mesh()
    if not isinstance(device_mesh, DeviceMesh):
        raise RuntimeError("MockDeviceMesh in comm_optimize. Please set device to torch DeviceMesh")
    # TODO can be modified to reduce args memory in total
    placements = [Replicate()] * device_mesh.ndim
    args = [DTensor.from_local(torch.randn(arg['shape'], dtype=arg['dtype']),
                               device_mesh, placements) \
                               if arg else None \
                               for n, arg in shape_info.items() \
                               if n.__contains__('arg')]
    perf.run(*args)
    node_profile = perf.node_runtime()
    
    # get the bandwidth
    bandwidth = bandwidth_profile() * 0.5
    
    # calculate communication time
    for node in fx_module.graph.nodes:
        if _is_comm_node(node):
            if node.all_input_nodes[0].name.__contains__('to_dtensor'):
                node_profile[node.name] = 0.005
                continue
            input_info = shape_info[node.all_input_nodes[0].name]
            t = 0.0
            if isinstance(input_info, list):
                for itm in input_info:
                    t += reduce((lambda x, y: x * y), itm['shape']) * 2 * 4 / bandwidth
            elif input_info is not None:
                t = reduce((lambda x, y: x * y), input_info['shape'], 0.002 * bandwidth) * 2 * 4 / bandwidth
            node_profile[node.name] = t
    
    # round the profile result to meet or-tools need
    int_node_profile = {}
    min_t = min(node_profile.values())
    for key in node_profile:
        int_node_profile[key] = min(int(node_profile[key] / min_t) + 20, 32768)
        assert(int_node_profile[key] > 0)

    if mdconfig.log_level <= logging.DEBUG:
        logger.info(f'bandwidth:{bandwidth}')
        for node_name, _time in int_node_profile.items():
            logger.info(f'{node_name}: {_time}')
        logger.info(max(int_node_profile.values()))
    
    return int_node_profile

def _is_comm_node(node):
    return node.op == 'call_function' and \
            _get_qualified_name(node.target) == \
            'easydist.torch.passes.sharding.redist_tensor_func'

def _output_strategy(node_list, opt_strategy):
    specs = []
    for node in node_list:
        if opt_strategy.get(node.name) is not None:
            strtgy = opt_strategy[node.name]['strategy']
            out_strtgies = strtgy.out_strtg_group.var_spmd_strategy_group
            spec = []
            for out_strtgy in out_strtgies:
                spec.append(list(_transform_to_Placement(out_strtgy)))
            if len(spec) == 1:
                spec = spec[0]
        else:
            spec = None
        specs.append(spec)
    return specs


'''
input:
fx_module: GraphModule goint to be scheduled
shape_info: Output shape of each node
output:
a tuple list where the first element denotes mode(comm/comp)
and the second denotes which node
'''
def rcpsp_trial(fx_module: torch.fx.GraphModule, shape_info):

    processing_time = graph_profile(fx_module, shape_info)

    # prepare RCPSP input
    task_data = []
    available_resources = {'comm': 1, 'comp': 1, 'mem': int(0.9 * mdconfig.available_mem)}
    
    # whether resource release only until all nodes depended on it have finished
    resource_dep_mask = [0, 0, 1]
    precedence_relations = []

    arg_num = 0
    arg_list = []
    for node in fx_module.graph.nodes:
        if node.name.__contains__('arg'):
            arg_list.append(node)
            arg_num += 1
            continue
        resource = []
        if _is_comm_node(node):
            duration = processing_time[node.name]
            resource.append(('comm', 1))

            pre_node = node.all_input_nodes[0]
            while shape_info.get(pre_node.name) is None:
                pre_node = pre_node.all_input_nodes[0]

            shape_node = pre_node
        else:
            if node.name in processing_time:
                duration = processing_time[node.name]
            else:
                duration = 20
            resource.append(('comp', 1))

            shape_node = node

        if shape_info.get(shape_node.name) is not None:
            mem_req = 0
            outputs = shape_info[shape_node.name]
            if isinstance(outputs, tuple):
                outputs = list(outputs)
            elif not isinstance(outputs, list):
                outputs = [outputs]
            for output in outputs:
                if output.get('shape') is not None:
                    mem_req += int(reduce(lambda x, y: x * y, output['shape'], 1) / 1024)
            #resource.append(('mem', mem_req))
        
        precedence = []
        for pre in node.all_input_nodes:
            if not pre.name.__contains__('arg'):
                precedence.append(pre)
        precedence_relations.append(precedence)
        task_data.append((node, duration, precedence, resource))

    assert(len(task_data) == len(fx_module.graph.nodes) - arg_num)

    # only rank 0 process do the calculation
    if torch.distributed.get_rank() == 0:
        logger.info('enter rcpsp')
        logger.info(f'task cnt: {len(task_data)}')
        start_t = time.perf_counter()
        raw_sche = rcpsp.rcpsp(task_data, available_resources, 
                               resource_dep_mask, 'general')
        logger.info(f"[RCPSP.time]:\t {time.perf_counter() - start_t} s.")
        logger.info('exit rcpsp')

        assert(len(raw_sche) == len(fx_module.graph.nodes) - arg_num)
    else:
        raw_sche = [None] * (len(fx_module.graph.nodes) - arg_num)
    torch.distributed.broadcast_object_list(raw_sche, src=0, device="cuda")

    node_sche = [task_data[i][0] for i in raw_sche]

    sche = arg_list + node_sche
    
    assert(len(sche) == len(fx_module.graph.nodes))

    return sche

def comm_group(sche, shape_info, cap_limit, rg_limit):
    idx = len(sche) - 1
    cur_cap = 0
    cur_range = 0
    cur_comm_list = []
    comm_list_dep = []
    while idx >= 0:
        cur_range += 1
        if not _is_comm_node(sche[idx]):
            # check dependency
            if sche[idx] in comm_list_dep:
                cur_comm_list.reverse()
                sche = sche[:idx + 1] + cur_comm_list + sche[idx + 1:]
                cur_cap = 0
                cur_range = 0
                cur_comm_list = []
                comm_list_dep = []
            idx -= 1
            continue
        
        if cur_range > rg_limit or cur_cap > cap_limit:
            cur_comm_list.reverse()
            sche = sche[:idx + 1] + cur_comm_list + sche[idx + 1:]
            cur_cap = 0
            cur_range = 0
            cur_comm_list = []
            comm_list_dep = []

        node = sche[idx]
        pre_node = node.all_input_nodes[0]
        while shape_info.get(pre_node.name) is None:
            pre_node = pre_node.all_input_nodes[0]
        comm_vol = reduce(lambda x, y: x * y, 
                          shape_info[pre_node.name]['shape'], 1)

        if comm_vol < cap_limit:
            cur_cap += comm_vol
            cur_comm_list.append(node)
            comm_list_dep.append(pre_node)

        idx -= 1
    assert(len(cur_comm_list) == 0)
    return sche

def comm_optimize(fx_module: torch.fx.GraphModule, shape_info=None, opt_strategy=None, grouping=False):
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    node_to_rank: dict[torch.fx.Node, int] = {}
    # processing
    rank = 0
    for node in fx_module.graph.nodes:
        if _is_comm_node(node):
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
                    if _is_comm_node(predecesor):
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

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    # node just computed -> commnications followed
    comm_map = {}
    comm_strtg = 'rcpsp'
    if comm_strtg == 'eager':
        for (from_node, comm_node, to_node) in comm_queue:
            if comm_map.get(from_node) is None:
                comm_map[from_node] = []
            comm_map[from_node].append((from_node, comm_node, to_node))
    elif comm_strtg == 'rcpsp':
        sche = rcpsp_trial(fx_module, shape_info)
        
        #sche = comm_group(sche, shape_info, 1024 * 1024, 40)
        
        # schedule node topological order according to sche
        fx_module.graph._root._next = sche[0]
        sche[0]._prev = fx_module.graph._root
        for idx, node in enumerate(sche):
            if idx + 1 < len(sche):
                node._next = sche[idx + 1]
                sche[idx + 1]._prev = node
            else:
                node._next = fx_module.graph._root
                fx_module.graph._root._prev = node
                break
            if not _is_comm_node(node) and _is_comm_node(sche[idx + 1]):
                comm_map[node] = []
                for follower in sche[idx + 1:]:
                    if _is_comm_node(follower):
                        comm_map[node].append((follower.all_input_nodes[0],
                                               follower,
                                               comm_dest[follower]))
                    else:
                        break
                assert(len(comm_map[node]) > 0)

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
        input_ori_specs = _output_strategy(input_nodes, opt_strategy)
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
