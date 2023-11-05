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
from torch.distributed._tensor.placement_types import DTensorSpec

import easydist
from easydist.torch.device_mesh import get_device_mesh, set_device_mesh
from easydist.torch.graph_profile import HyperPerfMeasure, to_meta
from easydist.torch.passes.rule_override import _transform_to_Placement
import easydist.torch.rcpsp as rcpsp
import easydist.torch.comm_rules as comm_rules
from easydist.torch.comm_rules import comm_redirect as comm_redirect
from easydist.torch.utils import EDInfo, EDNodeType
import easydist.config as mdconfig
from easydist.metashard.metair import (
    SPMD,
    NodeSPMDStrategy,
    VarSPMDStrategyGroup,
    VarSPMDStrategy,
)

logger = logging.getLogger(__name__)

'''
currently get maximum bandwidth through 
communicating a large tensor(4096 * 1024 * 16)
'''
def bandwidth_profile():
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''
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
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''

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
def rcpsp_schedule(fx_module: torch.fx.GraphModule, shape_info):
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''

    processing_time = graph_profile(fx_module, shape_info)
    negligible_duration = 20

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

        if node.ed_info.is_communication():
            duration = processing_time[node.name]
            resource.append(('comm', 1))

            shape_node = node.all_input_nodes[0]
        else:
            if node.name in processing_time:
                duration = processing_time[node.name]
            else:
                duration = negligible_duration
            resource.append(('comp', 1))

            shape_node = node

        mem_req = 0
        outputs = shape_info[shape_node.name]
        if isinstance(outputs, tuple):
            outputs = list(outputs)
        elif not isinstance(outputs, list):
            outputs = [outputs]
        for output in outputs:
            if output.get('shape') is not None:
                mem_req += int(reduce(lambda x, y: x * y, output['shape'], 1) * 4 / 1024)
        resource.append(('mem', mem_req))
        
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

def comm_nodes_group(fx_module, node_list):
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.


'''
    if len(node_list) <= 1:
        return
    # group the node and add proper decouple node to both the graph and the schedule
    sche = [node for node in fx_module.graph.nodes]
    from_nodes = [node.all_input_nodes[0] for node in node_list]
    to_nodes = [node.ed_info.comm_meta['to_node'] for node in node_list]
    total_size = 0
    retrive_points = []
    retrive_shapes = []
    for node in node_list:
        comm_vol = node.ed_info.comm_meta['comm_vol']
        comm_shape = node.ed_info.comm_meta['comm_shape']
        retrive_points.append(comm_vol)
        retrive_shapes.append(comm_shape)
        total_size += comm_vol

    def comm_couple(tensor_list):
        flattened_tensor_list = [t.flatten() for t in tensor_list]
        return torch.cat(tuple(flattened_tensor_list))

    def comm_decouple(tensor, retrive_points, retrive_shapes):
        tensor_list = torch.split(tensor, retrive_points)
        return [tensor.reshape(shape) for tensor, shape in zip(tensor_list, retrive_shapes)]

    to_node = sche[min([sche.index(to_node) for to_node in to_nodes])]

    with fx_module.graph.inserting_before(node_list[0]):
        comm_args = list(node_list[0].args[1:])
        new_from_node = fx_module.graph.call_function(
            comm_couple, args=tuple([from_nodes])
        )
        new_from_node.ed_info = EDInfo()
        new_from_node.ed_info.node_type = EDNodeType.COMPUTATION
        comm_op_name = _get_qualified_name(node_list[0].target)
        new_comm_node = fx_module.graph.call_function(
            eval(comm_op_name), 
            args=tuple([new_from_node] + comm_args)
        )

    with fx_module.graph.inserting_before(to_node):
        new_to_node = fx_module.graph.call_function(
            comm_decouple, args=(new_comm_node, tuple(retrive_points), tuple(retrive_shapes))
        )
        new_to_node.ed_info = EDInfo()
        new_to_node.ed_info.node_type = EDNodeType.COMPUTATION
        
    new_comm_node.ed_info = EDInfo()
    new_comm_node.ed_info.node_type = EDNodeType.COMMUNICATION
    new_comm_node.ed_info.comm_meta = {
        'to_node': new_to_node,
        'comm_vol': total_size,
        'comm_shape': torch.Size([total_size])
    }
    
    for idx, (comm_node, to_node) in enumerate(zip(node_list, to_nodes)):
        with fx_module.graph.inserting_before(to_node):
            retrive_node = fx_module.graph.call_function(
                operator.getitem, args=(new_to_node, idx)
            )
        to_node.replace_input_with(comm_node, retrive_node)
        retrive_node.ed_info = EDInfo()
        retrive_node.ed_info.node_type = EDNodeType.COMPUTATION

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

def groupable(n1, n2):
    n1_op_name = _get_qualified_name(n1.target)
    n2_op_name = _get_qualified_name(n2.target)
    return n1_op_name == n2_op_name and n1.args[1:] == n2.args[1:]

def comm_group(fx_module, cap_limit, rg_limit):
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''
    sche = [node for node in fx_module.graph.nodes]
    idx = len(sche) - 1
    cur_cap = 0
    cur_range = 0
    cur_comm_list = []
    comm_list_dep = []
    retrive_node = None
    # TODO return a map that map a start node to an end node (add if)
    while idx >= 0:
        cur_range += 1
        if not sche[idx].ed_info.is_communication():
            # check dependency
            if sche[idx] in comm_list_dep or \
                cur_range > rg_limit or \
                cur_cap > cap_limit:
                
                cur_comm_list.reverse()
                comm_nodes_group(fx_module, cur_comm_list)
                sche = [node for node in fx_module.graph.nodes]

                cur_cap = 0
                cur_range = 0
                cur_comm_list = []
                comm_list_dep = []

            if retrive_node:
                idx = sche.index(retrive_node)
                retrive_node = None
            else:
                idx -= 1
            continue
        
        if cur_range > rg_limit or cur_cap > cap_limit:

            cur_comm_list.reverse()
            comm_nodes_group(fx_module, cur_comm_list)
            sche = [node for node in fx_module.graph.nodes]

            cur_cap = 0
            cur_range = 0
            cur_comm_list = []
            comm_list_dep = []
            if retrive_node:
                idx = sche.index(retrive_node)
                retrive_node = None
                continue

        node = sche[idx]
        comm_vol = node.ed_info.comm_meta['comm_vol']

        if comm_vol < cap_limit:
            if len(cur_comm_list) == 0 or \
                groupable(node, cur_comm_list[0]):
                cur_cap += comm_vol
                del sche[idx]
                cur_comm_list.append(node)
                comm_list_dep.append(node.all_input_nodes[0])
            elif retrive_node is None:
                retrive_node = node

        idx -= 1

def comm_optimize(fx_module: torch.fx.GraphModule, shape_info, opt_strategy, grouping=False, mem_restrain=False):
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if mdconfig.log_level <= logging.DEBUG:
        fx_module.print_readable()

    for node in fx_module.graph.nodes:
        if node.ed_info.is_communication():
            assert len(node.all_input_nodes) == 1
            from_node = node.all_input_nodes[0]
            comm_shape = shape_info[from_node.name]['shape']
            node.ed_info.comm_meta = {'comm_vol': reduce(lambda x, y: x * y, comm_shape, 1), 
                                      'comm_shape': comm_shape}
        elif node.ed_info.is_computation():
            for pre in node.all_input_nodes:
                if pre.ed_info.is_communication():
                    pre.ed_info.comm_meta['to_node'] = node

    if torch.distributed.get_rank() == 3:
        fx_module.print_readable()
        for node in fx_module.graph.nodes:
            print(node.name)
            print(node.ed_info)
    #_shapeinfo_fill_up(shape_info, fx_module)
    #_strategy_fill_up(opt_strategy, shape_info, fx_module)

    grouping = True
    if grouping:
        comm_group(fx_module, 1024 * 1024, 10000)
        fx_module.graph.eliminate_dead_code()
        fx_module.recompile()

    # node just computed -> commnications followed
    comm_strtg = 'eager'
    if comm_strtg == 'eager':
        comm_map = {}
        for node in fx_module.graph.nodes:
            if node.ed_info.is_communication():
                from_node = node.all_input_nodes[0]
                if comm_map.get(from_node) is None:
                    comm_map[from_node] = []
                comm_map[from_node].append((from_node, node))
    elif comm_strtg == 'rcpsp':
        sche = rcpsp_schedule(fx_module, shape_info)
        
        _link_nodes(fx_module, sche)

        for idx, node in enumerate(sche):
            if not _is_comm_node(node) and _is_comm_node(sche[idx + 1]):
                comm_map[node] = []
                for follower in sche[idx + 1:]:
                    if _is_comm_node(follower):
                        comm_map[node].append((comm_info[follower]['from_node'],
                                               follower,
                                               comm_info[follower]['to_node']))
                    else:
                        break
                assert(len(comm_map[node]) > 0)

    
    def grouped_comm(input_tensors: list, comm_func: list, comm_args: list):
        res = []
        for input_tensor, comm_func, args in zip(input_tensors, comm_func, comm_args):
            res.append(eval(comm_func)(input_tensor, *args))
        return res

    # add new comm node after nodes that need comms after computation
    for node in comm_map:
        #if len(comm_map[node] <= 1):
        #    continue

        # redundancy remained
        input_nodes = [n for (n, _) in comm_map[node]]
        comm_funcs = [_get_qualified_name(n.target) for (_, n) in comm_map[node]]
        comm_args = [n.args[1:] for (_, n) in comm_map[node]]

        with fx_module.graph.inserting_after(node):
            new_comm_node = fx_module.graph.call_function(
                grouped_comm, args=(input_nodes, comm_funcs, comm_args))
        for idx, (_, comm_node) in enumerate(comm_map[node]):
            with fx_module.graph.inserting_after(new_comm_node):
                idx_node = fx_module.graph.call_function(operator.getitem, args=(new_comm_node, idx))
            comm_node.ed_info.comm_meta['to_node'].replace_input_with(comm_node, idx_node)

    # at this point all old comm operators should be eliminated
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if torch.distributed.get_rank() == 0:
        logger.info("Communication Optimization: Done!")
    return fx_module

def _strategy_fill_up(opt_strategy, shape_info, fx_module):
    for node in fx_module.graph.nodes:
        if not _is_comm_node(node):
            if opt_strategy.get(node.name) is None:
                if node.name.__contains__("getitem"):
                    idx = node.args[1]
                    pre_node = node.all_input_nodes[0]
                    opt_strategy[node.name] = {
                        'node': node.name,
                        'strategy': NodeSPMDStrategy(
                            VarSPMDStrategyGroup(
                                opt_strategy[pre_node.name]['strategy'].get_invar_strtg(idx)
                            ),
                            VarSPMDStrategyGroup(
                                opt_strategy[pre_node.name]['strategy'].get_outvar_strtg(idx)
                            )
                        )
                    }
                elif not (node.name.__contains__("arg") or node.name.__contains__("output")):
                    # Assumption: nodes without strategy and not being args or output are constant tensor
                    opt_strategy[node.name] = {
                        'node': node, 
                        'strategy': NodeSPMDStrategy(
                            VarSPMDStrategyGroup(
                                VarSPMDStrategy(*tuple([SPMD('REPLICATE')] * len(shape_info[node.name]['shape'])))
                            ),
                            VarSPMDStrategyGroup(
                                VarSPMDStrategy(*tuple([SPMD('REPLICATE')] * len(shape_info[node.name]['shape'])))
                            )
                        )    
                    }
                    #print(node.name)
                    #print(opt_strategy[node.name])

    if mdconfig.log_level <= logging.DEBUG:
        print(f"opt_strategy: {opt_strategy}")

def _shapeinfo_fill_up(shape_info, fx_module):
    for node in fx_module.graph.nodes:
        if not _is_comm_node(node):
            if shape_info.get(node.name) is None:
                if node.name.__contains__("to_dtensor"):
                    pre_node = node.all_input_nodes[0]
                    shape_info[node.name] = shape_info[pre_node.name]
                elif torch.distributed.get_rank() == 0:
                    print(node.name)
                    raise RuntimeError("_shapeinfo_fill_up: unmet node!")

    if mdconfig.log_level <= logging.DEBUG:
        print(f"shape_info: {shape_info}")


def _link_nodes(fx_module, node_list):
    '''
    Change the topological order of fx_module according to node_list
    '''
    fx_module.graph._root._next = node_list[0]
    node_list[0]._prev = fx_module.graph._root
    for idx, node in enumerate(node_list[:-1]):
        node._next = node_list[idx + 1]
        node_list[idx + 1]._prev = node
    node_list[-1]._next = fx_module.graph._root
    fx_module.graph._root._prev = node_list[-1]
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()