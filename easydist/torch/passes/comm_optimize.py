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

        if _is_comm_node(node):
            duration = processing_time[node.name]
            resource.append(('comm', 1))

            shape_node = pre_node
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

def comm_nodes_group(fx_module, node_list, comm_info):
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
    print(node_list)
    sche = [node for node in fx_module.graph.nodes]
    from_nodes = [comm_info[node]['from_node'] for node in node_list]
    to_nodes = [comm_info[node]['to_node'] for node in node_list]
    total_size = 0
    retrive_points = []
    retrive_shapes = []
    for node in node_list:
        comm_vol = comm_info[node]['comm_meta'].comm_vol
        comm_shape = comm_info[node]['comm_meta'].shape
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
        comm_args = list(comm_info[node_list[0]]['comm_meta'].comm_args)
        new_from_node = fx_module.graph.call_function(
            comm_couple, args=tuple([from_nodes])
        )
        new_comm_node = fx_module.graph.call_function(
            eval(comm_info[node_list[0]]['comm_meta'].op_meta[0]), 
            args=tuple([new_from_node] + comm_args)
        )

    with fx_module.graph.inserting_before(to_node):
        new_to_node = fx_module.graph.call_function(
            comm_decouple, args=(new_comm_node, tuple(retrive_points), tuple(retrive_shapes))
        )
        
    comm_info[new_comm_node] = {'from_node': new_from_node,
                                'to_node':   new_to_node,
                                'comm_meta':  comm_meta(new_comm_node.name,
                                                       torch.Size([total_size]),
                                                       comm_info[node_list[0]]['comm_meta'].op_meta,
                                                       *comm_info[node_list[0]]['comm_meta'].comm_args)}
    
    for idx, (comm_node, to_node) in enumerate(zip(node_list, to_nodes)):
        with fx_module.graph.inserting_before(to_node):
            retrive_node = fx_module.graph.call_function(
                operator.getitem, args=(new_to_node, idx)
            )
        to_node.replace_input_with(comm_node, retrive_node)

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    for node in node_list:
        del comm_info[node]

def comm_group(fx_module, comm_info, cap_limit, rg_limit):
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
        if not _is_comm_node(sche[idx]):
            # check dependency
            if sche[idx] in comm_list_dep or \
                cur_range > rg_limit or \
                cur_cap > cap_limit:
                
                cur_comm_list.reverse()
                comm_nodes_group(fx_module, cur_comm_list, comm_info)
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
            comm_nodes_group(fx_module, cur_comm_list, comm_info)
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
        comm_vol = comm_info[node]['comm_meta'].comm_vol

        if comm_vol < cap_limit:
            if len(cur_comm_list) == 0 or \
                comm_info[node]['comm_meta'] == comm_info[cur_comm_list[0]]['comm_meta']:
                cur_cap += comm_vol
                del sche[idx]
                cur_comm_list.append(node)
                comm_list_dep.append(comm_info[node]['from_node'])
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
    #if torch.distributed.get_rank() == 0:
    #    fx_module.print_readable()

    if mdconfig.log_level <= logging.DEBUG:
        fx_module.print_readable()

    _shapeinfo_fill_up(shape_info, fx_module)
    _strategy_fill_up(opt_strategy, shape_info, fx_module)

    # comm_node -> {from_node, to_node, comm_meta}
    comm_info = {}

    # assume single input, single consumer communication node
    if mdconfig.use_dtensor:
        # remove redundancy
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
                            
                            output_plac = comm_node.args[1]
                            # TODO check
                            from_node_placements = opt_strategy[from_node.name]['strategy'].get_outvar_strtg(0)
                            input_plac = comm_rules.to_DTensorPlacements(from_node_placements)

                            if input_plac == output_plac:
                                to_node.replace_input_with(comm_node, from_node)
                            else:
                                comm_op_name = _get_qualified_name(comm_node.target)
                                comm_info[comm_node] = {
                                    'from_node': from_node,
                                    'to_node':   to_node,
                                    'comm_meta': comm_meta(
                                        comm_node.name, 
                                        shape_info[from_node.name]['shape'], 
                                        (comm_op_name, input_plac, output_plac),
                                        *list(comm_node.args)[1:]
                                    )
                                }

        fx_module.graph.eliminate_dead_code()
        fx_module.recompile()

    #if torch.distributed.get_rank() == 0:
    #    fx_module.print_readable()

    grouping = True
    if grouping:
        comm_group(fx_module, comm_info, 1024 * 1024, 10000)
        fx_module.graph.eliminate_dead_code()
        fx_module.recompile()

    #if torch.distributed.get_rank() == 0:
        #sche = [node for node in fx_module.graph.nodes]
        #print(sche)
        #fx_module.print_readable()
        #exit(1)
    return fx_module

    # node just computed -> commnications followed
    comm_map = {}
    comm_strtg = 'rcpsp'
    if comm_strtg == 'eager':
        for comm_node in comm_info:
            if comm_map.get(from_node) is None:
                comm_map[from_node] = []
            comm_map[from_node].append((comm_info[comm_node]['from_node'],
                                        comm_node, 
                                        comm_info[comm_node]['to_node']))
    elif comm_strtg == 'rcpsp':
        sche = rcpsp_schdule(fx_module, comm_info)
        
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

    
    def redist_tensor_func_transformed(input_tensors: list, input_specs: list):
        res = []
        device_mesh = get_device_mesh()
        #res_cache: dict[DTensor, list] = {}
        for input_tensor, spec in zip(input_tensors, input_specs):
            if isinstance(input_tensor, DTensor) and input_tensor.size() != torch.Size([0]):
                if spec != input_tensor._spec.placements:
                    '''
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
                        res_cache[input_tensor].append((spec, len(res)))
                    '''
                    local_tensor = input_tensor._local_tensor
                    current_spec = input_tensor._spec
                    target_spec = DTensorSpec(
                        device_mesh, tuple(spec), tensor_meta=input_tensor._spec.tensor_meta
                    )
                    comm_steps = comm_rules.redist_steps(device_mesh, current_spec.shape, 
                                                         current_spec.placements, target_spec.placements)
                    res_tensor = local_tensor
                    for (comm_step, args) in comm_steps:
                        res_tensor = comm_redirect[comm_step](*args, res_tensor, device_mesh)
                    current_res = DTensor(
                        res_tensor,
                        device_mesh,
                        target_spec.placements,
                        shape=input_tensor.shape,
                        dtype=input_tensor.dtype,
                        requires_grad=input_tensor.requires_grad,
                        stride=input_tensor.stride(),
                    )
                    res.append(current_res)
                    continue
            res.append(input_tensor)
        return res

    # add new comm node after nodes that need comms after computation
    for node in comm_map:
        comm_list = comm_map[node]

        # redundancy remained
        input_nodes = [n for (n, _, _) in comm_list]
        input_ori_specs = _output_strategy(input_nodes, opt_strategy)
        input_specs = [n.args[1] for (_, n, _) in comm_list]

        # TODO do the grouping, grouped as a tuple
        # 1 make the rules
        # 2 select nodes from the tuple according to rules
        # 3 add a node for each group as a decoupling state before the fisrt node need
        # 4 change input of such nodes
        #input_nodes = []

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

class comm_meta:
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''

    def __init__(self, name, shape, op_meta, *args):
        self.name = name
        self.shape = shape
        self.comm_vol = reduce(lambda x, y: x * y, shape, 1)
        self.op_meta = op_meta
        self.comm_args = args
    
    def __eq__(self, other):
        if len(self.comm_args) != len(other.comm_args):
            return False
        if self.op_meta != other.op_meta:
            return False
        for i, j in zip(self.comm_args, other.comm_args):
            if i != j:
                return False
        return True

    def __str__(self):
        return f"Communication_{self.name}: (shape: {self.shape}, op_meta: {self.op_meta}, comm_args: {self.comm_args})"

    def __repr__(self) -> str:
        return self.__str__()
    