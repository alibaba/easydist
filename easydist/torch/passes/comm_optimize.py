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
from functools import reduce

import torch
from torch.fx.node import _get_qualified_name

import easydist
import easydist.config as mdconfig
import easydist.torch.rcpsp as rcpsp
from easydist.torch.passes.sharding import create_meta_from_node
from easydist.torch.passes.runtime_prof import runtime_prof
from easydist.torch.utils import EDInfo, EDNodeType
from easydist.metashard.metair import (
    SPMD,
    NodeSPMDStrategy,
    VarSPMDStrategyGroup,
    VarSPMDStrategy,
)

logger = logging.getLogger(__name__)


def bandwidth_profile():
    '''
    Currently get maximum bandwidth through communicating a large tensor(4096 * 1024 * 16)
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


def rcpsp_schedule(fx_module: torch.fx.GraphModule, shape_info, mem_constrain):
    '''
    This function returns the best schedule executing given graph under rcpsp

    Args:
    fx_module: fx graph to be optimized
    shape_info: generated shapes info of each node (holes remain)

    Returns:
    An ordering of nodes
    '''

    runtime_prof(fx_module)

    # prepare RCPSP input
    task_data = []
    available_resources = {'comm': 1, 'comp': 1}
    if mem_constrain:
        available_resources['mem'] = int(0.95 * mdconfig.available_mem)

    # whether resource release only until all nodes depended on it have finished
    resource_dep_mask = [0, 0, 1]
    precedence_relations = []

    arg_num = 0
    arg_list = []
    for node in fx_module.graph.nodes:
        duration = node.ed_info.normalized_int_runtime_ms
        assert (duration > 0)
        if node.name.__contains__('arg'):
            arg_list.append(node)
            arg_num += 1
            continue

        resource = []

        if node.ed_info.is_communication():
            resource.append(('comm', 1))
            if mem_constrain:
                mem_req = int(node.ed_info.comm_meta['comm_vol'] / 1024)
        else:
            resource.append(('comp', 1))
            if mem_constrain:
                output_shapes = shape_info[node.name]
                if isinstance(output_shapes, tuple):
                    output_shapes = list(output_shapes)
                elif not isinstance(output_shapes, list):
                    output_shapes = [output_shapes]
                mem_req = 0
                for output_shape in output_shapes:
                    if output_shape.get('shape') is not None:
                        mem_req += int(
                            reduce(lambda x, y: x * y, output_shape['shape'], 1) * 4 / 1024)
        if mem_constrain:
            resource.append(('mem', mem_req))

        precedence = []
        for pre in node.all_input_nodes:
            if not pre.name.__contains__('arg'):
                precedence.append(pre)
        precedence_relations.append(precedence)

        task_data.append((node, duration, precedence, resource))

    assert (len(task_data) == len(fx_module.graph.nodes) - arg_num)

    # only rank 0 process do the calculation
    if torch.distributed.get_rank() == 0:
        logger.info('enter rcpsp')
        logger.info(f'task cnt: {len(task_data)}')
        start_t = time.perf_counter()
        raw_sche = rcpsp.rcpsp(task_data, available_resources, resource_dep_mask, 'general')
        logger.info(f"[RCPSP.time]:\t {time.perf_counter() - start_t} s.")
        logger.info('exit rcpsp')

        assert (len(raw_sche) == len(fx_module.graph.nodes) - arg_num)
    else:
        raw_sche = [None] * (len(fx_module.graph.nodes) - arg_num)
    torch.distributed.broadcast_object_list(raw_sche, src=0, device="cuda")

    node_sche = [task_data[i][0] for i in raw_sche]

    sche = arg_list + node_sche

    assert (len(sche) == len(fx_module.graph.nodes))

    return sche


def comm_nodes_group(fx_module, node_list, shape_info):
    '''
    Group the nodes in node_list
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
        retrive_points.append(int(comm_vol / 4))
        retrive_shapes.append(comm_shape)
        total_size += comm_vol

    def comm_couple(*tensor_list):
        flattened_tensor_list = [t.flatten() for t in tensor_list]
        return torch.cat(tuple(flattened_tensor_list))

    def comm_decouple(tensor, retrive_points, retrive_shapes):
        tensor_list = torch.split(tensor, retrive_points)
        return [tensor.reshape(shape) for tensor, shape in zip(tensor_list, retrive_shapes)]

    to_node = sche[min([sche.index(to_node) for to_node in to_nodes])]

    with fx_module.graph.inserting_before(node_list[0]):
        comm_args = list(node_list[0].args[1:])
        new_from_node = fx_module.graph.call_function(comm_couple, args=tuple(from_nodes))
        new_from_node.meta = create_meta_from_node(new_from_node)
        new_from_node.ed_info = EDInfo()
        new_from_node.ed_info.node_type = EDNodeType.COMPUTATION
        shape_info[new_from_node.name] = {'shape': torch.Size([total_size])}

        comm_op_name = _get_qualified_name(node_list[0].target)
        new_comm_node = fx_module.graph.call_function(eval(comm_op_name),
                                                      args=tuple([new_from_node] + comm_args))
        new_comm_node.meta = create_meta_from_node(new_comm_node)

    with fx_module.graph.inserting_before(to_node):
        new_to_node = fx_module.graph.call_function(comm_decouple,
                                                    args=(new_comm_node, tuple(retrive_points),
                                                          tuple(retrive_shapes)))
        new_to_node.meta = create_meta_from_node(new_to_node)
        new_to_node.ed_info = EDInfo()
        new_to_node.ed_info.node_type = EDNodeType.COMPUTATION
        shape_info[new_to_node.name] = [{'shape': s} for s in retrive_shapes]

    new_comm_node.ed_info = EDInfo()
    new_comm_node.ed_info.node_type = EDNodeType.COMMUNICATION
    new_comm_node.ed_info.comm_meta = {
        'to_node': new_to_node,
        'comm_vol': total_size,
        'comm_shape': torch.Size([total_size])
    }

    for idx, (comm_node, to_node) in enumerate(zip(node_list, to_nodes)):
        with fx_module.graph.inserting_before(to_node):
            retrive_node = fx_module.graph.call_function(operator.getitem, args=(new_to_node, idx))
        to_node.replace_input_with(comm_node, retrive_node)
        retrive_node.meta = create_meta_from_node(retrive_node)
        retrive_node.ed_info = EDInfo()
        retrive_node.ed_info.node_type = EDNodeType.COMPUTATION
        shape_info[retrive_node.name] = {'shape': retrive_shapes[idx]}

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()


def groupable(n1, n2):
    n1_op_name = _get_qualified_name(n1.target)
    n2_op_name = _get_qualified_name(n2.target)
    if not n1_op_name.__contains__('reduce'):
        return False
    return n1_op_name == n2_op_name and n1.args[1:] == n2.args[1:]


def comm_group(fx_module, cap_limit, rg_limit, shape_info):
    '''
    This function performs grouping on a fx graph

    Scan reversely searching for small comms and grouped current selected 
    nodes when either dependencies or capacity limit is to be violated.
    
    Args:
    fx_module: fx graph to be optimized
    cap_limit: 
    rg_limit: search range
    shape_info: generated shapes info of each node (holes remain)

    Returns:
    A grouped fx_module
    '''
    sche = [node for node in fx_module.graph.nodes]
    idx = len(sche) - 1
    cur_cap = 0
    cur_range = 0
    cur_comm_list = []
    comm_list_dep = []
    retrive_node = None
    while idx >= 0:
        cur_range += 1

        if (not sche[idx].ed_info.is_communication() and sche[idx] in comm_list_dep) \
            or cur_range > rg_limit \
            or cur_cap > cap_limit:

            cur_comm_list.reverse()
            comm_nodes_group(fx_module, cur_comm_list, shape_info)
            sche = [node for node in fx_module.graph.nodes]

            cur_cap = 0
            cur_range = 0
            cur_comm_list = []
            comm_list_dep = []
            if retrive_node:
                idx = sche.index(retrive_node)
                retrive_node = None
                continue

        if not sche[idx].ed_info.is_communication():
            idx -= 1
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
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()
    return fx_module


def comm_optimize(fx_module: torch.fx.GraphModule,
                  shape_info,
                  sche_method,
                  grouping=False,
                  mem_restrain=False):
    '''
    This function performs multiple communciation optimizations on graph level

    Args:
    fx_module: fx graph to be optimized
    shape_info: generated shapes info of each node (holes remain)
    grouping: whether or not grouping is to be performed
    mem_restrain: whether or not mem_restrain is added to rcpsp

    Returns:
    A transformed fx_module with communication optimizations applied
    '''
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if mdconfig.log_level <= logging.DEBUG:
        fx_module.print_readable()

    # collect necessary communication node info, save at comm_meta in node.ed_info
    for node in fx_module.graph.nodes:
        if node.ed_info.is_communication():
            assert len(node.all_input_nodes) == 1
            from_node = node.all_input_nodes[0]
            comm_shape = shape_info[from_node.name]['shape']
            # TODO support mixed precision
            node.ed_info.comm_meta = {
                'comm_vol': reduce(lambda x, y: x * y, comm_shape, 1) * 4,  #Bytes
                'comm_shape': comm_shape
            }
        elif node.ed_info.is_computation():
            for pre in node.all_input_nodes:
                if pre.ed_info.is_communication():
                    pre.ed_info.comm_meta['to_node'] = node

    _shapeinfo_fill_up(shape_info, fx_module)

    if grouping:
        fx_module = comm_group(fx_module, 1024 * 1024, 10000, shape_info)

    # comm_map: node just computed -> commnications followed
    comm_map = {}
    if sche_method == 'eager':
        for node in fx_module.graph.nodes:
            if node.ed_info.is_communication():
                if comm_map.get(from_node) is None:
                    comm_map[from_node] = []
                comm_map[from_node].append(node)
    elif sche_method == 'rcpsp':
        sche = rcpsp_schedule(fx_module, shape_info, mem_restrain)

        _link_nodes(fx_module, sche)

        for idx, node in enumerate(sche):
            if not node.ed_info.is_communication() and \
                idx + 1 < len(sche) and \
                sche[idx + 1].ed_info.is_communication():
                comm_map[node] = []
                for follower in sche[idx + 1:]:
                    if follower.ed_info.is_communication():
                        comm_map[node].append(follower)
                    else:
                        break
                assert (len(comm_map[node]) > 0)

    def grouped_comm(input_tensors: list, comm_func: list, comm_args: list):
        res = []
        for input_tensor, comm_func, args in zip(input_tensors, comm_func, comm_args):
            res.append(eval(comm_func)(input_tensor, *args))
        return res

    # add after nodes followed by comms a grouped comm node
    for node in comm_map:
        if len(comm_map[node]) <= 1:
            continue

        input_nodes = [n.all_input_nodes[0] for n in comm_map[node]]
        comm_funcs = [_get_qualified_name(n.target) for n in comm_map[node]]
        comm_args = [n.args[1:] for n in comm_map[node]]

        # add grouped comm node
        with fx_module.graph.inserting_after(node):
            new_comm_node = fx_module.graph.call_function(grouped_comm,
                                                          args=(input_nodes, comm_funcs,
                                                                comm_args))

        # add retrive node
        for idx, comm_node in enumerate(comm_map[node]):
            with fx_module.graph.inserting_after(new_comm_node):
                idx_node = fx_module.graph.call_function(operator.getitem,
                                                         args=(new_comm_node, idx))
            comm_node.ed_info.comm_meta['to_node'].replace_input_with(comm_node, idx_node)

    # at this point all old comm operators should be eliminated
    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if torch.distributed.get_rank() == 0:
        logger.info("Communication Optimization: Done!")
    return fx_module


def _strategy_fill_up(opt_strategy, shape_info, fx_module):
    '''
    Rule-based filling up strategies of nodes in fx_module
    '''
    for node in fx_module.graph.nodes:
        if not node.ed_info.is_communication():
            if opt_strategy.get(node.name) is None:
                if node.name.__contains__("getitem"):
                    idx = node.args[1]
                    pre_node = node.all_input_nodes[0]
                    opt_strategy[node.name] = {
                        'node':
                        node.name,
                        'strategy':
                        NodeSPMDStrategy(
                            VarSPMDStrategyGroup(
                                opt_strategy[pre_node.name]['strategy'].get_invar_strtg(idx)),
                            VarSPMDStrategyGroup(
                                opt_strategy[pre_node.name]['strategy'].get_outvar_strtg(idx)))
                    }
                elif not (node.name.__contains__("arg") or node.name.__contains__("output")):
                    # Assumption: nodes without strategy and not being args or output are constant tensor
                    opt_strategy[node.name] = {
                        'node':
                        node,
                        'strategy':
                        NodeSPMDStrategy(
                            VarSPMDStrategyGroup(
                                VarSPMDStrategy(*tuple([SPMD('REPLICATE')] *
                                                       len(shape_info[node.name]['shape'])))),
                            VarSPMDStrategyGroup(
                                VarSPMDStrategy(*tuple([SPMD('REPLICATE')] *
                                                       len(shape_info[node.name]['shape'])))))
                    }

    if mdconfig.log_level <= logging.DEBUG:
        print(f"opt_strategy: {opt_strategy}")


def _shapeinfo_fill_up(shape_info, fx_module):
    '''
    Rule-based filling up shape_info of nodes in fx_module
    '''
    for node in fx_module.graph.nodes:
        if not node.ed_info.is_communication():
            if shape_info.get(node.name) is None:
                if node.name.__contains__("scatter_wrapper"):
                    pre_node = node.all_input_nodes[0]
                    shape_info[node.name] = shape_info[pre_node.name]
                elif node.name.__contains__("_end"):
                    pre_node = node.all_input_nodes[0]
                    shape_info[node.name] = shape_info[pre_node.all_input_nodes[0].name]
                elif torch.distributed.get_rank() == 0:
                    raise RuntimeError("_shapeinfo_fill_up: unmet node->{node.name}!")

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
