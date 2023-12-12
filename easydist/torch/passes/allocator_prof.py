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
import ctypes

import torch
import torch.utils._pytree as pytree
from torch.fx.node import _get_qualified_name
from torch.distributed._functional_collectives_impl import _wait_all

import easydist.config as mdconfig
from easydist.torch.utils import EDNodeType
from easydist.torch.init_helper import materialize_random
from easydist.torch.graph_profile_db import PerfDB



logger = logging.getLogger(__name__)

def allocator_prof(fx_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    logging.info("profiling fx_module's memory...")
    import __main__, os
    
    # only profile on rank 0
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank != 0:
        return fx_module

    # save all profiling information in this dict
    profiling_info = dict()

    for node in fx_module.graph.nodes:
        if not hasattr(node, "ed_info"):
            continue

        if not node.ed_info.node_type is EDNodeType.COMPUTATION:
            continue
        
        # filter allreduce add by easydist
        qualified_name = _get_qualified_name(node.target)
        if qualified_name == 'easydist.torch.passes.sharding.all_reduce_end':
            continue

        # create dict to store addresses for this node
        node_profiling_info = dict()

        # record qualified names
        node_profiling_info['qualified_name'] = qualified_name

        # materialize inputs
        inputs_signature = pytree.tree_map_only(torch.fx.Node, lambda n: n.meta['val'], node.args)
        materialized_inputs = pytree.tree_map_only(torch.Tensor, materialize_random, inputs_signature)

        # record input addresses
        node_profiling_info['input_address'] = []
        for materialized_input in materialized_inputs:
            if isinstance(materialized_input, torch.Tensor):
                node_profiling_info['input_address'].append(materialized_input.data_ptr())
        
        # set op_name for profiling_allocator.so
        __main__.op_name = node.name

        # tell profiling_allocator to start profiling
        __main__.ignore = False

        # execute the node
        node_output = node.target(*materialized_inputs, **node.kwargs)

        # record output addresses
        node_profiling_info['output_address'] = []

        # if output is a single tensor
        if isinstance(node_output, torch.Tensor):
            out_tensor = node_output
            node_profiling_info['output_address'].append(out_tensor.data_ptr())

        # if output is a tuple of tensors
        elif isinstance(node_output, tuple):
            for out_item in node_output:
                if isinstance(out_item, torch.Tensor):
                    node_profiling_info['output_address'].append(out_item.data_ptr())
                elif out_item is None:
                    continue
                else:
                    assert False, "Unexpected out_item!"
        else:
            assert False, "Unexpected type of out_tensors!"
        
        # record allocator addresses
        # for now, do nothing. we will write addresses later.
        node_profiling_info['allocator_address'] = []

        # tell profiling_allocator stop recording
        __main__.ignore = True

        # saving profiling info
        profiling_info[node.name] = node_profiling_info
    
    # process info from our profiling allocator
    # data format from allocator: (node_name, ptr_address)
    for allocator_info in __main__.allocator_profiling_info:
        node_name, ptr = allocator_info
        profiling_info[node_name]["allocator_address"].append(ptr)

    # calculating local indice of each malloc
    local_indice = dict()
    for node_name in profiling_info:
        node_info = profiling_info[node_name]
        local_indice[node_name] = []
        for i, alloc_addr in enumerate(node_info['allocator_address']):
            if alloc_addr in node_info['output_address']:
                local_indice[node_name].append(i)
    
    # check whether this node is inplace or not
    is_inplace = dict()
    for node_name in profiling_info:
        node_info = profiling_info[node_name]
        flag = False
        for out_addr in node_info['output_address']:
            if out_addr in node_info['input_address']:
                flag = True
                break
        is_inplace[node_name] = flag

    for node_name in profiling_info:
        if local_indice[node_name]:
            pass
            # print(node_name, profiling_info[node_name]['qualified_name'], local_indice[node_name])
        elif is_inplace[node_name]:
            print(node_name, profiling_info[node_name]['qualified_name'], 'inplace')
        else:
            assert False, "unexpected situation! " + profiling_info[node_name]

    return fx_module