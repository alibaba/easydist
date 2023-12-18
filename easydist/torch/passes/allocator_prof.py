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
import __main__
import os

import torch
import torch.utils._pytree as pytree
from torch.fx.node import _get_qualified_name
from torch.distributed._functional_collectives_impl import _wait_all

import easydist.config as mdconfig
from easydist.torch.utils import EDNodeType
from easydist.torch.init_helper import materialize_random
from easydist.torch.graph_profile_db import PerfDB



logger = logging.getLogger(__name__)

class ModuleProfilingInfo:
    def __init__(self) -> None:
        self.node_profiling_info = dict()

    @property
    def node_names(self):
        return self.node_profiling_info.keys()

    def local_indexes(self):
        return self.local_indexes

    def build_local_indexes(self):
        # calculating local indexes of each malloc
        self.local_indexes = dict()
        for node_name in self.node_names:
            node_info = self.get_node_profiling_info(node_name)
            indexes = []
            for i, alloc_addr in enumerate(node_info.allocator_address):
                if alloc_addr in node_info.output_address:
                    indexes.append(i)
            self.local_indexes[node_name] = indexes

    @property
    def is_inplace(self):
        # check whether this node is inplace or not
        result = dict()
        for node_name in self.node_names:
            node_info = self.get_node_profiling_info(node_name)
            flag = False
            for out_addr in node_info.output_address:
                if out_addr in node_info.input_address:
                    flag = True
                    break
            result[node_name] = flag
        return result

    def set_node_profiling_info(self, node_name, node_info):
        self.node_profiling_info[node_name] = node_info

    def __setitem__(self, node_name, node_info):
        self.set_node_profiling_info(node_name, node_info)

    def get_node_profiling_info(self, node_name):
        return self.node_profiling_info[node_name]

    def __getitem__(self, node_name):
        return self.get_node_profiling_info(node_name)

class NodeProfilingInfo:
    def __init__(self):
        self.qualified_name = ""
        self.input_address = []
        self.output_address = []
        self.allocator_address = []

    def set_qualified_name(self, qualified_name):
        self.qualified_name = qualified_name

    def add_input_address(self, input_address):
        self.input_address.append(input_address)

    def add_output_address(self, output_address):
        self.output_address.append(output_address)

    def add_allocator_address(self, allocator_address):
        self.allocator_address.append(allocator_address)

    def __str__(self) -> str:
        return_str = self.qualified_name + "\n"
        return_str += "input_address: " + ",".join([str(addr) for addr in self.input_address]) + "\n"
        return_str += "output_address: " + ",".join([str(addr) for addr in self.output_address]) + "\n"
        return_str += "allocator_address: " + ",".join([str(addr) for addr in self.allocator_address]) + "\n"

        return return_str

def allocator_prof(fx_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # only profile on rank 0
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank != 0:
        return fx_module

    logging.info("profiling fx_module's memory...")

    # save all profiling information in this dict
    profiling_info = ModuleProfilingInfo()

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
        node_profiling_info = NodeProfilingInfo()

        # record qualified names
        node_profiling_info.set_qualified_name(qualified_name)

        # materialize inputs
        inputs_signature = pytree.tree_map_only(torch.fx.Node, lambda n: n.meta['val'], node.args)
        materialized_inputs = pytree.tree_map_only(torch.Tensor, materialize_random, inputs_signature)

        # record input addresses
        for materialized_input in materialized_inputs:
            if isinstance(materialized_input, torch.Tensor):
                node_profiling_info.add_input_address(materialized_input.data_ptr())
        
        # set op_name for profiling_allocator.so
        __main__.op_name = node.name

        # tell profiling_allocator to start profiling
        __main__.ignore = False

        # execute the node
        node_output = node.target(*materialized_inputs, **node.kwargs)

        # record output addresses
        # flatten to handle possible tuples
        flat_outputs, _ = pytree.tree_flatten(node_output)

        for flat_output in flat_outputs:
            if isinstance(flat_output, torch.Tensor):
                node_profiling_info.add_output_address(flat_output.data_ptr())
            elif flat_output is None:
                continue
            else:
                assert False, "Unexpected output!"
        
        # tell profiling_allocator stop recording
        __main__.ignore = True

        # saving profiling info
        profiling_info.set_node_profiling_info(node.name, node_profiling_info)
    
    # process info from our profiling allocator
    # data format from allocator: (node_name, ptr_address)
    for allocator_info in __main__.allocator_profiling_info:
        node_name, ptr = allocator_info
        node_info = profiling_info.get_node_profiling_info(node_name)
        node_info.add_allocator_address(ptr)

    profiling_info.build_local_indexes()

    for node_name in profiling_info.node_names:
        if profiling_info.local_indexes[node_name]:
            print(node_name, profiling_info.get_node_profiling_info(node_name).qualified_name, profiling_info.local_indexes[node_name])
        elif profiling_info.is_inplace[node_name]:
            print(node_name, profiling_info.get_node_profiling_info(node_name).qualified_name, 'inplace')
        else:
            assert False, "unexpected situation! " + profiling_info.get_node_profiling_info(node_name)

    return fx_module
