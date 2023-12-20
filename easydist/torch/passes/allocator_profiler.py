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

from typing import Any

import __main__
import torch.utils._pytree as pytree
import torch
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter
from torch.fx.node import Node, _get_qualified_name
from torch.fx._compatibility import compatibility

from easydist.torch.utils import EDNodeType
from easydist.torch.init_helper import materialize_random
from easydist.torch.utils import to_meta
from .allocator_prof import ModuleProfilingInfo, NodeProfilingInfo

__all__ = ['AllocatorProfiler']


@compatibility(is_backward_compatible=True)
class AllocatorProfiler(Interpreter):
    @compatibility(is_backward_compatible=True)
    def __init__(self, module : GraphModule,
                 profiling_info : ModuleProfilingInfo,
                 garbage_collect_values : bool = True):
        super().__init__(module, garbage_collect_values)
        self.profiling_info = profiling_info

    @compatibility(is_backward_compatible=True)
    def run_node(self, n : Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        if n.ed_info.node_type is EDNodeType.COMMUNICATION:
            return None

        if n.op == "placeholder":
            return None

        if n.op == "output":
            qualified_name = "output"
        else:
            qualified_name = _get_qualified_name(n.target)
            if qualified_name == 'easydist.torch.passes.sharding.all_reduce_end':
                return None

        # create dict to store addresses for this node
        node_profiling_info = NodeProfilingInfo()
        node_profiling_info.set_qualified_name(qualified_name)

        with self._set_current_node(n):
            # tell profiling_allocator stop recording
            __main__.ignore = True

            inputs_signature = pytree.tree_map_only(torch.fx.Node, lambda nd: nd.meta['val'], n.args)
            materialized_inputs = pytree.tree_map_only(torch.Tensor, materialize_random, inputs_signature)

            # record input addresses
            for materialized_input in materialized_inputs:
                if isinstance(materialized_input, torch.Tensor):
                    node_profiling_info.add_input_address(materialized_input.data_ptr())

            # set op_name for profiling_allocator.so
            __main__.op_name = n.name

            # tell profiling_allocator to start profiling
            __main__.ignore = False

            output = getattr(self, n.op)(n.target, materialized_inputs, {})

            # flatten to handle possible tuples
            flat_outputs, _ = pytree.tree_flatten(output)

            # record output addresses
            for flat_output in flat_outputs:
                if isinstance(flat_output, torch.Tensor):
                    node_profiling_info.add_output_address(flat_output.data_ptr())
                elif flat_output is None:
                    continue
                else:
                    assert False, "Unexpected output!"

            # saving profiling info
            self.profiling_info.set_node_profiling_info(n.name, node_profiling_info)
            meta_out = pytree.tree_map(to_meta, output)
            return meta_out

    def finalize_allocator_info(self):
        # process info from our profiling allocator
        # data format from allocator: (node_name, ptr_address)
        for allocator_info in __main__.allocator_profiling_info:
            node_name, ptr = allocator_info
            node_info = self.profiling_info.get_node_profiling_info(node_name)
            node_info.add_allocator_address(ptr)

        self.profiling_info.build_local_indexes()
        print(self.profiling_info)

        for node_name in self.profiling_info.node_names:
            print("node name in prof info: ", node_name)
            print(("node info:\n{}").format(self.profiling_info.get_node_profiling_info(node_name)))
            if self.profiling_info.local_indexes[node_name]:
                print(node_name, self.profiling_info.get_node_profiling_info(node_name).qualified_name, self.profiling_info.local_indexes[node_name])
            elif self.profiling_info.is_inplace[node_name]:
                print(node_name, self.profiling_info.get_node_profiling_info(node_name).qualified_name, 'inplace')
            else:
                print("warning: unexpected situation!", node_name, ':', self.profiling_info.get_node_profiling_info(node_name).qualified_name)

