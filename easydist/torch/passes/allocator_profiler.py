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

from typing import Any, List, Dict, Tuple
from collections import defaultdict

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
from easydist.torch.mem_allocation_info import GraphMemInfo

__all__ = ['AllocatorProfiler']


class NodeProfilingInfo:
    def __init__(self, name) -> None:
        self.name = name
        self.qualified_name: str = ""

        # input info
        self.input_address: List[int] = []
        self.input_size: List[int] = []
        self.input_arg_index: List[int] = []
        self.input_tensor_index: List[int] = []

        # output info
        self.output_address: List[int] = []
        self.output_size: List[int] = []

        # allocator info
        self.allocator_address: List[int] = []
        self.allocator_size: List[int] = []

    def set_qualified_name(self, qualified_name: str) -> None:
        self.qualified_name = qualified_name

    def add_input_address(self, input_address: int) -> None:
        self.input_address.append(input_address)

    def add_input_size(self, input_size: int) -> None:
        self.input_size.append(input_size)

    def add_input_arg_index(self, arg_index: int) -> None:
        self.input_arg_index.append(arg_index)

    def add_input_tensor_index(self, tensor_index: int) -> None:
        self.input_tensor_index.append(tensor_index)

    def add_input_info(self, input_tensor: torch.Tensor, arg_index: int, tensor_index: int) -> None:
        self.add_input_address(input_tensor.data_ptr())
        self.add_input_size(input_tensor.element_size() * input_tensor.numel())
        self.add_input_arg_index(arg_index)
        self.add_input_tensor_index(tensor_index)

    def add_output_address(self, output_address: int) -> None:
        self.output_address.append(output_address)

    def add_output_size(self, output_size: int) -> None:
        self.output_size.append(output_size)

    def add_output_info(self, output_tensor: torch.Tensor) -> None:
        self.add_output_address(output_tensor.data_ptr())
        self.add_output_size(output_tensor.element_size() * output_tensor.numel())

    def add_allocator_address(self, allocator_address: int) -> None:
        self.allocator_address.append(allocator_address)

    def add_allocator_size(self, allocator_size: int) -> None:
        self.allocator_size.append(allocator_size)

    def add_allocator_info(self, allocator_info: List[Any]) -> None:
        address, size, *_ = allocator_info
        self.add_allocator_address(address)
        self.add_allocator_size(size)

    def __str__(self) -> str:
        return_str = self.name + "\n"
        return_str += "qualified_name: " + self.qualified_name + "\n"
        return_str += "input_address: " + ",".join([str(addr) for addr in self.input_address]) + "\n"
        return_str += "input_size: " + ','.join([str(size) for size in self.input_size]) + "\n"
        return_str += "output_address: " + ",".join([str(addr) for addr in self.output_address]) + "\n"
        return_str += "output_size: " + ','.join([str(size) for size in self.output_size]) + "\n"
        return_str += "allocator_address: " + ",".join([str(addr) for addr in self.allocator_address]) + "\n"
        return_str += "allocator_size: " + ','.join([str(size) for size in self.allocator_size]) + "\n"

        return return_str

    def __repr__(self) -> str:
        return f'NodeProfilingInfo(name={repr(self.name)}, ' + \
            f'qualified_name={repr(self.qualified_name)}, ' \
            f'input_address={repr(self.input_address)}, ' + \
            f'output_address={repr(self.output_address)}, ' + \
            f'allocator_address={repr(self.allocator_address)})'

class ModuleProfilingInfo:
    def __init__(self) -> None:
        self.node_profiling_info: Dict[str, NodeProfilingInfo] = dict()
        self._local_indexes: Dict[str, int] = None
        self._inplace_mapping: Dict[str, Tuple[int, int]] = None

    @property
    def node_names(self):
        return self.node_profiling_info.keys()

    def create_graph_mem_info(self) -> GraphMemInfo:
        graph_mem_info = GraphMemInfo()

        for node_name in self.node_names:
            node_mem_info = graph_mem_info.get_node_mem_info(node_name)
            node_profiling_info = self.get_node_profiling_info(node_name)

            for out_idx, out_addr in enumerate(node_profiling_info.output_address):
                is_allocated = False
                for alloc_idx, alloc_addr in enumerate(node_profiling_info.allocator_address):
                    if out_addr == alloc_addr:
                        alloc_size = node_profiling_info.allocator_size[alloc_idx]
                        node_mem_info.add_out_tensor_mem_info(
                                          out_idx, alloc_size, alloc_idx, False)
                        is_allocated = True
                        break;

                if not is_allocated:
                    for in_idx, in_addr in enumerate(node_profiling_info.input_address):
                        if out_addr == in_addr:
                            input_size = node_profiling_info.input_size[in_idx]
                            arg_index = node_profiling_info.input_arg_index[in_idx]
                            tensor_index = node_profiling_info.input_tensor_index[in_idx]
                            node_mem_info.add_out_tensor_mem_info(
                                              out_idx, input_size, in_idx, \
                                                True, arg_index, tensor_index)
                            break;

        return graph_mem_info

    def set_node_profiling_info(self, node_name: str, node_info: NodeProfilingInfo):
        self.node_profiling_info[node_name] = node_info

    def __setitem__(self, node_name: str, node_info: NodeProfilingInfo):
        self.set_node_profiling_info(node_name, node_info)

    def get_node_profiling_info(self, node_name: str):
        return self.node_profiling_info[node_name]

    def __getitem__(self, node_name: str):
        return self.get_node_profiling_info(node_name)

class NodeMemoryPlan:
    def __init__(self, output_indexes: List[int], assigned_addresses: List[int], total_mallocs: int):
        self.output_indexes = output_indexes
        self.assigned_addresses = assigned_addresses
        self.total_mallocs = total_mallocs

        # output_indexes should associate with assigned_addresses
        assert len(output_indexes) == len(assigned_addresses), \
            f'len of output_indexes({len(output_indexes)}) should equal to len of assigned_addresses({len(assigned_addresses)})!'

        # output_indexes must be in range [0, total_mallocs)
        assert min(output_indexes) >= 0, \
            f'min of output_indexes {min(output_indexes)} should be greater than zero!'
        assert total_mallocs > max(output_indexes), \
            f'total_mallocs {total_mallocs} should be greater than max of output_indexes {max(output_indexes)}!'

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

        if n.op == "placeholder":
            return None

        if n.op == "output":
            qualified_name = "output"
        else:
            qualified_name = _get_qualified_name(n.target)

        # create dict to store addresses for this node
        node_profiling_info = NodeProfilingInfo(n.name)
        node_profiling_info.set_qualified_name(qualified_name)

        with self._set_current_node(n):
            # tell profiling_allocator stop recording
            __main__.start_recording = False

            inputs_signature = pytree.tree_map_only(torch.fx.Node, lambda nd: nd.meta['val'], n.args)
            materialized_inputs = pytree.tree_map_only(torch.Tensor, materialize_random, inputs_signature)

            # record input addresses
            # TODO(wuhao): use customized tree_flatten, avoid flatten of non-tensors
            constant_types = [type(None), bool, int, float, torch.dtype, str]
            for arg_index, materialized_input in enumerate(materialized_inputs):
                for tensor_index, flat_input in enumerate(pytree.tree_flatten(materialized_input)[0]):
                    if isinstance(flat_input, torch.Tensor):
                        node_profiling_info.add_input_info(flat_input, arg_index, tensor_index)
                    elif type(flat_input) in constant_types:
                        continue
                    else:
                        assert False, f'Unexpected input: {type(flat_input)}, {flat_input}'

            # set op_name for profiling_allocator.so
            __main__.op_name = n.name

            # tell profiling_allocator to start profiling
            __main__.start_recording = True

            output = getattr(self, n.op)(n.target, materialized_inputs, n.kwargs)

            # flatten to handle possible tuples
            flat_outputs, _ = pytree.tree_flatten(output)

            # record output addresses
            for flat_output in flat_outputs:
                if isinstance(flat_output, torch.Tensor):
                    node_profiling_info.add_output_info(flat_output)
                elif flat_output is None:
                    continue
                else:
                    assert False, f'Unexpected output: {type(flat_input)}, {flat_input}'

            # saving profiling info
            self.profiling_info.set_node_profiling_info(n.name, node_profiling_info)
            meta_out = pytree.tree_map(to_meta, output)
            return meta_out

    def create_graph_mem_info(self) -> GraphMemInfo:
        # process info from our profiling allocator
        # data format from allocator: (node_name, ptr_address)
        for allocator_info in __main__.allocator_profiling_info:
            node_name, *info = allocator_info
            node_info = self.profiling_info.get_node_profiling_info(node_name)
            node_info.add_allocator_info(info)

        graph_mem_info = self.profiling_info.create_graph_mem_info()

        return graph_mem_info

    def load_memory_plan(self):
        memory_plan: Dict[str, NodeMemoryPlan] = dict()
        memory_plan['foo'] = NodeMemoryPlan([0, 2], [140231118422016, 140233702113280], 3)
        memory_plan['bar'] = NodeMemoryPlan([1, 3], [140245869126656, 140245869175808], 4)
        __main__.memory_plan = memory_plan
        graph_execution_order = ['bar', 'foo']
        __main__.graph_execution_order = graph_execution_order