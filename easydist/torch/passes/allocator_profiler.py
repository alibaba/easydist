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

__all__ = ['AllocatorProfiler']


class NodeProfilingInfo:
    def __init__(self) -> None:
        self.qualified_name: str = ""
        self.input_address: List[int] = []
        self.input_size: List[int] = []
        self.output_address: List[int] = []
        self.output_size: List[int] = []
        self.allocator_address: List[int] = []
        self.allocator_size: List[int] = []

    def set_qualified_name(self, qualified_name: str) -> None:
        self.qualified_name = qualified_name

    def add_input_address(self, input_address: int) -> None:
        self.input_address.append(input_address)

    def add_input_size(self, input_size: int) -> None:
        self.input_size.append(input_size)

    def add_input_info(self, input_tensor: torch.Tensor) -> None:
        self.add_input_address(input_tensor.data_ptr())
        self.add_input_size(input_tensor.element_size() * input_tensor.numel())

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
        return_str = self.qualified_name + "\n"
        return_str += "input_address: " + ",".join([str(addr) for addr in self.input_address]) + "\n"
        return_str += "input_size: " + ','.join([str(size) for size in self.input_size]) + "\n"
        return_str += "output_address: " + ",".join([str(addr) for addr in self.output_address]) + "\n"
        return_str += "output_size: " + ','.join([str(size) for size in self.output_size]) + "\n"
        return_str += "allocator_address: " + ",".join([str(addr) for addr in self.allocator_address]) + "\n"
        return_str += "allocator_size: " + ','.join([str(size) for size in self.allocator_size]) + "\n"

        return return_str

    def __repr__(self) -> str:
        return f'NodeProfilingInfo(qualified_name={repr(self.qualified_name)}, ' + \
            f'input_address={repr(self.input_address)}, ' + \
            f'output_address={repr(self.output_address)}, ' + \
            f'allocator_address={repr(self.allocator_address)})'

class ModuleProfilingInfo:
    def __init__(self) -> None:
        self.node_profiling_info: Dict[str, NodeProfilingInfo] = dict()
        self._is_inplace: Dict[str, bool] = None
        self._local_indexes: Dict[str, int] = None
        self._inplace_mapping: Dict[str, Tuple[int, int]] = None

    @property
    def node_names(self):
        return self.node_profiling_info.keys()

    @property
    def local_indexes(self):
        if not self._local_indexes:
            self.build_local_indexes()
        return self._local_indexes

    def build_local_indexes(self):
        # calculating local indexes of each malloc
        self._local_indexes = dict()
        for node_name in self.node_names:
            node_info = self.get_node_profiling_info(node_name)
            indexes = []
            for i, alloc_addr in enumerate(node_info.allocator_address):
                if alloc_addr in node_info.output_address:
                    indexes.append(i)
            self._local_indexes[node_name] = indexes

    @property
    def is_inplace(self):
        if not self._is_inplace:
            self._is_inplace = dict()
            for node_name in self.node_names:
                node_info = self.get_node_profiling_info(node_name)
                flag = False
                for out_addr in node_info.output_address:
                    if out_addr in node_info.input_address:
                        flag = True
                        break
                self._is_inplace[node_name] = flag

        return self._is_inplace

    @property
    def inplace_mapping(self):
        if not self._inplace_mapping:
            self._inplace_mapping = dict()
            for node_name in self.node_names:
                self._inplace_mapping[node_name] = []
                node_info = self.get_node_profiling_info(node_name)
                for out_index, out_addr in enumerate(node_info.output_address):
                    for in_index, in_addr in enumerate(node_info.input_address):
                        if out_addr == in_addr:
                            self._inplace_mapping[node_name].append((in_index, out_index))

        return self._inplace_mapping


    def set_node_profiling_info(self, node_name: str, node_info: NodeProfilingInfo):
        self.node_profiling_info[node_name] = node_info

    def __setitem__(self, node_name: str, node_info: NodeProfilingInfo):
        self.set_node_profiling_info(node_name, node_info)

    def get_node_profiling_info(self, node_name: str):
        return self.node_profiling_info[node_name]

    def __getitem__(self, node_name: str):
        return self.get_node_profiling_info(node_name)

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
            __main__.start_recording = False

            inputs_signature = pytree.tree_map_only(torch.fx.Node, lambda nd: nd.meta['val'], n.args)
            materialized_inputs = pytree.tree_map_only(torch.Tensor, materialize_random, inputs_signature)

            # record input addresses
            flat_inputs, _ = pytree.tree_flatten(materialized_inputs)
            constant_types = [bool, int, float, torch.dtype]
            for flat_input in flat_inputs:
                if isinstance(flat_input, torch.Tensor):
                    node_profiling_info.add_input_info(flat_input)
                elif flat_input is None:
                    continue
                elif type(flat_input) in constant_types:
                    continue
                else:
                    print("Unexpected input!", type(flat_input), flat_input)

            # set op_name for profiling_allocator.so
            __main__.op_name = n.name

            # tell profiling_allocator to start profiling
            __main__.start_recording = True

            output = getattr(self, n.op)(n.target, materialized_inputs, {})

            # flatten to handle possible tuples
            flat_outputs, _ = pytree.tree_flatten(output)

            # record output addresses
            for flat_output in flat_outputs:
                if isinstance(flat_output, torch.Tensor):
                    node_profiling_info.add_output_info(flat_output)
                elif flat_output is None:
                    continue
                else:
                    print("Unexpected output!", type(flat_input), flat_input)

            # saving profiling info
            self.profiling_info.set_node_profiling_info(n.name, node_profiling_info)
            meta_out = pytree.tree_map(to_meta, output)
            return meta_out

    def finalize_allocator_info(self):
        # process info from our profiling allocator
        # data format from allocator: (node_name, ptr_address)
        for allocator_info in __main__.allocator_profiling_info:
            node_name, *info = allocator_info
            node_info = self.profiling_info.get_node_profiling_info(node_name)
            node_info.add_allocator_info(info)

        for node_name in self.profiling_info.node_names:
            if 'getitem' not in node_name:
                continue
            print("node name in prof info: ", node_name)
            print(self.profiling_info.inplace_mapping[node_name])
            print(self.profiling_info[node_name])