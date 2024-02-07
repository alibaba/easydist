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

from collections import defaultdict

class OutVar:
    def __init__(
        self,
        out_index,
        mem_size,
        is_reference,
        alloc_index,
        arg_index,
        tensor_index
    ):
        self.out_index = out_index
        self.mem_size = mem_size

        # bool
        # True: it is a reference of an input tensor
        # False: it is new allocated memory
        self.is_reference = is_reference

        # allocation info:
        self.alloc_index = alloc_index

        # reference info:
        self.arg_index = arg_index
        self.tensor_index = tensor_index

    def size(self) -> int:
        return self.mem_size

    def __str__(self) -> str:
        mem_info_str = ""
        if self.is_reference:
            mem_info_str = "idx: " + str(self.out_index) + ", size: " + \
                           str(self.mem_size) + ", arg idx: " + \
                           str(self.arg_index) + ", tensor idx: " + \
                           str(self.tensor_index)
        else:
            mem_info_str = "idx: " + str(self.out_index) + ", size: " + \
                           str(self.mem_size) + ", alloc idx: " + \
                           str(self.alloc_index)
        return mem_info_str

class NodeMemInfo:
    def __init__(self):
        self.out_vars = []  # list of OutVar

    def add_out_var(self, out_index, mem_size, is_reference, alloc_index=-1,
                    arg_index=-1, tensor_index=-1):
        out_var = OutVar(out_index, mem_size, is_reference, alloc_index,
                         arg_index, tensor_index)
        self.out_vars.append(out_var)

    def get_out_var(self, out_idx):
        var = self.out_vars[out_idx]
        assert var.out_index == out_idx
        return var

    def __str__(self) -> str:
        mem_info_str = ""
        for tensor_mem_info in self.out_vars:
            mem_info_str += str(tensor_mem_info) + "\n"
        return mem_info_str

class GraphMemInfo:
    def __init__(self):
        self.node_mem_infos = defaultdict(lambda: NodeMemInfo())

    def get_node_mem_info(self, node_name) -> NodeMemInfo:
        return self.node_mem_infos[node_name]

    def get_out_vars(self, node):
        node_mem_info = self.get_node_mem_info(node.name)
        return node_mem_info.out_vars

    def get_out_var(self, node, out_idx):
        node_mem_info = self.get_node_mem_info(node.name)
        var = node_mem_info[out_idx]
        assert var.out_index == out_idx
        return var

    def __str__(self) -> str:
        graph_mem_info_str = ""
        for node_name, mem_info in self.node_mem_infos.items():
            graph_mem_info_str += node_name + ":\n" + str(mem_info) + "\n"

        return graph_mem_info_str

