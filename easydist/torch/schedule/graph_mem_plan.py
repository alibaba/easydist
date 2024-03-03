# Copyright (c) 2024, Alibaba Group;
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


class GraphMemPlan:
    def __init__(
        self,
        mem_size: int,
        temp_mem_size: int
    ):
        self.mem_size = mem_size
        self.temp_mem_size = temp_mem_size
        self.raw_mem_allocs: List[(int, int, bool, str)] = [] # element: (addr, size, is_temp_mem, node_name)

    def append_addr_size(self, addr: int, size: int, is_temp_mem: bool, node_name: str):
        self.raw_mem_allocs.append((addr, size, is_temp_mem, node_name))

    def get_addr(self, idx: int):
        assert idx<len(self.raw_mem_allocs)
        return self.raw_mem_allocs[idx]

    def __str__(self) -> str:
        mem_plan_str = ""
        for raw_mem_alloc in self.raw_mem_allocs:
            if raw_mem_alloc[2]:
                is_temp = "True"
            else:
                is_temp = "False"
            mem_plan_str += f"addr: {raw_mem_alloc[0]}, size: {raw_mem_alloc[1]}, is_temp: {is_temp}, node: {raw_mem_alloc[3]}\n"

        return mem_plan_str


