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

from collections import defaultdict

import sys
import logging
import torch
import intervaltree
import matplotlib.pyplot as plt

import easydist.config as mdconfig

from easydist.torch.schedule.lifetime_info import LifetimeInfo
from easydist.torch.schedule.memory_scheduler import MemoryScheduler


logger = logging.getLogger(__name__)

class EfficientMemoryScheduler(MemoryScheduler):
    """
    """
    def __init__(
        self,
        fx_module,      # torch.fx.GraphModule
        graph_mem_info, # GraphMemInfo
        align_scale=4
    ):
        super().__init__(fx_module, graph_mem_info, align_scale)

    def build_lifetime(self, pre_scheded_nodes=None) -> LifetimeInfo:
        manual_alloc = True
        if not pre_scheded_nodes:   # pre_scheded_nodes: map of node -> step
            manual_alloc = False

        for node in self.nodes_to_schedule:
            if node not in pre_scheded_nodes:
                manual_alloc = False
                break

        if not manual_alloc:
            logger.info(f"Some ops are not scheduled yet, ignore memory address generation")
            return (None, None, None, None)

        lifetime_info = LifetimeInfo(self.fx_module.graph,
                                     self.nodes_to_schedule,
                                     self.graph_mem_info,
                                     pre_scheded_nodes,
                                     True)
        lifetime_info.compute_buffer_makespans()

        #print("asap and alap information:")
        #print(lifetime_info.node_makespans)
        #print("edge lifetime information:")
        #print(lifetime_info.edge_makespans)

        #print("buffer lifetime information:")
        #print(lifetime_info.buffer_makespans)

        return lifetime_info

    def export_mem_plan(self, group_addresses, mem_locations, mem_alloc_info):
        max_end_addr = 0

        for group, addr_size in group_addresses.items():
            if max_end_addr < addr_size[0]+addr_size[1]:
                max_end_addr = addr_size[0]+addr_size[1]

            group_addr = addr_size[0]*self.gcd
            group_end = group_addr + addr_size[1]*self.gcd
            for tensor in group.tensors:
                node = tensor[0]
                out_idx = tensor[1]
                out_var = self.graph_mem_info.get_out_var(node, out_idx)
                addr = group_addr + out_var.offset
                mem_locations[node.name][out_idx] = (addr, out_var.mem_size)
                if not out_var.is_reference:
                    if node.name not in mem_alloc_info:
                        mem_alloc_info[node.name] = [(out_var.alloc_index, addr, out_var.mem_size, 1)]
                    else:
                        mem_alloc_info[node.name].append((out_var.alloc_index, addr, out_var.mem_size, 1))

                assert addr + out_var.mem_size <= group_end

        required_memory = max_end_addr*self.gcd

        for nd_name, alloc_list in mem_alloc_info.items():
            # sort allocation by allocation index
            alloc_list.sort(key=lambda x:x[0])
            for idx, alloc_info in enumerate(alloc_list):
                assert idx == alloc_info[0], f"Invalid allocation index in node {nd_name}, alloc info: {alloc_list}"

        return required_memory

    def plan_mem_addr_by_min_skyline(self, lifetime_info):
        # 1. generate memory address for tensor between ops
        buf_makespans = lifetime_info.buffer_makespans
        tensor_groups = set()
        for ten_group in buf_makespans.keys():
            if not ten_group.src_from_ctx:
                tensor_groups.add(ten_group)

        num_steps = lifetime_info.num_steps
        group_addresses = {}
        mem_used = intervaltree.IntervalTree()  # skyline intervals
        mem_used[0:num_steps] = (-1, 0)  # a specific interval to cover whole range
        while len(tensor_groups)>0:
            # find min(max(skyline(overlap_interval)))
            min_max_addr = sys.maxsize
            min_max_ten_groups = []
            for ten_group in tensor_groups:
                span = buf_makespans[ten_group]
                max_addr = 0
                for interval in mem_used.overlap(span[0], span[1] + 1):
                    if interval.data[1] > max_addr:
                        max_addr = interval.data[1]

                if max_addr < min_max_addr:
                    min_max_addr = max_addr
                    min_max_ten_groups = [ten_group]
                elif max_addr == min_max_addr:
                    min_max_ten_groups.append(ten_group)

            assert min_max_addr < sys.maxsize and len(min_max_ten_groups) > 0
            max_duration = 0
            fixed_ten_group = None
            for min_ten_group in min_max_ten_groups:
                lb, ub = buf_makespans[min_ten_group]
                duration = ub - lb + 1
                if max_duration < duration:
                    max_duration = duration
                    fixed_ten_group = min_ten_group

            assert fixed_ten_group
            lb, ub = buf_makespans[fixed_ten_group]
            scaled_size = (fixed_ten_group.src_tensor_size + self.gcd - 1) // self.gcd
            #print(f"scaled_size: {scaled_size}, source tensor size: {fixed_ten_group.src_tensor_size}")

            # update intervals
            mem_used.remove_envelop(lb, ub+1)
            overlap_intervals = mem_used.overlap(lb, ub+1)

            assert len(overlap_intervals) <= 2
            for overlap_interval in overlap_intervals:
                if overlap_interval.begin < lb:
                    mem_used[overlap_interval.begin:lb] = overlap_interval.data
                if overlap_interval.end > (ub+1):
                    mem_used[ub+1:overlap_interval.end] = overlap_interval.data

            mem_used.remove_overlap(lb, ub+1)

            mem_used[lb:ub+1] = (min_max_addr, min_max_addr + scaled_size)

            group_addresses[fixed_ten_group] = (min_max_addr, scaled_size)
            tensor_groups.remove(fixed_ten_group)

        # 2. generate memory address for temp allocation inside an op
        temp_mem_locations = {}
        temp_mem_alloc_info = {}
        max_temp_size = 0
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            temp_mem_locations[node.name] = [None]*len(out_vars)

            temp_vars = self.graph_mem_info.get_temp_vars(node)
            node_temp_addr = 0
            for temp_var in temp_vars:
                if node.name not in temp_mem_alloc_info:
                    temp_mem_alloc_info[node.name] = [(temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0)]
                else:
                    temp_mem_alloc_info[node.name].append((temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0))
                node_temp_addr += temp_var.mem_size
            if node_temp_addr > max_temp_size:
                max_temp_size = node_temp_addr

        return (group_addresses, temp_mem_locations, temp_mem_alloc_info, max_temp_size)

    def plan_mem_addr_by_ignore_reuse(self, lifetime_info):  # as a golden for memory reusing version
        # 1. generate memory address for tensor between ops
        buf_makespans = lifetime_info.buffer_makespans
        tensor_groups = set()
        for ten_group in buf_makespans.keys():
            if not ten_group.src_from_ctx:
                tensor_groups.add(ten_group)

        group_addresses = {}
        max_addr = 0;
        for ten_group in tensor_groups:
            scaled_size = (ten_group.src_tensor_size + self.gcd - 1) // self.gcd
            group_addresses[ten_group] = (max_addr, scaled_size)
            max_addr += scaled_size

        # 2. generate memory address for temp allocation inside an op
        temp_mem_locations = {}
        temp_mem_alloc_info = {}
        node_temp_addr = 0
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            temp_mem_locations[node.name] = [None]*len(out_vars)

            temp_vars = self.graph_mem_info.get_temp_vars(node)
            for temp_var in temp_vars:
                if node.name not in temp_mem_alloc_info:
                    temp_mem_alloc_info[node.name] = [(temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0)]
                else:
                    temp_mem_alloc_info[node.name].append((temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0))
                node_temp_addr += temp_var.mem_size
        max_temp_size = node_temp_addr

        return (group_addresses, temp_mem_locations, temp_mem_alloc_info, max_temp_size)

    def create_min_mem_plan(self, pre_scheded_nodes=None):
        lifetime_info = self.build_lifetime(pre_scheded_nodes)

        if mdconfig.ignore_memory_reuse:
            group_addresses, mem_locations, mem_alloc_info, max_temp_size = \
                              self.plan_mem_addr_by_ignore_reuse(lifetime_info)
        else:
            group_addresses, mem_locations, mem_alloc_info, max_temp_size = \
                              self.plan_mem_addr_by_min_skyline(lifetime_info)

        required_memory = self.export_mem_plan(group_addresses,
                                               mem_locations,
                                               mem_alloc_info)

        # dump peak memory
        logger.info(f"required memory: {required_memory/1024/1024/1024}GB")

        if mdconfig.dump_mem_usage_graph:
            # dump memory addresses
            self.dump_mem_usage_graph(mem_locations,
                                      lifetime_info.buffer_makespans,
                                      lifetime_info.num_steps,
                                      required_memory,
                                      lifetime_info)

        ordered_schedules = []
        for node in self.nodes_to_schedule:
            ordered_schedules.append(node.name)

        return (required_memory, max_temp_size, None, ordered_schedules, mem_alloc_info, mem_locations)

    def dump_mem_usage_graph(self,
                             mem_locations,
                             buf_makespans,
                             num_steps,
                             required_memory,
                             lifetime_info):
        graph_mem_addr_str = "graph memory addresses:\n"
        for node_name, mem_addrs in mem_locations.items():
            node_mem_str = node_name + ": "
            for addr_size in mem_addrs:
                if addr_size:
                    node_mem_str += "([" + str(addr_size[0]) + "~" + \
                                    str(addr_size[0]+addr_size[1]-1) + \
                                    "], " + str(addr_size[1]) + "),"
                else:
                    node_mem_str += "([NA ~ NA], NA),"

            graph_mem_addr_str += node_mem_str + "\n"

        logger.info(graph_mem_addr_str)

        # dump memory usage in graph
        node_map = {}
        for node in self.nodes_to_schedule:
            node_map[node.name] = node

        _, ax = plt.subplots()
        ax.axis([0,num_steps,0,required_memory])
        dumped_group = set()
        for node_name, mem_addrs in mem_locations.items():
            node = node_map[node_name]
            for out_idx,mem_addr_size in enumerate(mem_addrs):
                group = lifetime_info.get_group(node, out_idx)
                if group.src_from_ctx:
                    continue
                if group in dumped_group:
                    continue
                dumped_group.add(group)
                lb, ub = buf_makespans[group]
                left = lb
                width = ub - lb + 1
                bottom = mem_addr_size[0]
                height = mem_addr_size[1]
                ax.add_patch(plt.Rectangle((left, bottom), width, height, fill=None, alpha=1))

        plt.savefig("./tmp/mem_fig.pdf")

