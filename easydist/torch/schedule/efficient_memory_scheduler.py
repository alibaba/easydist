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

from easydist.torch.schedule.lifetime_info import ScheduledLifetimeInfo
from easydist.torch.schedule.memory_scheduler import MemoryScheduler


logger = logging.getLogger(__name__)

class EfficientMemoryScheduler(MemoryScheduler):
    """
    """
    def __init__(
        self,
        fx_module,      # torch.fx.GraphModule
        graph_mem_info, # GraphMemInfo
        op_streams,
        align_scale=4
    ):
        super().__init__(fx_module, graph_mem_info, op_streams, align_scale)

    def build_lifetime(self) -> ScheduledLifetimeInfo:
        manual_alloc = True
        if not self.schedule_result:
            manual_alloc = False
        else:
            for node in self.nodes_to_schedule:
                if node not in self.schedule_result.global_node_schedules:
                    manual_alloc = False
                    break

        if not manual_alloc:
            logger.info(f"Some ops are not scheduled yet, ignore memory address generation")
            return (None, None, None, None)

        lifetime_info = ScheduledLifetimeInfo(
                                     self.fx_module.graph,
                                     self.nodes_to_schedule,
                                     self.graph_mem_info,
                                     self.schedule_result)
        lifetime_info.build_anchors_between_streams()
        #print(f"stream anchors:\n{lifetime_info.anchors_between_streams}")
        lifetime_info.build_extended_schedule_maps()
        lifetime_info.compute_buffer_makespans()

        #print("asap and alap information:")
        #print(lifetime_info.node_makespans)
        #print("edge lifetime information:")
        #print(lifetime_info.edge_makespans)

        #print("buffer lifetime information:")
        #print(lifetime_info.buffer_makespans)

        return lifetime_info

    def export_mem_plan(self, group_addresses, inter_op_mems, mem_alloc_info):
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
                assert not inter_op_mems[node.name][out_idx]
                inter_op_mems[node.name][out_idx] = (addr, out_var)  # the information is stored in mem_alloc_info as well
                if not out_var.is_reference:
                    if node.name not in mem_alloc_info:
                        mem_alloc_info[node.name] = [(out_var.alloc_index, addr, out_var.mem_size, 1)]
                    else:
                        mem_alloc_info[node.name].append((out_var.alloc_index, addr, out_var.mem_size, 1))

                assert addr + out_var.mem_size <= group_end

        #print(f"scaled max_end_addr: {max_end_addr}")
        scaled_end_addr = max_end_addr

        return scaled_end_addr

    def plan_mem_addr_by_min_skyline(self, lifetime_info, stream_id,
                                     scaled_mem_start, temp_mem_start,
                                     mem_alloc_info):
        # 1. generate memory address for tensor between ops
        buf_makespans = lifetime_info.buffer_makespan_maps[stream_id]
        tensor_groups = set()
        for ten_group in buf_makespans.keys():
            assert not ten_group.src_from_ctx
            src_node = ten_group.src_tensor[0]
            src_node_sched = self.schedule_result.get_schedule(src_node)
            src_node_stream_id = src_node_sched.log_stream_id
            assert src_node_stream_id == stream_id

            tensor_groups.add(ten_group)

        cur_stream_seq = self.schedule_result.get_sequence(stream_id)
        local_num_steps = len(cur_stream_seq)
        group_addresses = {}
        mem_used = intervaltree.IntervalTree()  # skyline intervals
        mem_used[0:local_num_steps] = (-1, 0)  # a specific interval to cover whole range
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

            group_addresses[fixed_ten_group] = (scaled_mem_start+min_max_addr, scaled_size)
            tensor_groups.remove(fixed_ten_group)

        # 2. generate memory address for temp allocation inside an op
        max_temp_addr = 0
        for node in cur_stream_seq:
            temp_vars = self.graph_mem_info.get_temp_vars(node)
            node_temp_addr = temp_mem_start
            for temp_var in temp_vars:
                if node.name not in mem_alloc_info:
                    mem_alloc_info[node.name] = [(temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0)]
                else:
                    mem_alloc_info[node.name].append((temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0))
                node_temp_addr += temp_var.mem_size
            if node_temp_addr > max_temp_addr:
                max_temp_addr = node_temp_addr

        max_temp_end_addr = max_temp_addr
        return (group_addresses, max_temp_end_addr)

    def plan_mem_addr_by_ignore_reuse(self, lifetime_info, stream_id,
                                      scaled_mem_start, temp_mem_start,
                                      mem_alloc_info):                      # as a golden for memory reusing version
        # 1. generate memory address for tensor between ops
        buf_makespans = lifetime_info.buffer_makespan_maps[stream_id]
        tensor_groups = set()
        for ten_group in buf_makespans.keys():
            assert not ten_group.src_from_ctx
            src_node = ten_group.src_tensor[0]
            src_node_sched = self.schedule_result.get_schedule(src_node)
            src_node_stream_id = src_node_sched.log_stream_id
            assert src_node_stream_id == stream_id

            tensor_groups.add(ten_group)

        group_addresses = {}
        max_addr = 0;
        for ten_group in tensor_groups:
            scaled_size = (ten_group.src_tensor_size + self.gcd - 1) // self.gcd
            group_addresses[ten_group] = (scaled_mem_start+max_addr, scaled_size)
            max_addr += scaled_size

        # 2. generate memory address for temp allocation inside an op
        node_temp_addr = temp_mem_start
        cur_stream_seq = self.schedule_result.get_sequence(stream_id)
        for node in cur_stream_seq:
            temp_vars = self.graph_mem_info.get_temp_vars(node)
            for temp_var in temp_vars:
                if node.name not in mem_alloc_info:
                    mem_alloc_info[node.name] = [(temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0)]
                else:
                    mem_alloc_info[node.name].append((temp_var.alloc_index, node_temp_addr, temp_var.mem_size, 0))
                node_temp_addr += temp_var.mem_size
        max_temp_end_addr = node_temp_addr

        return (group_addresses, max_temp_end_addr)

    def create_min_mem_plan(self):
        lifetime_info = self.build_lifetime()

        inter_op_mems = {}
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            inter_op_mems[node.name] = [None]*len(out_vars)

        mem_alloc_info = {}
        mem_plans = []
        scaled_mem_start = 0
        temp_mem_start = 0
        for log_stream_id in range(self.schedule_result.num_streams):
            scaled_stream_end_addr, stream_temp_end_addr, group_addresses \
                        = self.create_stream_plan(lifetime_info,
                                                  log_stream_id,
                                                  scaled_mem_start,
                                                  temp_mem_start,
                                                  mem_alloc_info,
                                                  inter_op_mems)
            scaled_mem_start = scaled_stream_end_addr
            temp_mem_start = stream_temp_end_addr
            mem_plans.append((scaled_stream_end_addr, stream_temp_end_addr, group_addresses))

        for nd_name, alloc_list in mem_alloc_info.items():
            # sort allocation by allocation index
            alloc_list.sort(key=lambda x:x[0])
            for idx, alloc_info in enumerate(alloc_list):
                assert idx == alloc_info[0], f"Invalid allocation index in node {nd_name}, alloc info: {alloc_list}"

        scaled_required_mem = 0
        required_temp_mem = 0
        for scaled_stream_end_addr, stream_temp_end_addr, _ in mem_plans:
            if scaled_required_mem < scaled_stream_end_addr:
                scaled_required_mem = scaled_stream_end_addr
            if required_temp_mem < stream_temp_end_addr:
                required_temp_mem = stream_temp_end_addr

        #print(f"scaled_required_mem: {scaled_required_mem}")
        required_mem = scaled_required_mem*self.gcd
        #print(f"required_mem: {required_mem}")
        # dump peak memory
        logger.info(f"required memory: {required_mem/1024/1024/1024}GB")
        logger.info(f"required temp memory: {required_temp_mem/1024/1024/1024}GB")

        ordered_schedules = []
        for node in self.nodes_to_schedule:
            ordered_schedules.append(node.name)

        if mdconfig.dump_mem_usage_graph:
            # dump memory addresses
            self.dump_mem_usage_graph(inter_op_mems,
                                      mem_alloc_info,
                                      required_mem,
                                      lifetime_info,
                                      mem_plans,
                                      ordered_schedules)

        return (required_mem, required_temp_mem, None, ordered_schedules, mem_alloc_info, inter_op_mems)

    def create_stream_plan(self, lifetime_info, stream_id, scaled_mem_start,
                           temp_mem_start, mem_alloc_info, inter_op_mems):
        if mdconfig.ignore_memory_reuse:
            group_addresses, stream_temp_end_addr = \
                    self.plan_mem_addr_by_ignore_reuse(lifetime_info,
                                                       stream_id,
                                                       scaled_mem_start,
                                                       temp_mem_start,
                                                       mem_alloc_info)
        else:
            group_addresses, stream_temp_end_addr = \
                    self.plan_mem_addr_by_min_skyline(lifetime_info,
                                                      stream_id,
                                                      scaled_mem_start,
                                                      temp_mem_start,
                                                      mem_alloc_info)

        scaled_stream_end_addr = self.export_mem_plan(group_addresses,
                                                    inter_op_mems,
                                                    mem_alloc_info)

        #print(f"scaled_stream_end_addr: {scaled_stream_end_addr}")
        return (scaled_stream_end_addr, stream_temp_end_addr, group_addresses)

    def dump_mem_usage_graph(self,
                             inter_op_mems,
                             mem_alloc_info,
                             required_mem,
                             lifetime_info,
                             mem_plans,
                             ordered_schedules):
        node_map = {}
        for node in self.nodes_to_schedule:
            node_map[node.name] = node

        op_out_addr_str = "op output addresses:\n"
        for idx,node_name in enumerate(ordered_schedules):
            assert node_name in inter_op_mems
            addr_vars = inter_op_mems[node_name]
            node = node_map[node_name]
            sid = self.schedule_result.get_schedule(node).log_stream_id
            node_mem_str = str(idx) + ": " + node_name + "(stream: " + str(sid) + "): "
            for addr_var in addr_vars:
                if addr_var:
                    addr = addr_var[0]
                    var = addr_var[1]
                    size = var.mem_size
                    alloc_flag = "ref" if var.is_reference else "alloc"
                    node_mem_str += "([" + str(addr) + "~" + \
                                    str(addr+size-1) + "], " + str(size) + \
                                    ", " + alloc_flag + "),"
                else:
                    node_mem_str += "([NA ~ NA], NA, NA) from context,"

            op_out_addr_str += node_mem_str + "\n"

        logger.info(op_out_addr_str)

        alloc_addr_str = "op allocated addresses:\n"
        for nd_idx,node_name in enumerate(ordered_schedules):
            alloc_addr_str += f"[{nd_idx}] {node_name}:\n"
            if node_name not in mem_alloc_info:
                continue
            alloc_list = mem_alloc_info[node_name]
            for alloc in alloc_list:
                temp_flag = "True" if alloc[3] == 0 else "False"
                alloc_addr_str += f"    {alloc[0]}). addr: {alloc[1]}, size: {alloc[2]}], is_temp: {temp_flag}\n"

        logger.info(alloc_addr_str)

        # dump memory usage in graph
        for sid in range(self.schedule_result.num_streams):
            _, ax = plt.subplots()
            seq = self.schedule_result.get_sequence(sid)
            node_num = len(seq)
            ax.axis([0,node_num,0,required_mem])
            buf_makespans = lifetime_info.buffer_makespan_maps[sid]
            group_addresses = mem_plans[sid][2]
            for group,span in buf_makespans.items():
                lb, ub = span
                left = lb
                width = ub - lb + 1
                assert width > 0
                assert group in group_addresses
                addr_size = group_addresses[group]
                bottom = addr_size[0]*self.gcd
                height = addr_size[1]*self.gcd
                ax.add_patch(plt.Rectangle((left, bottom), width, height, fill=None, alpha=1))

            f_name = f"./tmp/mem_fig{sid}.pdf"
            plt.savefig(f_name)

