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

import numpy as np
import logging
import torch
from collections import defaultdict

import easydist.config as mdconfig
from easydist.torch.schedule.schedule_result import ScheduleResult

logger = logging.getLogger(__name__)


class MemoryScheduler:
    def __init__(
        self,
        fx_module,      # torch.fx.GraphModule
        graph_mem_info, # GraphMemInfo
        op_streams,
        align_scale
    ):
        self.fx_module = fx_module
        self.graph_mem_info = graph_mem_info

        self.nodes_to_schedule = []
        self.args = []
        self.outputs = []
        self.op_streams = op_streams
        self.schedule_result = None

        for node in fx_module.graph.nodes:
            if node.op == 'placeholder' or node.op == 'get_attr':
                self.args.append(node)
            elif node.op == 'output':
                self.outputs.append(node)
            else:
                self.nodes_to_schedule.append(node)
                assert node.name in op_streams, f"node {node.name} misses stream id"

        self.node_set = set(self.nodes_to_schedule)

        tensor_sizes = set()
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                tensor_sizes.add(out_var.size())

        logger.info(f"memory align value: {align_scale}")

        align_sizes = [(ten_size+align_scale-1)//align_scale for ten_size in tensor_sizes]

        self.align_scale = align_scale
        self.gcd = np.gcd.reduce(list(align_sizes))
        self.gcd *= align_scale
        logger.info(f"gcd of memory sizes: {self.gcd}")

    def gen_mem_addresses(self):
        if not mdconfig.enable_reschedule:
            self.schedule_result = ScheduleResult()
            for node in self.fx_module.graph.nodes:
                if (
                    node.op != 'output' and
                    node.op != 'placeholder' and
                    node.op != 'get_attr'
                ):
                    phy_stream_id = self.op_streams[node.name]
                    self.schedule_result.schedule_node_at_end(node, phy_stream_id)

        #print(f"node schedule result:\n{str(self.schedule_result)}")
        required_memory, temp_memory, schedules, ordered_schedules, mem_alloc_info, inter_op_mems = \
                                                      self.create_min_mem_plan()

        return (required_memory, temp_memory, schedules, ordered_schedules, mem_alloc_info, inter_op_mems)

