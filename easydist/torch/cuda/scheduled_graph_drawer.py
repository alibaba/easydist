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

import pydot
import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer

from easydist.torch.schedule.schedule_result import ScheduleResult
from easydist.torch.schedule.lifetime_info import ScheduledLifetimeInfo

class ScheduledFxGraphDrawer(FxGraphDrawer):
    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        name: str,
        schedule_result: ScheduleResult,
        lifetime_info: ScheduledLifetimeInfo,
        ignore_getattr: bool = False,
        ignore_parameters_and_buffers: bool = False,
        skip_node_names_in_args: bool = True,
    ):
        self.schedule_result = schedule_result
        self.lifetime_info = lifetime_info
        super().__init__(graph_module, name, ignore_getattr=ignore_getattr,
                         ignore_parameters_and_buffers=ignore_parameters_and_buffers,
                         skip_node_names_in_args=skip_node_names_in_args)

    def _get_node_label(
        self,
        module: torch.fx.GraphModule,
        node: torch.fx.Node,
        skip_node_names_in_args: bool,
    ) -> str:
        orig_label = super()._get_node_label(module, node, skip_node_names_in_args)
        label = orig_label[0:-1]  # strip ending "}"

        anchors_between_streams = self.lifetime_info.anchors_between_streams
        anchor_str = ""
        if node in anchors_between_streams:
            node_anchors = anchors_between_streams[node]
            for sid,anchor in node_anchors.items():
                anchor_str += f"s{sid}:{anchor},"
        else:
            print(f"anchor is missed for node {node}")

        if node in self.schedule_result.global_node_schedules:
            node_sched = self.schedule_result.get_schedule(node)
            label += f"|idx=s{node_sched.log_stream_id}:{node_sched.local_idx}" + r"\n"
        label += f"|cross anchors=[{anchor_str}]" + r"\n"
        label += "}"

        return label

    def _to_dot(
        self,
        graph_module: torch.fx.GraphModule,
        name: str,
        ignore_getattr: bool,
        ignore_parameters_and_buffers: bool,
        skip_node_names_in_args: bool,
    ) -> pydot.Dot:
        dot_graph = super()._to_dot(graph_module, name, ignore_getattr,
                                    ignore_parameters_and_buffers,
                                    skip_node_names_in_args)

        # add edges to represent executing sequence inside a stream
        for log_id, sequence in enumerate(self.schedule_result.node_sequences):
            pre_node = None
            for node in sequence:
                if pre_node:
                    dot_graph.add_edge(pydot.Edge(pre_node.name, node.name,
                                                  label=f"s{log_id}",
                                                  style="dashed", penwidth=6,
                                                  color="orange"))
                pre_node = node

        # add edges to represent dependency between streams
        for node,node_sched in self.schedule_result.global_node_schedules.items():
            for user in node.users:
                if user not in self.schedule_result.global_node_schedules:
                    # e.g. output node is missed in the map
                    continue
                user_sched = self.schedule_result.get_schedule(user)
                if user_sched.log_stream_id != node_sched.log_stream_id:
                    label = f"s{node_sched.log_stream_id}:{node_sched.local_idx}" + \
                            f"->s{user_sched.log_stream_id}:{user_sched.local_idx}"
                    dot_graph.add_edge(pydot.Edge(node.name, user.name,
                                                  label=label, style="dashed",
                                                  penwidth=6, color="red"))

        return dot_graph

