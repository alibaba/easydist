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
import logging
from torch.fx.passes.graph_drawer import FxGraphDrawer

import easydist.config as mdconfig
from easydist.torch.schedule.schedule_result import ScheduleResult
from easydist.torch.schedule.lifetime_info import ScheduledLifetimeInfo
from packaging.version import parse as parse_version

torch_version = torch.__version__
parsed_version = parse_version(torch_version)

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
        parse_stack_trace: bool,
    ) -> str:
        if parsed_version >= parse_version("2.2.0"):
            orig_label = super()._get_node_label(module, node, skip_node_names_in_args, parse_stack_trace)
        else:
            orig_label = super()._get_node_label(module, node, skip_node_names_in_args)

        label = orig_label[0:-1]  # strip ending "}"

        # 1. local index information
        if node in self.schedule_result.global_node_schedules:
            node_sched = self.schedule_result.get_schedule(node)
            label += f"|idx=s{node_sched.log_stream_id}:{node_sched.local_idx}\n"

        # 2. anchor information
        anchors_between_streams = self.lifetime_info.anchors_between_streams
        anchor_str = ""
        if node in anchors_between_streams:
            node_anchors = anchors_between_streams[node]
            for sid,anchor in node_anchors.items():
                anchor_str += f"s{sid}:{anchor},"
        else:
            if mdconfig.log_level <= logging.DEBUG:
                print(f"anchor is missed for node {node}")

        label += f"|cross anchors=[{anchor_str}]\n"
        label += "}"

        return label

    def _to_dot(
        self,
        graph_module: torch.fx.GraphModule,
        name: str,
        ignore_getattr: bool,
        ignore_parameters_and_buffers: bool,
        skip_node_names_in_args: bool,
        parse_stack_trace: bool,
    ) -> pydot.Dot:
        if parsed_version >= parse_version("2.2.0"):
            dot_graph = super()._to_dot(graph_module, name, ignore_getattr,
                                        ignore_parameters_and_buffers,
                                        skip_node_names_in_args,
                                        parse_stack_trace)
        else:
            dot_graph = super()._to_dot(graph_module, name, ignore_getattr,
                                        ignore_parameters_and_buffers,
                                        skip_node_names_in_args)

        # 1. add edges to represent executing sequence inside a stream
        for sid, sequence in enumerate(self.schedule_result.node_sequences):
            pre_node = None
            for node in sequence:
                if pre_node:
                    dot_graph.add_edge(pydot.Edge(pre_node.name, node.name,
                                                  label=f"s{sid}",
                                                  style="dashed", penwidth=6,
                                                  color="orange"))
                pre_node = node

        for node,node_sched in self.schedule_result.global_node_schedules.items():
            node_sid = node_sched.log_stream_id
            for user in node.users:
                if user not in self.schedule_result.global_node_schedules:
                    # e.g. output node is missed in the map
                    continue
                user_sched = self.schedule_result.get_schedule(user)
                if user_sched.log_stream_id == node_sid:
                    continue

                # 2. add an edge to represent op dependency across streams
                label = f"s{node_sid}:{node_sched.local_idx}" + \
                        f"->s{user_sched.log_stream_id}:{user_sched.local_idx}"
                dot_graph.add_edge(pydot.Edge(node.name, user.name,
                                              label=label, style="dashed",
                                              penwidth=6, color="red"))

                # 3. add an edge from user to mirror node
                if user in self.lifetime_info.extended_schedule_maps[node_sid]:
                    # get mirror position on other streams (derived through anchors)
                    mirror_pos = self.lifetime_info.extended_schedule_maps[node_sid][user]
                    mirror_node = self.schedule_result.node_sequences[node_sid][mirror_pos]
                    mirror_sched = self.schedule_result.get_schedule(mirror_node)
                    assert mirror_sched.log_stream_id == node_sid, (
                               f"{user} mirror node {mirror_node} is on stream"
                               f"{mirror_sched.log_stream_id}, but {node} is on"
                               f" stream {node_sid}"
                           )
                    assert mirror_sched.local_idx == mirror_pos, (
                               f"{user} mirror node idx is {mirror_sched.local_idx}"
                               f", but mirror_pos is {mirror_pos}"
                           )
                    # create an edge
                    label = (
                        f"user:s{user_sched.log_stream_id}:{user_sched.local_idx}"
                        f"->mirror:s{node_sid}:{mirror_pos}"
                    )
                    dot_graph.add_edge(pydot.Edge(user.name, mirror_node.name,
                                                  label=label, style="dashed",
                                                  penwidth=6, color="blue"))

        # 4. add lifetime edges for dependency happens across stream
        for sid, makespan_map in enumerate(self.lifetime_info.edge_makespan_maps):
            for node,span in makespan_map.items():
                across_stream = False
                for user in node.users:
                    if user not in self.schedule_result.global_node_schedules:
                        if mdconfig.log_level <= logging.DEBUG:
                            print(f"user {user} not in global node schedules")
                        # e.g. output node is missed in the map
                        continue
                    user_sched = self.schedule_result.get_schedule(user)
                    if user_sched.log_stream_id != sid:
                        across_stream = True
                        break
                    else:
                        if mdconfig.log_level <= logging.DEBUG:
                            print(f"user stream id {user_sched.log_stream_id} is equal to node stream id {sid}")

                if across_stream:
                    end_node = self.schedule_result.node_sequences[sid][span[1]]
                    label = f"{node.name}:{span[0]}->{end_node.name}:{span[1]}"
                    dot_graph.add_edge(pydot.Edge(node.name, end_node.name,
                                                  label=label, style="dashed",
                                                  penwidth=9, color="purple"))

                if mdconfig.log_level <= logging.DEBUG:
                    if across_stream:
                        print(f"[across lifetime] sid: {sid}, node: {node}, span: {span}")
                    else:
                        print(f"[simple lifetime] sid: {sid}, node: {node}, span: {span}")
        return dot_graph

