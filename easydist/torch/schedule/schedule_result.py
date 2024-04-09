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

import torch

class NodeSchedule:
    def __init__(
        self,
        log_stream_id,
        local_idx,
    ):
        self.log_stream_id = log_stream_id
        self.local_idx = local_idx

class ScheduleResult:
    def __init__(
        self
    ):
        self.node_sequences = []    # node sequence list, i.e. list of list
        self.node_idx_maps = []     # per stream maps
        self.phy_log_stream_id_map = {}
        self.phy_stream_ids = []
        self.num_streams = 0
        self.global_node_schedules = {} # node -> node schedule(NodeSchedule)

    def schedule_node_at_end(self, node: torch.fx.Node, phy_stream_id: int):
        if phy_stream_id not in self.phy_log_stream_id_map:
            log_stream_id = self.num_streams
            self.num_streams += 1
            self.phy_stream_ids.append(phy_stream_id)
            self.phy_log_stream_id_map[phy_stream_id] = log_stream_id
        else:
            log_stream_id = self.phy_log_stream_id_map[phy_stream_id]
        while log_stream_id >= len(self.node_sequences):
            self.node_sequences.append([])
            self.node_idx_maps.append({})

        sequence = self.node_sequences[log_stream_id]
        node_idx_map = self.node_idx_maps[log_stream_id]
        local_idx = len(sequence)
        node_schedule = NodeSchedule(log_stream_id, local_idx)
        sequence.append(node)
        node_idx_map[node] = local_idx
        assert node not in self.global_node_schedules
        self.global_node_schedules[node] = node_schedule

    def get_node_idx_map(self, stream_id: int):
        return self.node_idx_maps[stream_id]

    def get_schedule(self, node: torch.fx.Node):
        assert node in self.global_node_schedules
        return self.global_node_schedules[node]

    def get_log_stream_id(self, node: torch.fx.Node):
        assert node in self.global_node_schedules
        return self.global_node_schedules[node].log_stream_id

    def get_sequence(self, log_stream_id: int):
        return self.node_sequences[log_stream_id]

    def get_node(self, log_stream_id: int, local_idx: int):
        return self.node_sequences[log_stream_id][local_idx]

    def __str__(self) -> str:
        ret = ""
        for log_id, sequence in enumerate(self.node_sequences):
            ret += f"logic stream id: {log_id}, real stream id: {self.phy_stream_ids[log_id]}\n"
            for idx, node in enumerate(sequence):
                ret += f"  {idx}: {node.name}\n"
        return ret

