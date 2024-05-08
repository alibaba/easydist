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

import sys
import logging
import torch
import queue
from enum import Enum
from typing import Tuple, Set
from collections import defaultdict

from easydist.torch.schedule.schedule_result import ScheduleResult


logger = logging.getLogger(__name__)

class OverlapType(Enum):
    FORCE_SHARE = 1          # definitely share memory: inplace or getitem operator
    OVERLAP_NOT_SHARE = 2    # lifetimes definitely overlap: neither inplace nor getitem operator
    POSSIBLE_OVERLAP = 3     # not sure if lifetimes overlap before scheduling
    NEVER_OVERLAP = 4        # lifetimes definitely not overlap: no requirement for share

class MemSharedTensorGroup:
    def __init__(self):
        self.tensors = {}
        self.sink_nodes_cache = set()
        self.src_tensor_size = 0
        self.src_tensor = None  # Tuple[torch.fx.Node, int]
        self.src_from_ctx = False

    def __str__(self) -> str:
        ret = f"src_tensor_size: {self.src_tensor_size}, src_tensor: {self.src_tensor}"
        if self.src_from_ctx:
            ret += "(from context)"
        for tensor,mem_size in self.tensors.items():
            ret += f"\n  node: {tensor[0].name}, out idx: {tensor[1]}, mem size: {mem_size}"

        return ret

    def set_src_tensor(self, tensor: Tuple[torch.fx.Node, int], tensor_size, src_from_ctx = False):
        assert self.src_tensor_size == 0 or (src_from_ctx and self.src_tensor_size == tensor_size)
        assert (not self.src_tensor) or (src_from_ctx and self.src_tensor == tensor)
        self.src_tensor_size = tensor_size
        self.src_tensor = tensor
        self.src_from_ctx = src_from_ctx

    def add_tensor(self, tensor: Tuple[torch.fx.Node, int], tensor_size):
        self.tensors[tensor] = tensor_size
        assert self.src_tensor_size == 0 or self.src_tensor_size >= tensor_size, f"core tensor: {src_tensor[0]}:{src_tensor[1]}, core tensore size: {self.src_tensor_size}, ref tensor size: {tensor_size}"

    def has_mult_tensors(self):
        return len(self.tensors) > 1

    def source_asap_max(self, depend_info, node_set: Set[torch.fx.Node]) -> int:
        asap_max = -1
        for tensor in self.tensors.keys():
            if tensor[0] not in node_set:
                continue
            node_asap = depend_info.asap(tensor[0])
            if asap_max < node_asap:
                asap_max = node_asap
        if asap_max == -1:
            asap_max = 0
        return asap_max

    def source_asap_min(self, depend_info, node_set: Set[torch.fx.Node]) -> int:
        asap_min = sys.maxsize
        for tensor in self.tensors.keys():
            if tensor[0] not in node_set:
                asap_min = 0
                break
            node_asap = depend_info.asap(tensor[0])
            if asap_min > node_asap:
                asap_min = node_asap
        return asap_min

    def source_alap_max(self, depend_info, node_set: Set[torch.fx.Node]) -> int:
        alap_max = -1
        for tensor in self.tensors.keys():
            if tensor[0] not in node_set:
                continue
            node_alap = depend_info.alap(tensor[0])
            if alap_max < node_alap:
                alap_max = node_alap
        if alap_max == -1:
            alap_max = 0
        return alap_max

    def source_alap_min(self, depend_info, node_set: Set[torch.fx.Node]) -> int:
        alap_min = sys.maxsize
        for tensor in self.tensors.keys():
            if tensor[0] not in node_set:
                alap_min = 0
                break
            node_alap = depend_info.alap(tensor[0])
            if alap_min > node_alap:
                alap_min = node_alap
        return alap_min

    def sink_nodes(self, node_set: Set[torch.fx.Node]) -> Set[torch.fx.Node]:
        if not self.sink_nodes_cache:
            nodes_in_group = set()
            for tensor in self.tensors.keys():
                nodes_in_group.add(tensor[0])

            for node in nodes_in_group:
                for user in node.users:
                    #if user in nodes_in_group:
                    #    continue
                    if user not in node_set:
                        continue
                    self.sink_nodes_cache.add(user)
        return self.sink_nodes_cache

    def sink_asap_max(self, depend_info, node_set: Set[torch.fx.Node]) -> int:
        sink_nodes_ = self.sink_nodes(node_set)
        asap_max = -1
        for sink in sink_nodes_:
            asap = depend_info.asap(sink)
            if asap_max < asap:
                asap_max = asap
        return asap_max

    def sink_alap_max(self, depend_info, node_set: Set[torch.fx.Node]) -> int:
        sink_nodes_ = self.sink_nodes(node_set)
        alap_max = -1
        for sink in sink_nodes_:
            alap = depend_info.alap(sink)
            if alap_max < alap:
                alap_max = alap
        return alap_max


class LifetimeInfo:
    def __init__(self,
                 graph,                 # torch.fx.Graph
                 nodes_on_all_streams,  # list of nodes we care lifetimes
                 graph_mem_info):       # GraphMemInfo
        self.graph = graph
        self.nodes_on_all_streams = nodes_on_all_streams
        self.graph_mem_info = graph_mem_info
        self.node_set = set(nodes_on_all_streams)

        # map(key: node, value: map(key: other stream id, value: smallest anchor across stream))
        self.anchors_between_streams = defaultdict(lambda: {})

        # list of map:
        # per stream maps extended with across-stream consumer nodes
        self.extended_schedule_maps = []

        # list of map:
        # per stream edge makespan maps
        self.edge_makespan_maps = None

        # list of map:
        # per stream edge makespan maps
        self.buffer_makespan_maps = None

        # build memory share groups
        # map: (node, tensor_idx) -> the set of nodes which share the same memory
        self.mem_share_info = defaultdict(lambda: MemSharedTensorGroup())
        # NOTE: It is more efficient to traverse all nodes by topological order
        for node in self.nodes_on_all_streams:
            out_vars = graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                #print(f"node: {node}, var: {out_var}")
                out_tensor = (node, out_var.out_index)
                if out_var.is_reference:
                    arg_idx = out_var.arg_index
                    tensor_idx = out_var.tensor_index
                    pre_node = node.args[arg_idx]

                    pre_shared_group = self.mem_share_info[(pre_node, tensor_idx)]
                    if out_tensor in self.mem_share_info:
                        out_shared_group = self.mem_share_info[out_tensor]
                        if id(pre_shared_group) != id(out_shared_group):
                            # merge two groups and keep pre_shared_group
                            for out_set_item in out_shared_group:
                                pre_shared_group.add_tensor(out_set_item, out_var.mem_size)
                                self.mem_share_info[out_set_item] = pre_shared_group
                    else:
                        pre_shared_group.add_tensor(out_tensor, out_var.mem_size)
                        self.mem_share_info[out_tensor] = pre_shared_group

                    if pre_node not in self.node_set:
                        pre_tensor = (pre_node, tensor_idx)
                        pre_tensor_var = graph_mem_info.get_out_var(pre_node, tensor_idx)
                        #print(f"context tensor: {pre_tensor}, size: {pre_tensor_var.mem_size}")
                        pre_shared_group.set_src_tensor(pre_tensor, pre_tensor_var.mem_size, True)
                else:
                    out_shared_group = self.mem_share_info[out_tensor]
                    out_shared_group.add_tensor(out_tensor, out_var.mem_size)
                    out_shared_group.set_src_tensor(out_tensor, out_var.mem_size, False)

        visited_group_set = set()
        for tensor_group in self.mem_share_info.values():
            if id(tensor_group) in visited_group_set:
                continue

            visited_group_set.add(id(tensor_group))

            # debug: dump memory-shared tensor group
            print(str(tensor_group))

            assert tensor_group.src_tensor_size > 0, f"no core tensor in tensor group: {str(tensor_group)}"

    def get_group(self, node: torch.fx.Node, out_idx: int):
        assert (node, out_idx) in self.mem_share_info
        return self.mem_share_info[(node, out_idx)]


class UnscheduledLifetimeInfo(LifetimeInfo):
    def __init__(self,
                 graph,                 # torch.fx.Graph
                 nodes_on_all_streams,  # list of nodes we care lifetimes
                 graph_mem_info,        # GraphMemInfo
                 pre_scheded_nodes,
                 one_step_one_op):
        super().__init__(graph, nodes_on_all_streams, graph_mem_info)

        self.cache_prev_nodes = {}
        self.cache_post_nodes = {}

        # map: node -> (asap, alap)
        self.node_makespans = {}

        # map: edge(node out) -> (asap, alap)
        self.edge_makespans = {}

        self.buffer_makespans = {}

        self.one_step_one_op = one_step_one_op
        if one_step_one_op:
            self.num_steps = len(self.nodes_on_all_streams)
        else:
            depths = {}
            self.max_depth = 0
            for node in self.nodes_on_all_streams:
                if node in depths:
                    continue
                max_arg_depth = 0
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        if arg not in self.node_set:
                            continue
                        assert arg in depths
                        max_arg_depth = max(max_arg_depth, depths[arg])
                depth = max_arg_depth+1
                depths[node] = depth
                self.max_depth = max(self.max_depth, depth)

            self.num_steps = self.max_depth + 1

    def get_overlap_type(self, node1: torch.fx.Node, out_idx1: int,
                         node2: torch.fx.Node, out_idx2: int) -> OverlapType:
        # lifetime relationship between two variables:
        # 1. var1 memory and var2 memory never overlap
        #    1.1. all users of one var are predecessors of another var's source
        #         set type OverlapType.NEVER_OVERLAP
        #    1.2. spans of two variables don't overlap
        #         set type OverlapType.NEVER_OVERLAP
        # 2. var1 memory and var2 memory definitely overlap sometime
        #    2.1. var1 and var2 are in the same shared memory group
        #         set type OverlapType.FORCE_SHARE
        #    2.2. var1's memory and var2's memory are used by the same node.
        #         such as
        #             1). var1 and var2 are connected by a node
        #             2). one var in var1's memory-shared group and another var
        #                 in var2's memory-shared group are connected by a node
        #         set type OverlapType.OVERLAP_NOT_SHARE
        #    2.3. determine share type by analyzing variable span:
        #         define range1 as below:
        #         [min(alap(sources(var1_mem))), max(asap(sinks(var1_mem)))]
        #         where var1's memory is kept in use during range1.
        #         define range2 as below:
        #         [min(alap(sources(var2_mem))), max(asap(sinks(var2_mem)))]
        #         where var2's memory is kept in use during range2.
        #         2.3.1. If any variable that resides in var2_mem is definitely
        #                created in range1, that means var1's memory is
        #                overlapped with var2's memory
        #                set type OverlapType.OVERLAP_NOT_SHARE
        #         2.3.2. If any variable that resides in var1_mem is definitely
        #                created in range2, that means var2's memory is
        #                overlapped with var1's memory
        #                set type OverlapType.OVERLAP_NOT_SHARE
        # 3. others:
        #         var1 memory and var2 memory may overlap or not, it depends on
        #         schedule:
        #         set type OverlapType.POSSIBLE_OVERLAP

        tensor1_key = (node1, out_idx1)
        tensor1_group = self.mem_share_info[tensor1_key]
        tensor2_key = (node2, out_idx2)
        tensor2_group = self.mem_share_info[tensor2_key]
        if id(tensor1_group) == id(tensor2_group):
            logger.info(f"{node1}:{out_idx1} with rng [{self.edge_makespans[node1]}] and {node2}:{out_idx2} with rng [{self.edge_makespans[node2]}] share memory")
            return OverlapType.FORCE_SHARE

        src_asap_min1, snk_alap_max1 = self.get_possible_live_span(tensor1_group)
        src_asap_min2, snk_alap_max2 = self.get_possible_live_span(tensor2_group)

        if snk_alap_max1 < src_asap_min2 or src_asap_min1 > snk_alap_max2:
            # two tensors never overlap
            #edge_lb1, edge_ub1 = self.edge_makespans[node1]
            #edge_lb2, edge_ub2 = self.edge_makespans[node2]
            #print(("(1. {}:{}) with rng [{}, {}] and (2. {}:{}) with rng [{}, {}] never overlap").format(node1, out_idx1, edge_lb1, edge_ub1, node2, out_idx2, edge_lb2, edge_ub2))
            #print(("  1. src asap min: {}, snk alap max: {}").format(src_asap_min1, snk_alap_max1))
            #print(("  2. src asap min: {}, snk alap max: {}").format(src_asap_min2, snk_alap_max2))
            return OverlapType.NEVER_OVERLAP

        if (
            (not tensor1_group.has_mult_tensors())
            and (not tensor2_group.has_mult_tensors())
        ):
            if self.is_user_before(node1, node2):
                # never overlap
                logger.info(f"{node1}:{out_idx1} with rng [{self.edge_makespans[node1]}] is definitely before {node2}:{out_idx2} with rng [{self.edge_makespans[node2]}]")
                return OverlapType.NEVER_OVERLAP

            if self.is_user_before(node2, node1):
                # never overlap
                logger.info(f"{node2}:{out_idx2} with rng [{self.edge_makespans[node2]}] is definitely before {node1}:{out_idx1} with rng [{self.edge_makespans[node1]}]")
                return OverlapType.NEVER_OVERLAP

        live_same_time = False
        if self.are_connected_by_node(node1, out_idx1, node2, out_idx2):
            live_same_time = True

        if not live_same_time:
            tensor1_live_span = self.get_definite_live_span(node1, out_idx1)
            if tensor1_live_span[0] <= tensor1_live_span[1]:
                if self.is_created_in_span(tensor2_group, tensor1_live_span):
                    live_same_time = True

        if not live_same_time:
            tensor2_live_span = self.get_definite_live_span(node2, out_idx2)
            if tensor2_live_span[0] <= tensor2_live_span[1]:
                if self.is_created_in_span(tensor1_group, tensor2_live_span):
                    live_same_time = True

        if live_same_time:
            return OverlapType.OVERLAP_NOT_SHARE
        else:
            return OverlapType.POSSIBLE_OVERLAP

    def is_created_in_span(self, tensor_group, span):
        for tensor in tensor_group.tensors.keys():
            node_asap, node_alap = self.node_makespans[tensor[0]]
            if span[0] <= node_asap and span[1] >= node_alap:
                return True

        return False

    def get_possible_live_span(self, tensor_group):
        if tensor_group.has_mult_tensors():
            src_asap_min = tensor_group.source_asap_min(self, self.node_set)
            if src_asap_min == -1:
                src_asap_min = 0
            snk_alap_max = tensor_group.sink_alap_max(self, self.node_set)
            if snk_alap_max == -1:
                snk_alap_max = self.num_steps - 1
        else:
            tensor_lst = list(tensor_group.tensors.keys())
            node = tensor_lst[0][0]
            src_asap_min, snk_alap_max = self.edge_makespans[node]

        return (src_asap_min, snk_alap_max)

    def get_definite_live_span(self, node, out_idx):
        tensor_key = (node, out_idx)
        tensor_group = self.mem_share_info[tensor_key]
        if tensor_group.has_mult_tensors():
            src_alap_min = tensor_group.source_alap_min(self, self.node_set)
            if src_alap_min == -1:
                src_alap_min = 0
            snk_asap_max = tensor_group.sink_asap_max(self, self.node_set)
            if snk_asap_max == -1:
                snk_asap_max = self.num_steps - 1
        else:
            tensor_lst = list(tensor_group.tensors.keys())
            node = tensor_lst[0][0]
            src_alap_min, snk_asap_max = self.edge_makespans[node]

        return (src_alap_min, snk_asap_max)

    def are_connected_by_node(self, node1, out_idx1, node2, out_idx2):
        nodes_of_tensor1 = set()
        for user in node1.users.keys():
            nodes_of_tensor1.add(user)
        for arg in node1.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                nodes_of_tensor1.add(arg)

        nodes_of_tensor2 = set()
        for user in node2.users.keys():
            nodes_of_tensor2.add(user)
        for arg in node2.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                nodes_of_tensor2.add(arg)
        return not nodes_of_tensor1.isdisjoint(nodes_of_tensor2)

    def is_user_before(self, node1, node2): # if users of node1 are all before node2
        pre_nodes2 = self.find_all_prev_nodes(node2)
        if node1 in pre_nodes2:
            user_before_node2 = True
            for user in node1.users:
                if user not in pre_nodes2:
                    user_before_node2 = False
                    break
            if user_before_node2:
                return True

        return False

    def find_all_prev_nodes(self, node):
        if node in self.cache_prev_nodes.keys():
            return self.cache_prev_nodes[node]

        prev_nodes = set()
        if node not in self.node_set:
            return prev_nodes

        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                prev_nodes.add(arg)
                if arg not in self.cache_prev_nodes:
                    self.cache_prev_nodes[arg] = self.find_all_prev_nodes(arg)
                prev_nodes.update(self.cache_prev_nodes[arg])
            else:
                print(("ignore arg ").format(arg))

        self.cache_prev_nodes[node] = prev_nodes
        return prev_nodes

    def find_all_post_nodes(self, node):
        if node in self.cache_post_nodes.keys():
            return self.cache_post_nodes[node]

        post_nodes = set()
        if node not in self.node_set:
            return post_nodes

        for user in node.users:
            if user not in self.node_set:
                continue
            post_nodes.add(user)
            if user not in self.cache_post_nodes:
                self.cache_post_nodes[user] = self.find_all_post_nodes(user)
            post_nodes.update(self.cache_post_nodes[user])

        self.cache_post_nodes[node] = post_nodes
        return post_nodes

    def asap(self, node) -> int:
        assert self.node_makespans
        _asap,_ = self.node_makespans[node]
        return _asap

    def alap(self, node) -> int:
        assert self.node_makespans
        _, _alap = self.node_makespans[node]
        return _alap

    def compute_asap_by_depth(self):
        asaps = {}
        for node in self.nodes_on_all_streams:
            self._compute_asap_by_depth(node, asaps)
        return asaps

    def _compute_asap_by_depth(self, node, asaps):
        if node in asaps:
            return asaps[node]

        asap = 0
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                arg_asap = self._compute_asap_by_depth(arg, asaps)
                asap = max(arg_asap+1, asap)
        if self.pre_scheded_nodes and node in self.pre_scheded_nodes:
            if asap > self.pre_scheded_nodes[node]:
                raise ValueError(
                    "Infeasible user schedule constraint for node %s: user specified at %d, min feasible %d"
                    % (node.name, self.pre_scheded_nodes[node], asap))
            asap = self.pre_scheded_nodes[node]

        asaps[node] = asap
        return asap

    def compute_alap_by_depth(self):
        alaps = {}
        for node in self.nodes_on_all_streams:
            self._compute_alap_by_depth(node, alaps)
        return alaps

    def _compute_alap_by_depth(self, node, alaps):
        if node in alaps:
            return alaps[node]

        alap = self.max_depth
        for user in node.users:
            if user not in self.node_set:
                continue
            user_alap = self._compute_alap_by_depth(user, alaps)
            alap = min(user_alap-1, alap)

        if self.pre_scheded_nodes and node in self.pre_scheded_nodes:
            if alap < self.pre_scheded_nodes[node]:
                raise ValueError(
                    "Infeasible user schedule constraint for node %s: user specified at %d, max feasible %d"
                    % (node.name, self.pre_scheded_nodes[node], alap))
            alap = self.pre_scheded_nodes[node]

        alaps[node] = alap
        return alap

    def compute_node_makespans_by_depth(self):
        asaps = self.compute_asap_by_depth()
        alaps = self.compute_alap_by_depth()
        for node in self.nodes_on_all_streams:
            asap = asaps[node]
            alap = alaps[node]
            self.node_makespans[node] = (asap, alap)
        return self.node_makespans

    def compute_node_makespans_by_cone(self):
        # refresh post cache for post nodes to avoid calling deep incursive call
        for node in reversed(self.nodes_on_all_streams):
            self.find_all_post_nodes(node)

        for node in self.nodes_on_all_streams:
            prev_node_num = len(self.find_all_prev_nodes(node))
            post_node_num = len(self.find_all_post_nodes(node))

            # compute the asap and alap
            self.node_makespans[node] = (prev_node_num, node_num - post_node_num - 1)
        return self.node_makespans

    def compute_node_makespans(self):
        assert not self.node_makespans
        node_num = len(self.nodes_on_all_streams)
        if self.pre_scheded_nodes:
            for node,step in self.pre_scheded_nodes.items():
                self.node_makespans[node] = (step, step)

        if self.node_makespans:
            assert len(self.node_makespans) == node_num, \
                f"scheduled node num: {len(self.node_makespans)}, node num: {node_num}"
            return self.node_makespans

        if self.one_step_one_op:
            return self.compute_node_makespans_by_cone()
        else:
            return self.compute_node_makespans_by_depth()

    def compute_edge_makespans(self):
        # must compute node makespans(asap/alap) before computing edge makespans
        self.compute_node_makespans()

        for node in self.nodes_on_all_streams:
            lb, ub = self.node_makespans[node]
            user_ub_max = ub
            for user in node.users:
                if user not in self.node_set:
                    user_ub_max = self.num_steps - 1
                    break
                _, user_ub = self.node_makespans[user]
                user_ub_max = max(user_ub_max, user_ub)

            # node represents its output edge
            self.edge_makespans[node] = (lb, user_ub_max)

        return self.edge_makespans

    def compute_buffer_makespans(self):
        if not self.edge_makespans:
            self.compute_edge_makespans()

        for nd in self.nodes_on_all_streams:
            assert nd in self.edge_makespans
            lb, ub = self.edge_makespans[nd]
            out_vars = self.graph_mem_info.get_out_vars(nd)
            for var in out_vars:
                out_tensor = (nd, var.out_index)
                out_shared_group = self.mem_share_info[out_tensor]
                if out_shared_group.src_from_ctx:
                    if out_shared_group not in self.buffer_makespans:
                        self.buffer_makespans[out_shared_group] = [0, self.num_steps - 1]
                    continue
                if out_shared_group in self.buffer_makespans:
                    buffer_span = self.buffer_makespans[out_shared_group]
                    if lb < buffer_span[0]:
                        buffer_span[0] = lb
                    if ub > buffer_span[1]:
                        buffer_span[1] = ub
                else:
                    self.buffer_makespans[out_shared_group] = [lb, ub]

        return self.buffer_makespans

class ScheduledLifetimeInfo(LifetimeInfo):
    def __init__(self,
                 graph,                 # torch.fx.Graph
                 nodes_on_all_streams,  # list of nodes we care lifetimes
                 graph_mem_info,        # GraphMemInfo
                 schedule_result):      # ScheduleResult
        super().__init__(graph, nodes_on_all_streams, graph_mem_info)

        self.schedule_result = schedule_result

        self.nodes_out_of_ctx = []

    def build_extended_schedule_maps(self):
        # copy original per stream schedule
        for node_idx_map in self.schedule_result.node_idx_maps:
            ext_sched_map = node_idx_map.copy()
            self.extended_schedule_maps.append(ext_sched_map)
            self.nodes_out_of_ctx.append(set())

        # update schedule map with across-stream consumers
        for sid in range(self.schedule_result.num_streams):
            seq = self.schedule_result.get_sequence(sid)
            last_pos = len(seq) - 1
            for node in seq:
                for user in node.users:
                    if user not in self.node_set:
                        continue
                    user_schedule = self.schedule_result.get_schedule(user)
                    user_sid = user_schedule.log_stream_id
                    if user_sid == sid:
                        continue

                    self.nodes_out_of_ctx[sid].add(user)
                    if (
                        (user not in self.anchors_between_streams) or
                        (sid not in self.anchors_between_streams[user])
                    ):
                        assert user not in self.extended_schedule_maps[sid]
                        self.extended_schedule_maps[sid][user] = last_pos
                    else:
                        anchor_pos = self.anchors_between_streams[user][sid]
                        assert anchor_pos > 0
                        if user not in self.extended_schedule_maps:
                            self.extended_schedule_maps[sid][user] = last_pos        # Lansong(TODO): safe but more memory usage
                            #self.extended_schedule_maps[sid][user] = anchor_pos - 1  # Lansong(TODO): aggressive version, a bug to be fixed, uncomment it later
                        elif self.extended_schedule_maps[sid][user] < (anchor_pos - 1):
                            #print(f"update anchor position for {node} due to user {user}")
                            self.extended_schedule_maps[sid][user] = last_pos        # Lansong(TODO): safe but more memory usage
                            #self.extended_schedule_maps[sid][user] = anchor_pos - 1  # Lansong(TODO): aggressive version, a bug to be fixed, uncomment it later

        # debug
        #for sid in range(self.schedule_result.num_streams):
        #    print(f"stream {sid}:\n  extended_schedule_map:\n  {self.extended_schedule_maps[sid]}")
        #    print(f"  nodes out of context:\n  {self.nodes_out_of_ctx[sid]}")

    def build_anchors_between_streams(self):
        stream_pointers = [0]*self.schedule_result.num_streams
        for sid in range(self.schedule_result.num_streams):
            seq = self.schedule_result.get_sequence(sid)
            stream_pointers[sid] = len(seq)-1

        rdy_queue = queue.Queue()

        # initialize ready queue
        for sid in range(self.schedule_result.num_streams):
            last_node = self.schedule_result.get_node(sid, stream_pointers[sid])
            ready = True
            for user in last_node.users:
                if (
                    user in self.node_set and
                    user not in self.anchors_between_streams
                ):
                    ready = False
                    break
            if ready:
                rdy_queue.put(last_node)
                stream_pointers[sid] -= 1

        while not rdy_queue.empty():
            cur_node = rdy_queue.get()
            cur_schedule = self.schedule_result.get_schedule(cur_node)
            cur_sid = cur_schedule.log_stream_id
            cur_seq = self.schedule_result.get_sequence(cur_sid)
            cur_seq_ub = len(cur_seq)-1
            cur_pos = cur_schedule.local_idx
            assert stream_pointers[sid] < cur_schedule.local_idx

            ready = True
            cur_anchors_between_streams = {}   # other stream id(int) -> smallest anchor(int)
            for user in cur_node.users:
                if user not in self.node_set:
                    continue
                user_schedule = self.schedule_result.get_schedule(user)
                user_sid = user_schedule.log_stream_id
                if user_sid == cur_sid:
                    continue
                user_idx = user_schedule.local_idx
                #print(f"cur node {cur_node}, user {user}, user schedule: ({user_sid}, {user_idx})")
                if user_sid in cur_anchors_between_streams:
                    if user_idx < cur_anchors_between_streams[user_sid]:
                        cur_anchors_between_streams[user_sid] = user_idx
                else:
                    cur_anchors_between_streams[user_sid] = user_idx

            successor_anchors_between_streams = {}
            if cur_pos < cur_seq_ub:
                successor = cur_seq[cur_pos+1]
                successor_anchors_between_streams = self.anchors_between_streams[successor]
                #print(f"cur_sid {cur_sid}, cur_seq_ub {cur_seq_ub}, cur_pos {cur_pos}, successor {successor},\nsuccessor_anchors_between_streams {successor_anchors_between_streams}")

            assert cur_node not in self.anchors_between_streams, f"{cur_node} records anchors twice"

            #print(f"current node {cur_node}, cur_anchors_between_streams: {cur_anchors_between_streams}")

            # merge successor_anchors_between_streams and cur_anchors_between_streams
            if not cur_anchors_between_streams:
                self.anchors_between_streams[cur_node] = successor_anchors_between_streams
                #print(f"final current node {cur_node}, successor_anchors_between_streams: {successor_anchors_between_streams}")
            elif not successor_anchors_between_streams:
                self.anchors_between_streams[cur_node] = cur_anchors_between_streams
                #print(f"final current node {cur_node}, cur_anchors_between_streams: {cur_anchors_between_streams}")
            else:
                for sid,anchor in successor_anchors_between_streams.items():
                    assert sid != cur_sid
                    if sid not in cur_anchors_between_streams:
                        cur_anchors_between_streams[sid] = anchor
                    else:
                        if anchor < cur_anchors_between_streams[sid]:
                            cur_anchors_between_streams[sid] = anchor

                self.anchors_between_streams[cur_node] = cur_anchors_between_streams
                #print(f"final current node {cur_node}, updated cur_anchors_between_streams: {cur_anchors_between_streams}")

            # update ready queue
            for sid in range(self.schedule_result.num_streams):
                pos = stream_pointers[sid]
                if pos < 0:
                    continue
                seq = self.schedule_result.get_sequence(sid)
                pre_node = seq[pos]
                pre_rdy = True
                for user in pre_node.users:
                    if (
                        user in self.node_set and
                        user not in self.anchors_between_streams
                    ):
                        pre_rdy = False
                        break
                if pre_rdy:
                    rdy_queue.put(pre_node)
                    stream_pointers[sid] = pos-1

    def compute_edge_makespans(self):
        self.edge_makespan_maps = [{} for _ in range(self.schedule_result.num_streams)]
        for sid in range(self.schedule_result.num_streams):
            max_step = len(self.schedule_result.node_sequences[sid])-1
            for node,step in self.extended_schedule_maps[sid].items():
                if node in self.nodes_out_of_ctx[sid]:
                    continue

                user_max_step = step
                for user in node.users:
                    if user not in self.extended_schedule_maps[sid]:
                        user_max_step = max_step
                        break
                    user_step = self.extended_schedule_maps[sid][user]
                    user_max_step = max(user_max_step, user_step)

                self.edge_makespan_maps[sid][node] = (step, user_max_step)

        # debug
        #for sid in range(self.schedule_result.num_streams):
        #    print(f"edge makespan on stream {sid}:\n{self.edge_makespan_maps[sid]}")

    def compute_buffer_makespans(self):
        self.buffer_makespan_maps = [{} for _ in range(self.schedule_result.num_streams)]
        if not self.edge_makespan_maps:
            self.compute_edge_makespans()
        for sid in range(self.schedule_result.num_streams):
            cur_seq = self.schedule_result.get_sequence(sid)
            cur_edge_makespan_map = self.edge_makespan_maps[sid]
            cur_buffer_makespan_map = self.buffer_makespan_maps[sid]
            cur_max_step = len(cur_seq)-1

            for nd in cur_seq:
                assert nd in cur_edge_makespan_map
                lb, ub = cur_edge_makespan_map[nd]
                out_vars = self.graph_mem_info.get_out_vars(nd)
                for var in out_vars:
                    out_tensor = (nd, var.out_index)
                    out_shared_group = self.mem_share_info[out_tensor]
                    if out_shared_group.src_from_ctx:
                        #if out_shared_group not in cur_buffer_makespan_map:
                        #    cur_buffer_makespan_map[out_shared_group] = [0, cur_max_step]
                        continue

                    src_node = out_shared_group.src_tensor[0]
                    src_node_sched = self.schedule_result.get_schedule(src_node)
                    src_node_stream_id = src_node_sched.log_stream_id
                    if src_node_stream_id != sid:
                        continue

                    if out_shared_group in cur_buffer_makespan_map:
                        buffer_span = cur_buffer_makespan_map[out_shared_group]
                        if lb < buffer_span[0]:
                            buffer_span[0] = lb
                        if ub > buffer_span[1]:
                            buffer_span[1] = ub
                    else:
                        cur_buffer_makespan_map[out_shared_group] = [lb, ub]


