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

import torch.utils._pytree as pytree

import logging
import torch
from bitarray import bitarray
from torch.fx.node import Node
from torch.utils.flop_counter import FlopCounterMode
from torch._guards import detect_fake_mode

import easydist.config as mdconfig

class ReachabilityMap:
    def __init__(self,
                 graph: torch.fx.Graph    # each node has been assigned an unique id
                ) -> None:
        ordered_node_list = list(graph.nodes)
        node_num = len(ordered_node_list)
        # 1. build reachable matrix
        self.reachable_matrix = [bitarray(node_num) for _ in range(node_num)]
        self.node_names = [node.name for node in ordered_node_list]
        self.name_idx_map = {}
        for idx, nd_name in enumerate(self.node_names):
            self.name_idx_map[nd_name] = idx

        self.compute_cost = [0 for _ in range(node_num)]
        for idx in range(node_num):
            assert idx == ordered_node_list[idx].unique_id
            self.reachable_matrix[idx].setall(0)
            self.reachable_matrix[idx][idx] = 1    # node is reachable from itself.

        for idx in range(node_num):
            node = ordered_node_list[idx]
            if node.op == 'call_function' and hasattr(node, "meta") and 'val' in node.meta:
                fake_args = pytree.tree_map_only(torch.fx.Node, lambda n: n.meta['val'], node.args)
                fake_mode = detect_fake_mode(fake_args)
                if fake_mode is not None:
                    fake_args = pytree.tree_map_only(torch.Tensor, lambda n: fake_mode.from_tensor(n), fake_args)
                with FlopCounterMode(display=False) as flop_counter_mode:
                    node.target(*fake_args, **node.kwargs)
                counted_flops = flop_counter_mode.get_total_flops()
                counted_flops = counted_flops // (1024**2)
                if mdconfig.log_level <= logging.DEBUG:
                    print(f"flops of {node}: {counted_flops} MFlops")
                self.compute_cost[idx] = counted_flops
            assert idx == node.unique_id
            node_reachable_flags = self.reachable_matrix[idx]
            node_args_flatten = pytree.tree_flatten(node.args)[0]
            for arg in node_args_flatten:
                if isinstance(arg, Node):
                    arg_reachable_flags = self.reachable_matrix[arg.unique_id]
                    for k in range(node_num):
                        node_reachable_flags[k] |= arg_reachable_flags[k]

        # 2. build parallel matrix
        self.parallel_matrix = [bitarray(node_num) for _ in range(node_num)]
        for idx_a in range(node_num):
            self.parallel_matrix[idx_a][idx_a] = 0
            for idx_b in range(idx_a+1, node_num):
                if (
                    self.reachable_matrix[idx_a][idx_b] == 0
                    and self.reachable_matrix[idx_b][idx_a] == 0
                ):
                    self.parallel_matrix[idx_a][idx_b] = 1
                    self.parallel_matrix[idx_b][idx_a] = 1
                else:
                    self.parallel_matrix[idx_a][idx_b] = 0
                    self.parallel_matrix[idx_b][idx_a] = 0

        # 3. build parallel peers flops of each node
        self.parallel_peer_flops = [0 for _ in range(node_num)]
        for idx_a in range(node_num):
            sum_flops = 0
            for idx_b in range(node_num):
                if self.parallel_matrix[idx_a][idx_b] == 1:
                    sum_flops += self.compute_cost[idx_b]

            self.parallel_peer_flops[idx_a] = sum_flops
            if mdconfig.log_level <= logging.DEBUG:
                print(f"peer flops of {ordered_node_list[idx_a]}: {sum_flops} MFlops")

    def get_parallel_peer_flops(self, node_name: str):
        idx = self.name_idx_map[node_name]
        return self.parallel_peer_flops[idx]

    def __str__(self) -> str:
        res = "reachability matrix:"
        for idx, name in enumerate(self.node_names):
            node_reachable_flags = self.reachable_matrix[idx]
            pre_node_count = 0
            ancestors = ""
            for pre_node_idx, flag in enumerate(node_reachable_flags):
                if flag == 1:
                    ancestors += self.node_names[pre_node_idx] + ", "
                    pre_node_count += 1
                    if pre_node_count % 10 == 0:
                        ancestors += "\n"
            res += "\nnode idx: " + str(idx) + ", name: " + name + \
                   ", ancestors num: " + str(pre_node_count) + ":\n" + ancestors

        res += "\nparallel matrix:"
        for idx, name in enumerate(self.node_names):
            node_parallel_flags = self.parallel_matrix[idx]
            parallel_node_count = 0
            parallel_nodes = ""
            for parallel_node_idx, flag in enumerate(node_parallel_flags):
                if flag == 1:
                    parallel_nodes += self.node_names[parallel_node_idx] + ", "
                    parallel_node_count += 1
                    if parallel_node_count % 10 == 0:
                        parallel_nodes += "\n"
            res += "\n\nnode idx: " + str(idx) + ", name: " + name + \
                   ", parallel node num: " + str(parallel_node_count) + ":\n" + parallel_nodes

        return res

