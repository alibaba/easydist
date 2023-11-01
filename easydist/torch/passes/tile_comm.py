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

import logging

import torch

logger = logging.getLogger(__name__)


def tile_comm(fx_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    logger.info("tile communication pass")
    # Step 1: categorize nodes into computation and communication
    computation_nodes = []
    communication_nodes = []

    _cache_prev_node = {}
    _cache_post_node = {}

    def find_all_prev_nodes(node):
        prev_nodes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg.op == 'placeholder':
                    continue
                if arg in computation_nodes:
                    prev_nodes.append(arg)
                if arg not in _cache_prev_node:
                    _cache_prev_node[arg] = find_all_prev_nodes(arg)
                prev_nodes.extend(_cache_prev_node[arg])
        return prev_nodes

    def find_all_post_nodes(node):
        post_nodes = []
        for user in node.users:
            if user.op == 'output':
                continue
            if user in computation_nodes:
                post_nodes.append(user)
            if user not in _cache_post_node:
                _cache_post_node[user] = find_all_post_nodes(user)
            post_nodes.extend(_cache_post_node[user])
        return post_nodes

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if node.ed_info.is_communication():
                communication_nodes.append(node)
            else:
                computation_nodes.append(node)
    
    # Step 2: find the communication nodes that are critical
    critical_communication = {}
    for node in fx_module.graph.nodes:
        if node.op == 'call_function' and node.ed_info.is_communication():
            all_prev_nodes = set(find_all_prev_nodes(node))
            all_post_nodes = set(find_all_post_nodes(node))
            independent_nodes = set(computation_nodes) - all_prev_nodes - all_post_nodes
            
            # TODO need to extend if independent_nodes can not cover the communication
            if len(independent_nodes) == 0:
                critical_communication[node] = None

    logger.info(f"Number of critical communication nodes: {len(critical_communication)}")

    # Step 3: determine the strategy for each critical communication node

    # Step 4: tile the critical communication nodes and the computation context
    
    return fx_module