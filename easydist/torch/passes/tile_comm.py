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
from dataclasses import dataclass, field
from typing import List, Dict
import operator

import mip
import rich
import torch
import torch.utils._pytree as pytree

from easydist import mdconfig
from easydist.metashard.combination import CombinationFunc
from easydist.torch.utils import EDInfo, EDNodeType, create_meta_from_node
from .sharding import (all_gather_start, all_reduce_start, all_gather_end, all_reduce_end,
                       view_op_map, scatter_wrapper)

logger = logging.getLogger(__name__)


@dataclass
class TiledNode:
    node: torch.fx.Node
    tile_axis: int
    num_tiles: int


@dataclass
class TileStrategy:
    num_tiles: int
    tms_comm: float = 0.0
    tms_prev_comp: float = 0.0
    tms_post_comp: float = 0.0
    tiled_comm_node: List[TiledNode] = field(default_factory=list)
    tiled_prev_node: List[TiledNode] = field(default_factory=list)
    tiled_post_node: List[TiledNode] = field(default_factory=list)

    def append_prev_node(self, tiled_node: TiledNode):
        if self.tiled_prev_node is None:
            self.tiled_prev_node = []
        self.tiled_prev_node.append(tiled_node)
        self.tms_prev_comp += tiled_node.node.ed_info.runtime_ms

    def append_post_node(self, tiled_node: TiledNode):
        if self.tiled_post_node is None:
            self.tiled_post_node = []
        self.tiled_post_node.append(tiled_node)
        self.tms_post_comp += tiled_node.node.ed_info.runtime_ms


def forward_tiled_node(tiled_node: TiledNode, user: torch.fx.Node) -> TiledNode:

    if user.target == scatter_wrapper:
        scatter_dim = user.args[2]
        if tiled_node.tile_axis != scatter_dim:
            return TiledNode(user, tiled_node.tile_axis, tiled_node.num_tiles)
        else:
            return None

    flatten_args, _ = pytree.tree_flatten(user.args)
    flatten_tensor_args = [i for i in flatten_args if isinstance(i, torch.fx.Node)]

    arg_index = flatten_tensor_args.index(tiled_node.node)
    shard_dim = user.ed_info.spmd_annotation['sharding_ann'][arg_index][tiled_node.tile_axis]
    # is not NoShardDim
    if shard_dim.shard_dim_id != 0:
        out_spmd = user.ed_info.spmd_annotation['combination_ann'][shard_dim.shard_dim_id]
        # (TODO) support forward through multi-output operation
        if isinstance(out_spmd, list):
            return None
        if out_spmd.func == CombinationFunc.gather:
            return TiledNode(user, out_spmd.keywords['dim'], tiled_node.num_tiles)

    return None


def get_arg_tile_axis(node: torch.fx.Node, tile_axis, arg_index) -> int:

    if node.target in [all_gather_start, all_reduce_start]:
        return tile_axis

    for shard_dim_id, combine_func in node.ed_info.spmd_annotation['combination_ann'].items():
        if combine_func.func == CombinationFunc.gather and combine_func.keywords[
                'dim'] == tile_axis:
            arg_spmd = node.ed_info.spmd_annotation['sharding_ann'][arg_index]
            arg_spmd_shard_dim_id = [i.shard_dim_id for i in arg_spmd]
            if shard_dim_id in arg_spmd_shard_dim_id:
                return arg_spmd_shard_dim_id.index(shard_dim_id)
    return None


def backward_tiled_node(tiled_node: TiledNode, arg: torch.fx.Node) -> TiledNode:

    # return None for meet communication node
    if arg.target in [all_gather_end, all_reduce_end]:
        return None

    if tiled_node.node.target == scatter_wrapper:
        scatter_dim = tiled_node.node.args[2]
        if tiled_node.tile_axis != scatter_dim:
            return TiledNode(arg, tiled_node.tile_axis, tiled_node.num_tiles)
        else:
            return None

    flatten_args, _ = pytree.tree_flatten(tiled_node.node.args)
    flatten_tensor_args = [i for i in flatten_args if isinstance(i, torch.fx.Node)]
    arg_index = flatten_tensor_args.index(arg)

    tile_axis = get_arg_tile_axis(tiled_node.node, tiled_node.tile_axis, arg_index)
    if tile_axis is not None:
        return TiledNode(arg, tile_axis, tiled_node.num_tiles)

    return None


class TileSolver:

    def __init__(self, strategy_for_all: Dict[torch.fx.Node, List[TileStrategy]]) -> None:
        self.strategy_for_all = strategy_for_all
        self.m = mip.Model("tile-solver")

        self.node = {}
        self.tile_cost = 0

        for c_node in self.strategy_for_all:
            node_prev_strategy, node_post_strategy = [], []
            cnode_axis_prev_strategy, cnode_axis_post_strategy = [], []

            prev_tile_space, post_tile_space = [], []

            for strategy in self.strategy_for_all[c_node]:
                for prev_node in strategy.tiled_prev_node:
                    prev_tile_space.append(prev_node.node.name)
                for post_node in strategy.tiled_post_node:
                    post_tile_space.append(post_node.node.name)

                for i in range(len(strategy.tiled_prev_node) + 1):
                    node_prev_strategy.append(strategy.tiled_prev_node[:i])
                    cnode_axis_prev_strategy.append(strategy.tiled_comm_node[0].tile_axis)

                for i in range(len(strategy.tiled_post_node) + 1):
                    node_post_strategy.append(strategy.tiled_post_node[:i])
                    cnode_axis_post_strategy.append(strategy.tiled_comm_node[0].tile_axis)

            self.node[c_node] = {
                'prev':
                node_prev_strategy,
                'post':
                node_post_strategy,
                'cnode_axis_prev_strategy':
                cnode_axis_prev_strategy,
                'cnode_axis_post_strategy':
                cnode_axis_post_strategy,
                'prev_tile_space':
                set(prev_tile_space),
                'post_tile_space':
                set(post_tile_space),
                'prev_mip_var':
                [self.m.add_var(var_type=mip.BINARY) for _ in range(len(node_prev_strategy))],
                'post_mip_var':
                [self.m.add_var(var_type=mip.BINARY) for _ in range(len(node_post_strategy))],
            }
            shape_1 = len(self.node[c_node]['prev'])
            shape_2 = len(self.node[c_node]['post'])

            self.m += mip.xsum(self.node[c_node]['prev_mip_var'][i] for i in range(shape_1)) == 1
            self.m += mip.xsum(self.node[c_node]['post_mip_var'][i] for i in range(shape_2)) == 1

            mip_matrix = [[self.m.add_var(var_type=mip.BINARY) for _ in range(shape_2)]
                          for _ in range(shape_1)]

            for i in range(shape_1):
                for j in range(shape_2):
                    self.m += mip_matrix[i][j] <= self.node[c_node]['prev_mip_var'][i]
                    self.m += mip_matrix[i][j] <= self.node[c_node]['post_mip_var'][j]
                    self.m += mip_matrix[i][j] >= self.node[c_node]['prev_mip_var'][i] + self.node[
                        c_node]['post_mip_var'][j] - 1

            cost_matrix = self.get_cost_matrix(c_node, node_prev_strategy, node_post_strategy,
                                               cnode_axis_prev_strategy, cnode_axis_post_strategy)
            # add tile cost for each communication node
            self.tile_cost = self.tile_cost + mip.xsum(mip_matrix[i][j] * cost_matrix[i][j]
                                                       for i in range(shape_1)
                                                       for j in range(shape_2))

            # add tile cost between communication node who share the same tile node
            for node in self.node:
                if node == c_node:
                    continue
                if any(i in self.node[node]['prev_tile_space']
                       for i in self.node[c_node]['prev_tile_space']):
                    self.add_edge(c_node, node, 'prev', 'prev')
                if any(i in self.node[node]['prev_tile_space']
                       for i in self.node[c_node]['post_tile_space']):
                    self.add_edge(c_node, node, 'post', 'prev')
                if any(i in self.node[node]['post_tile_space']
                       for i in self.node[c_node]['prev_tile_space']):
                    self.add_edge(c_node, node, 'prev', 'post')
                if any(i in self.node[node]['post_tile_space']
                       for i in self.node[c_node]['post_tile_space']):
                    self.add_edge(c_node, node, 'post', 'post')

        self.m.objective = mip.minimize(self.tile_cost)

    def add_edge(self, c_node, node, cnode_edge, node_edge):

        shape_1 = len(self.node[c_node][cnode_edge])
        c_node_mip_var_key = 'prev_mip_var' if cnode_edge == 'prev' else 'post_mip_var'
        c_node_mip_var = self.node[c_node][c_node_mip_var_key]
        shape_2 = len(self.node[node][node_edge])
        node_mip_var_key = 'prev_mip_var' if node_edge == 'prev' else 'post_mip_var'
        node_mip_var = self.node[node][node_mip_var_key]

        mip_matrix = [[self.m.add_var(var_type=mip.BINARY) for _ in range(shape_2)]
                      for _ in range(shape_1)]

        for i in range(shape_1):
            for j in range(shape_2):
                self.m += mip_matrix[i][j] <= c_node_mip_var[i]
                self.m += mip_matrix[i][j] <= node_mip_var[j]
                self.m += mip_matrix[i][j] >= c_node_mip_var[i] + node_mip_var[j] - 1

        cost_matrix = self.get_edge_cost_matrix(self.node[c_node][cnode_edge],
                                                self.node[node][node_edge])

        self.tile_cost = self.tile_cost + mip.xsum(mip_matrix[i][j] * cost_matrix[i][j]
                                                   for i in range(shape_1) for j in range(shape_2))

    def get_cost_matrix(self, node, node_prev_strategy, node_post_strategy,
                        cnode_axis_prev_strategy, cnode_axis_post_strategy):
        runtime_ms = node.ed_info.runtime_ms
        cost_matrix_1 = [[runtime_ms for _ in range(len(node_post_strategy))]
                         for _ in range(len(node_prev_strategy))]
        cost_matrix_2 = [[runtime_ms for _ in range(len(node_post_strategy))]
                         for _ in range(len(node_prev_strategy))]
        cost_matrix_3 = [[runtime_ms for _ in range(len(node_post_strategy))]
                         for _ in range(len(node_prev_strategy))]
        for i in range(len(node_prev_strategy)):
            for j in range(len(node_post_strategy)):
                # reduce the cost when only tile prev or post, or same tile axis for prev/post
                if len(node_prev_strategy[i]) == 0 or len(
                        node_post_strategy[j]
                ) == 0 or cnode_axis_prev_strategy[i] == cnode_axis_post_strategy[j]:
                    for tiled_node in node_prev_strategy[i]:
                        num_tiles = tiled_node.num_tiles
                        cost_matrix_1[i][j] -= (1 + mdconfig.nvlink_processor_usage) * 2 * (
                            num_tiles - 1) / num_tiles * tiled_node.node.ed_info.runtime_ms
                        cost_matrix_2[i][j] -= (1 + mdconfig.nvlink_processor_usage) * (
                            num_tiles - 1) * tiled_node.node.ed_info.runtime_ms
                    for tiled_node in node_post_strategy[j]:
                        num_tiles = tiled_node.num_tiles
                        cost_matrix_1[i][j] -= (1 + mdconfig.nvlink_processor_usage) * 2 * (
                            num_tiles - 1) / num_tiles * tiled_node.node.ed_info.runtime_ms
                        cost_matrix_3[i][j] -= (1 + mdconfig.nvlink_processor_usage) * (
                            num_tiles - 1) * tiled_node.node.ed_info.runtime_ms

        for i in range(len(node_prev_strategy)):
            for j in range(len(node_post_strategy)):
                cost_matrix_1[i][j] = max(cost_matrix_1[i][j], cost_matrix_2[i][j],
                                          cost_matrix_3[i][j], 0)

        return cost_matrix_1

    def get_edge_cost_matrix(self, cnode_edge, node_edge):
        cost_matrix = [[0 for _ in range(len(node_edge))] for _ in range(len(cnode_edge))]

        for i in range(len(cnode_edge)):
            for j in range(len(node_edge)):
                for cnode_tiled_node in cnode_edge[i]:
                    for node_tiled_node in node_edge[j]:
                        if cnode_tiled_node.node.name == node_tiled_node.node.name:
                            num_tiles = cnode_tiled_node.num_tiles
                            if cnode_tiled_node.tile_axis == node_tiled_node.tile_axis:
                                cost_matrix[i][j] += (mdconfig.nvlink_processor_usage + 1) * (
                                    num_tiles -
                                    2) / num_tiles * cnode_tiled_node.node.ed_info.runtime_ms
                            else:
                                cost_matrix[i][j] += (mdconfig.nvlink_processor_usage + 1) * (
                                    num_tiles -
                                    1) / num_tiles * cnode_tiled_node.node.ed_info.runtime_ms

        return cost_matrix

    def solve(self):
        self.m.verbose = 0
        status = self.m.optimize()
        logger.info(f'[TileSolver.status]:\t {status}')
        logger.info(f'[TileSolver.solution_cost]:\t {self.m.objective_value}')

        final_strategy = {}

        for c_node in self.node:

            prev_strtg_idx = [mip_var.x for mip_var in self.node[c_node]['prev_mip_var']].index(1)
            post_strtg_idx = [mip_var.x for mip_var in self.node[c_node]['post_mip_var']].index(1)

            prev_tile_node = self.node[c_node]['prev'][prev_strtg_idx]
            post_tile_node = self.node[c_node]['post'][post_strtg_idx]

            if len(prev_tile_node + post_tile_node) == 0:
                continue

            tile_axis = (prev_tile_node + post_tile_node)[0].tile_axis
            num_tiles = (prev_tile_node + post_tile_node)[0].num_tiles

            c_end_node = list(c_node.users.keys())[0]
            tiled_comm_node = [
                TiledNode(c_node, tile_axis, num_tiles),
                TiledNode(c_end_node, tile_axis, num_tiles)
            ]
            final_strategy[c_node.name] = TileStrategy(num_tiles=2,
                                                       tms_comm=c_node.ed_info.runtime_ms,
                                                       tiled_comm_node=tiled_comm_node)

            for tiled_node in prev_tile_node:
                final_strategy[c_node.name].append_prev_node(tiled_node)

            for tiled_node in post_tile_node:
                final_strategy[c_node.name].append_post_node(tiled_node)

        # for broadcast, cannot contain meta tensor
        for key in final_strategy:
            for i in range(len(final_strategy[key].tiled_comm_node)):
                final_strategy[key].tiled_comm_node[i].node = final_strategy[key].tiled_comm_node[
                    i].node.name
            for i in range(len(final_strategy[key].tiled_prev_node)):
                final_strategy[key].tiled_prev_node[i].node = final_strategy[key].tiled_prev_node[
                    i].node.name
            for i in range(len(final_strategy[key].tiled_post_node)):
                final_strategy[key].tiled_post_node[i].node = final_strategy[key].tiled_post_node[
                    i].node.name

        return final_strategy


def override_view_op_args(tiled_node, node, tile_axis, num_tiles, tile_id):
    out_shape = node.meta['val'].shape

    tiled_out_shape = [i for i in out_shape]
    tile_size = (tiled_out_shape[tile_axis] + num_tiles - 1) // num_tiles

    if tile_id == num_tiles - 1:
        tiled_out_shape[tile_axis] = tiled_out_shape[tile_axis] - tile_size * (num_tiles - 1)
    else:
        tiled_out_shape[tile_axis] = tile_size

    shape_argnum = 1
    tiled_node.update_arg(shape_argnum, tiled_out_shape)

    return tiled_node


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

    # solve tile strategy on the rank 0
    final_strategy = {}
    if torch.distributed.get_rank() == 0:
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
                prev_nodes = set(find_all_prev_nodes(node))
                post_nodes = set(find_all_post_nodes(node))
                independent_nodes = set(computation_nodes) - \
                    prev_nodes - post_nodes

                independent_nodes_runtime = 0.0
                for indpd_node in independent_nodes:
                    independent_nodes_runtime += indpd_node.ed_info.runtime_ms

                node_info = f"[{node.name}] pre={len(prev_nodes)} post={len(post_nodes)} indpd={len(independent_nodes)}"

                logger.debug(
                    f"{node_info} runtime={node.ed_info.runtime_ms:.4f}/{independent_nodes_runtime:.4f}"
                )

                # TODO need to extend if independent_nodes can not cover the communication
                if independent_nodes_runtime <= node.ed_info.runtime_ms:
                    critical_communication[node] = None

        logger.info(f"Number of critical communication nodes: {len(critical_communication)}")

        # (TODO) here we assume the number of tiles is 2
        num_tiles = 2

        all_tile_strategies = {}

        # Step 3: determine the strategy for each critical communication node
        for c_node in critical_communication:
            if c_node.target in [all_gather_start, all_reduce_start]:
                comm_tensor_shape = c_node.prev.meta['val'].shape
                tile_dim_list = list(range(len(comm_tensor_shape)))

                if c_node.target == all_gather_start:
                    gather_dim = c_node.args[1]
                    del tile_dim_list[gather_dim]

                tms_comm = c_node.ed_info.runtime_ms
                # (TODO) nvlink and ib have different processor_usage
                min_comp_needed = tms_comm / (1 + mdconfig.nvlink_processor_usage) / (num_tiles -
                                                                                      1)
                max_comp_needed = 0.5 * min_comp_needed * num_tiles - min_comp_needed
                max_comp_needed = max(max_comp_needed, min_comp_needed)

                c_end_node = list(c_node.users.keys())[0]

                # explore all the possible tile axis
                for tile_dim in tile_dim_list:
                    node_tile_strategy = TileStrategy(num_tiles=num_tiles,
                                                      tms_comm=tms_comm,
                                                      tiled_comm_node=[
                                                          TiledNode(c_node, tile_dim, num_tiles),
                                                          TiledNode(c_end_node, tile_dim,
                                                                    num_tiles)
                                                      ])

                    # BFS to find the previous computation nodes that can be tiled
                    prev_node_queue = [TiledNode(c_node, tile_dim, num_tiles)]
                    while len(prev_node_queue
                              ) > 0 and node_tile_strategy.tms_prev_comp < max_comp_needed:
                        tiled_node = prev_node_queue.pop(0)
                        for arg in tiled_node.node.args:
                            if isinstance(arg, torch.fx.Node):
                                if arg.op == 'placeholder':
                                    continue
                                if arg.ed_info.is_computation():

                                    # backward the tile axis to the previous computation node
                                    tiled_arg = backward_tiled_node(tiled_node, arg)
                                    if tiled_arg is not None:
                                        node_tile_strategy.append_prev_node(tiled_arg)
                                        prev_node_queue.append(tiled_arg)
                                        logger.debug(tiled_arg, node_tile_strategy.tms_prev_comp)

                            if node_tile_strategy.tms_prev_comp >= max_comp_needed:
                                break

                    # BFS to find the post computation nodes that can be tiled
                    post_node_queue = [TiledNode(c_end_node, tile_dim, num_tiles)]
                    while len(post_node_queue
                              ) > 0 and node_tile_strategy.tms_post_comp < max_comp_needed:
                        tiled_node = post_node_queue.pop(0)
                        for user in tiled_node.node.users:
                            if user.op == 'output':
                                continue
                            if user.ed_info.is_computation():

                                # forward the tile axis to the post computation node
                                tiled_user = forward_tiled_node(tiled_node, user)

                                if tiled_user is not None:
                                    node_tile_strategy.append_post_node(tiled_user)
                                    post_node_queue.append(tiled_user)
                                    logger.debug(tiled_user, node_tile_strategy.tms_post_comp)

                            if node_tile_strategy.tms_post_comp >= max_comp_needed:
                                break

                    if not c_node in all_tile_strategies:
                        all_tile_strategies[c_node] = []
                    all_tile_strategies[c_node].append(node_tile_strategy)

        # Step 4: solve the tile strategy with mip
        if len(all_tile_strategies) > 0:
            final_strategy = TileSolver(all_tile_strategies).solve()

    # broadcast the tile strategy to all the ranks
    broadcast_result = [final_strategy]
    torch.distributed.broadcast_object_list(broadcast_result, src=0, device="cuda")
    final_strategy = broadcast_result[0]
    torch.distributed.barrier()

    if mdconfig.log_level <= logging.DEBUG:
        rich.print(final_strategy)

    # Step 5: tile the critical communication nodes and the computation context
    fx_module = tile_transform(fx_module, final_strategy)

    return fx_module


def tile_transform(fx_module: torch.fx.GraphModule,
                   final_strategy: Dict[str, TileStrategy]) -> torch.fx.GraphModule:

    to_tiled_node_info = {}
    tiled_node_info = {}

    for node_name in final_strategy:
        tiled_nodes = final_strategy[node_name].tiled_prev_node + final_strategy[
            node_name].tiled_comm_node + final_strategy[node_name].tiled_post_node
        for t_node in tiled_nodes:
            if t_node.node not in to_tiled_node_info:
                to_tiled_node_info[t_node.node] = t_node

    for node in fx_module.graph.nodes:
        if node.name in to_tiled_node_info:
            num_tiles = to_tiled_node_info[node.name].num_tiles
            tile_axis = to_tiled_node_info[node.name].tile_axis

            with fx_module.graph.inserting_before(node):
                flatten_args, _ = pytree.tree_flatten(node.args)
                flatten_tensor_args = [i for i in flatten_args if isinstance(i, torch.fx.Node)]
                flatten_tiled_args = []
                for arg_index, arg_node in enumerate(flatten_tensor_args):
                    if arg_node.name in tiled_node_info and tile_axis == tiled_node_info[
                            arg_node.name]['tile_axis']:
                        tiled_arg = tiled_node_info[arg_node.name]["tiled_output_node"]
                    else:
                        tiled_arg = []
                        arg_tile_index = get_arg_tile_axis(node,
                                                           to_tiled_node_info[node.name].tile_axis,
                                                           arg_index)
                        if arg_tile_index is None:
                            tiled_arg = [arg_node] * num_tiles
                        else:
                            chunk_arg_node = fx_module.graph.call_function(torch.ops.aten.chunk,
                                                                           args=(arg_node,
                                                                                 num_tiles,
                                                                                 arg_tile_index))
                            chunk_arg_node.ed_info = EDInfo()
                            chunk_arg_node.ed_info.node_type = EDNodeType.COMPUTATION
                            chunk_arg_node.meta = create_meta_from_node(chunk_arg_node)
                            for tile_id in range(num_tiles):
                                tiled_arg.append(
                                    fx_module.graph.call_function(operator.getitem,
                                                                  args=(chunk_arg_node, tile_id)))
                                tiled_arg[-1].ed_info = EDInfo()
                                tiled_arg[-1].ed_info.node_type = EDNodeType.AUXILIARY
                                tiled_arg[-1].meta = create_meta_from_node(tiled_arg[-1])
                    flatten_tiled_args.append(tiled_arg)

                # split the node
                tiled_nodes = []
                for tile_id in range(num_tiles):
                    flatten_args, specs = pytree.tree_flatten(node.args)
                    flatten_args_idx = 0
                    for arg_index, arg_node in enumerate(flatten_args):
                        if isinstance(arg_node, torch.fx.Node):
                            flatten_args[arg_index] = flatten_tiled_args[flatten_args_idx][tile_id]
                            flatten_args_idx += 1
                    tiled_args = pytree.tree_unflatten(flatten_args, specs)
                    tiled_node = fx_module.graph.call_function(node.target, args=tiled_args)
                    if node.target in view_op_map:
                        tiled_node = override_view_op_args(tiled_node, node, tile_axis, num_tiles,
                                                           tile_id)
                    tiled_node.ed_info = EDInfo()
                    tiled_node.ed_info.node_type = EDNodeType.COMPUTATION
                    tiled_node.meta = create_meta_from_node(tiled_node)
                    tiled_nodes.append(tiled_node)

                concat_node = fx_module.graph.call_function(torch.ops.aten.concat,
                                                            args=(tiled_nodes, tile_axis))
                concat_node.ed_info = EDInfo()
                concat_node.ed_info.node_type = EDNodeType.COMPUTATION
                concat_node.meta = create_meta_from_node(concat_node)

            with fx_module.graph.inserting_after(node):
                node.replace_all_uses_with(concat_node)

            # get tiled_output_node and concat_output_node

            tiled_node_info[node.name] = {
                "tile_axis": tile_axis,
                "num_tiles": num_tiles,
                "tiled_output_node": tiled_nodes,
                "concat_output_node": concat_node,
            }

            tiled_node_info[concat_node.name] = tiled_node_info[node.name]

    fx_module.graph.eliminate_dead_code()
    fx_module.recompile()

    if mdconfig.log_level <= logging.DEBUG:
        fx_module.print_readable()

    return fx_module
