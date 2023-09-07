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
import random
from pprint import pprint
from typing import List

import metadist.config as mdconfig
import mip
from metadist.metashard.metair import (SPMD, MetaGraph, MetaNode, MetaNodeCluster, MetaVar,
                                       NodeSPMDStrategy, VarSPMDStrategy, VarSPMDStrategyGroup,
                                       ClusterStrategyPool)

logger = logging.getLogger(__name__)


def shuffle_list(*ls):
    l = list(zip(*ls))

    random.shuffle(l)
    return zip(*l)


def get_idx_in_var_list(var: MetaVar, var_list: List[MetaVar]):
    var_list = [v.name if v else None for v in var_list]
    if var.name in var_list:
        return var_list.index(var.name)
    return None


def calculate_resharding_cost(var: MetaVar, strategy_in: VarSPMDStrategy,
                              strategy_out: VarSPMDStrategy, device_mesh):
    var_size = var.get_var_size()

    all_gather = lambda x, ndevices: x * (ndevices - 1) / ndevices
    all_reduce = lambda x, ndevices: 2 * x * (ndevices - 1) / ndevices

    def all_to_all(x, ndevices):
        factor = 1.
        if ndevices > 2:
            factor = mdconfig.all_to_all_punish_factor
        return factor * x * (ndevices - 1) / ndevices / ndevices

    resharding_cost = 0

    s1_in, s2_in = strategy_in[0], strategy_in[1]
    s1_out, s2_out = strategy_out[0], strategy_out[1]

    if device_mesh[0] > 1:
        message_size = var_size
        if s2_in.is_shard():
            message_size /= device_mesh[1]
        if s1_in.is_shard():
            if s1_out.is_shard():
                if s1_in.args != s1_out.args:
                    resharding_cost += all_to_all(message_size, device_mesh[0])
            else:
                resharding_cost += all_gather(message_size, device_mesh[0])
        elif s1_in.is_partial():
            resharding_cost += all_reduce(message_size, device_mesh[0])

    if device_mesh[1] > 1:
        message_size = var_size
        if s1_in.is_shard():
            message_size /= device_mesh[0]
        if s2_in.is_shard():
            if s2_out.is_shard():
                if s2_in.args != s2_out.args:
                    resharding_cost += all_to_all(message_size, device_mesh[1])
            else:
                resharding_cost += all_gather(message_size, device_mesh[1])
        elif s2_in.is_partial():
            resharding_cost += all_reduce(message_size, device_mesh[1])

    return resharding_cost


def calculate_memory_cost(var: MetaVar, strategy_in: VarSPMDStrategy,
                          strategy_out: VarSPMDStrategy, device_mesh):
    var_size = var.get_var_size()

    memory_cost = 0

    # FIXME: only use out_strategy here for shard_size is ok?
    for strategy in [strategy_in, strategy_out]:
        s1, s2 = strategy[0], strategy[1]
        shard_size = 1
        if s1.is_shard():
            shard_size *= device_mesh[0]
        if s2.is_shard():
            shard_size *= device_mesh[1]

        memory_cost += var_size // shard_size

    return memory_cost


def gen_comm_cost_matrix(
        var: MetaVar,
        up_strategy,  # list of VarSPMDStrategy
        down_strategy,  # list of VarSPMDStrategy
        device_mesh):
    comm_matrix = [[0 for _ in range(len(down_strategy))] for _ in range(len(up_strategy))]

    for i in range(len(up_strategy)):
        for j in range(len(down_strategy)):
            var_up_strategy = up_strategy[i]
            var_down_strategy = down_strategy[j]
            comm_matrix[i][j] = calculate_resharding_cost(var, var_up_strategy, var_down_strategy,
                                                          device_mesh)

    return comm_matrix


# Lansong(TODO) generate_comm_matrix is deprecated, use gen_comm_cost_matrix instead
def generate_comm_matrix(
        var: MetaVar,
        up_strategy,  # list of NodeSPMDStrategy
        down_strategy,  # list of NodeSPMDStrategy
        idx_for_up,
        idx_for_down,
        device_mesh):
    comm_matrix = [[0 for _ in range(len(down_strategy))] for _ in range(len(up_strategy))]

    for i in range(len(up_strategy)):
        for j in range(len(down_strategy)):
            var_up_strategy = up_strategy[i].out_strtg_group[idx_for_up]
            var_down_strategy = down_strategy[j].in_strtg_group[idx_for_down]
            comm_matrix[i][j] = calculate_resharding_cost(var, var_up_strategy, var_down_strategy,
                                                          device_mesh)

    return comm_matrix


def gen_mem_cost_matrix(var: MetaVar, up_strategy, down_strategy, device_mesh):
    comm_matrix = [[0 for _ in range(len(down_strategy))] for _ in range(len(up_strategy))]

    for i in range(len(up_strategy)):
        for j in range(len(down_strategy)):
            var_up_strategy = up_strategy[i]
            var_down_strategy = down_strategy[j]
            comm_matrix[i][j] = calculate_memory_cost(var, var_up_strategy, var_down_strategy,
                                                      device_mesh)

    return comm_matrix


# Lansong(TODO) generate_mem_matrix is deprecated, use gen_mem_cost_matrix instead
def generate_mem_matrix(var: MetaVar, up_strategy, down_strategy, idx_for_up, idx_for_down,
                        device_mesh):
    comm_matrix = [[0 for _ in range(len(down_strategy))] for _ in range(len(up_strategy))]

    for i in range(len(up_strategy)):
        for j in range(len(down_strategy)):
            var_up_strategy = up_strategy[i].out_strtg_group[idx_for_up]
            var_down_strategy = down_strategy[j].in_strtg_group[idx_for_down]
            comm_matrix[i][j] = calculate_memory_cost(var, var_up_strategy, var_down_strategy,
                                                      device_mesh)

    return comm_matrix


class ClusterMipInfo:

    def __init__(self, cluster: MetaNodeCluster, cluster_strtg_pool: ClusterStrategyPool,
                 mip_var: list):
        self.cluster = cluster
        self.cluster_strtg_pool = cluster_strtg_pool
        self.mip_var = mip_var

    def get_node_strtg(self, nd_id: int, strtg_idx: int) -> NodeSPMDStrategy:
        return self.cluster_strtg_pool.get_node_strtg(nd_id, strtg_idx)

    def __str__(self) -> str:
        res = str(self.cluster)
        res += "\n" + str(self.cluster_strtg_pool)
        return res

    def __repr__(self) -> str:
        return self.__str__()


class ClusterEdgeMipInfo:

    def __init__(self, edge: MetaVar):
        self.edge = edge
        self.up_node = None
        self.down_nodes = []
        self.idx_for_up = None
        self.idx_for_down = []
        self.comm_matrix = []
        self.mem_matrix = []
        self.mip_var = []

    def __str__(self) -> str:
        res = str(self.edge)
        return res

    def __repr__(self) -> str:
        return self.__str__()


class AutoFlowSolver:

    def __init__(self,
                 device_mesh=None,
                 constraints=None,
                 memory_ratio=0.9,
                 total_memery=None) -> None:
        self.m = mip.Model("autoflow")
        self.nodes = {}
        self.edges = {}
        self.device_mesh = device_mesh
        self.constraints = constraints

        self.memory_ratio = memory_ratio
        self.total_memery = 32 * 1024 * 1024 * 1024 if total_memery is None else total_memery
        self.max_memory_constrain = self.memory_ratio * self.total_memery

        self.clusters = {}
        self.cluster_edges = {}

    def add_coarsen_graph(self, graph: MetaGraph) -> None:
        self.graph = graph

        for cluster in graph.node_clusters:
            self.add_cluster(cluster)

        if mdconfig.log_level <= logging.DEBUG:
            for mip_info in self.clusters.values():
                print(str(mip_info))

    def add_cluster(self, cluster: MetaNodeCluster) -> None:
        cluster_id = cluster.unique_id
        cluster_strtg_pool = cluster.get_strtg_pool()
        cluster_strtg_num = cluster_strtg_pool.get_strtg_num()
        if cluster_strtg_num > 0:
            mip_vars = [self.m.add_var(var_type=mip.BINARY) for _ in range(cluster_strtg_num)]
            mip_info = ClusterMipInfo(cluster, cluster_strtg_pool, mip_vars)
            self.clusters[cluster_id] = mip_info

            # Lansong 2do
            # add edges from user side
            for input_node, input_idx in cluster.args.descs:
                if input_node.op_name == "placeholder" or input_node.op_name == "get_attr":
                    # placeholder or get_attr for a stateful variable has an implicit input
                    if input_node in self.graph.state_io_map:
                        output_var = self.graph.state_io_map[input_node]

                        # output var is the implicit input var of placeholder and get_attr
                        self.add_cluster_edge(output_var, input_idx, down_node=input_node)
                else:
                    var = input_node.invars[input_idx]
                    self.add_cluster_edge(var, input_idx, down_node=input_node)

            assert cluster.output_node
            for out_idx, var in enumerate(cluster.output_node.outvars):
                if var:
                    self.add_cluster_edge(var, out_idx, up_node=cluster.output_node)

    # Lansong(TODO) add_graph is deprecated, use add_coarsen_graph instead
    def add_graph(self, graph: MetaGraph) -> None:
        self.graph = graph

        if self.constraints:
            for cons in self.constraints:
                if cons.name not in [i.name for i in self.graph.input_list]:
                    continue
                cons_node = MetaNode("cons_" + cons.name, [], [cons], None)
                outvars_strategy = VarSPMDStrategyGroup(VarSPMDStrategy(self.constraints[cons]))
                self.nodes[cons_node.unique_key()] = {
                    "node": cons_node,
                    "strategy": NodeSPMDStrategy(None, outvars_strategy),
                    "mip_var": [self.m.add_var(var_type=mip.BINARY)]
                }
                self.add_edge(cons, up_node=cons_node)

        self.liveness = graph.liveness()
        if mdconfig.liveness_only_input:
            self.liveness = self.liveness[:1]

        for op in graph.op_list:
            self.add_node(op)

    # Lansong(TODO) add_node is deprecated, use add_cluster instead
    def add_node(self, node: MetaNode) -> None:
        unique_key_ = node.unique_key()

        strategies = node.get_strtg_pool().strategies
        if len(strategies) > 0:
            self.nodes[unique_key_] = {
                "node": node,
                "strategy": strategies,
                "mip_var": [self.m.add_var(var_type=mip.BINARY) for _ in range(len(strategies))]
            }

            for var in node.invars:
                self.add_edge(var, down_node=node)

            for var in node.outvars:
                if var:
                    self.add_edge(var, up_node=node)

    def add_cluster_edge(self, edge: MetaVar, io_idx: int, up_node=None, down_node=None) -> None:
        unique_key_ = edge.name

        if unique_key_ not in self.cluster_edges:
            edge_mip_info = ClusterEdgeMipInfo(edge=edge)
            self.cluster_edges[unique_key_] = edge_mip_info

        if up_node is not None:
            self.cluster_edges[unique_key_].up_node = up_node
            self.cluster_edges[unique_key_].idx_for_up = io_idx

        if down_node is not None:
            self.cluster_edges[unique_key_].down_nodes.append(down_node)
            self.cluster_edges[unique_key_].idx_for_down.append(io_idx)


        def _add_cluster_edge(up_node, idx_for_up, down_node, idx_for_down):
            up_cluster_strtg_pool = self.clusters[up_node.cluster_id].cluster_strtg_pool
            up_out_strategy_list = up_cluster_strtg_pool.get_outvar_strtg_list(
                up_node.unique_id, idx_for_up)

            down_cluster_strtg_pool = self.clusters[down_node.cluster_id].cluster_strtg_pool
            down_in_strategy_list = down_cluster_strtg_pool.get_invar_strtg_list(
                down_node.unique_id, idx_for_down)

            self.cluster_edges[unique_key_].mip_var.append([[
                self.m.add_var(var_type=mip.BINARY) for _ in range(len(down_in_strategy_list))
            ] for _ in range(len(up_out_strategy_list))])

            # calculate ``comm_matrix`` for this edge
            idx_for_up = self.cluster_edges[unique_key_].idx_for_up
            idx_for_down = self.cluster_edges[unique_key_].idx_for_down[-1]

            self.cluster_edges[unique_key_].comm_matrix.append(
                gen_comm_cost_matrix(
                    self.cluster_edges[unique_key_].edge,
                    up_out_strategy_list,
                    down_in_strategy_list,
                    self.device_mesh,
                ))

            self.cluster_edges[unique_key_].mem_matrix.append(
                gen_mem_cost_matrix(
                    self.cluster_edges[unique_key_].edge,
                    up_out_strategy_list,
                    down_in_strategy_list,
                    self.device_mesh,
                ))

        # if adding an up_node forms an edge
        if up_node and self.cluster_edges[unique_key_].down_nodes:
            for down_node, idx_for_down in zip(self.cluster_edges[unique_key_].down_nodes, 
                                               self.cluster_edges[unique_key_].idx_for_down):
                _add_cluster_edge(up_node, io_idx, down_node, idx_for_down)
        # if adding a down_node forms an edge
        elif self.cluster_edges[unique_key_].up_node and down_node:
            _add_cluster_edge(self.cluster_edges[unique_key_].up_node,
                              self.cluster_edges[unique_key_].idx_for_up,
                              down_node, io_idx)


    # Lansong(TODO) add_edge is deprecated, use add_cluster_edge instead
    def add_edge(self, edge: MetaVar, up_node=None, down_node=None) -> None:
        unique_key_ = edge.name

        if unique_key_ not in self.edges:
            self.edges[unique_key_] = {
                "edge": edge,
                "up_node": None,
                "down_node": [],
                "idx_for_up": None,
                "idx_for_down": [],
                "comm_matrix": [],
                "mem_matrix": [],
                "mip_var": [],
            }
        if up_node is not None:
            self.edges[unique_key_]["up_node"] = up_node.unique_key()
            self.edges[unique_key_]["idx_for_up"] = get_idx_in_var_list(edge, up_node.outvars)

        if down_node is not None:
            self.edges[unique_key_]["down_node"].append(down_node.unique_key())
            self.edges[unique_key_]["idx_for_down"].append(
                get_idx_in_var_list(edge, down_node.invars))

            if self.edges[unique_key_]["up_node"] is not None:

                up_node_key = self.edges[unique_key_]["up_node"]
                up_strategy = self.nodes[up_node_key]["strategy"]

                down_node_key = self.edges[unique_key_]["down_node"][-1]
                down_strategy = self.nodes[down_node_key]["strategy"]

                self.edges[unique_key_]["mip_var"].append(
                    [[self.m.add_var(var_type=mip.BINARY) for _ in range(len(down_strategy))]
                     for _ in range(len(up_strategy))])

                # calculate ``comm_matrix`` for this edge
                idx_for_up = self.edges[unique_key_]["idx_for_up"]
                idx_for_down = self.edges[unique_key_]["idx_for_down"][-1]

                self.edges[unique_key_]["comm_matrix"].append(
                    generate_comm_matrix(
                        self.edges[unique_key_]["edge"],
                        up_strategy,  # list of NodeSPMDStrategy
                        down_strategy,  # list of NodeSPMDStrategy
                        idx_for_up,
                        idx_for_down,
                        self.device_mesh,
                    ))

                self.edges[unique_key_]["mem_matrix"].append(
                    generate_mem_matrix(
                        self.edges[unique_key_]["edge"],
                        up_strategy,  # list of NodeSPMDStrategy
                        down_strategy,  # list of NodeSPMDStrategy
                        idx_for_up,
                        idx_for_down,
                        self.device_mesh,
                    ))

    # Lansong(TODO) ilp_optimize is deprecated, use ilp_solve instead
    def ilp_optimize(self, count_invars=False):
        comm_cost, mem_cost = 0, 0
        for edge in self.edges.values():
            for idx in range(len(edge["mip_var"])):
                mip_var = edge["mip_var"][idx]
                comm_matrix = edge["comm_matrix"][idx]
                mem_matrix = edge["mem_matrix"][idx]
                shape_1 = len(mip_var)
                shape_2 = len(mip_var[0])
                comm_cost = comm_cost + mip.xsum(mip_var[i][j] * comm_matrix[i][j]
                                                 for i in range(shape_1) for j in range(shape_2))
                mem_cost = mem_cost + mip.xsum(mip_var[i][j] * mem_matrix[i][j]
                                               for i in range(shape_1) for j in range(shape_2))

            def _mem_cost(var_size, down_strategy, idx_for_down):
                memory_cost_list = []
                for i in range(len(down_strategy)):
                    strategy = down_strategy[i].in_strtg_group[idx_for_down]

                    # FIXME: only use out_strategy here for shard_size is ok?
                    s1, s2 = strategy[0], strategy[1]
                    shard_size = 1
                    if s1.state == SPMD.SHARD:
                        shard_size *= self.device_mesh[0]
                    if s2.state == SPMD.SHARD:
                        shard_size *= self.device_mesh[1]

                    memory_cost_list.append(var_size // shard_size)

                return memory_cost_list

            if count_invars and edge["up_node"] is None:
                var_size = edge["edge"].get_var_size()
                for down_node_key, idx_for_down in zip(edge["down_node"], edge["idx_for_down"]):
                    down_strategy = self.nodes[down_node_key]["strategy"]
                    __mem_cost = _mem_cost(var_size, down_strategy, idx_for_down)
                    down_node_mip_var = self.nodes[down_node_key]["mip_var"]
                    mem_cost = mem_cost + mip.xsum(down_node_mip_var[i] * __mem_cost[i]
                                                   for i in range(len(down_node_mip_var)))

        for edge in self.edges.values():
            for idx in range(len(edge["mip_var"])):
                mip_var = edge["mip_var"][idx]
                shape_1 = len(mip_var)
                shape_2 = len(mip_var[0])
                self.m += mip.xsum(mip_var[i][j] for i in range(shape_1)
                                   for j in range(shape_2)) == 1

                up_node_key = edge["up_node"]
                up_node_mip_var = self.nodes[up_node_key]["mip_var"]

                down_node_key = edge["down_node"][idx]
                down_node_mip_var = self.nodes[down_node_key]["mip_var"]

                for i in range(shape_1):
                    for j in range(shape_2):
                        self.m += mip_var[i][j] <= up_node_mip_var[i]
                        self.m += mip_var[i][j] <= down_node_mip_var[j]
                        self.m += mip_var[i][j] >= up_node_mip_var[i] + down_node_mip_var[j] - 1

        idx_record = {}
        for idx_ in range(len(self.liveness)):
            mem_live = self.liveness[idx_]
            op_ = self.graph.op_list[idx_]

            mip_var_list = []
            mem_matrix_list = []
            for tensor_name in mem_live:
                if tensor_name not in self.edges:
                    continue
                if tensor_name not in idx_record:
                    idx_record[tensor_name] = 0
                if len(self.edges[tensor_name]["mip_var"]) > 0:
                    edge_idx = min(
                        len(self.edges[tensor_name]["mip_var"]) - 1, idx_record[tensor_name])
                    mip_var_list.append(self.edges[tensor_name]["mip_var"][edge_idx])
                    mem_matrix_list.append(self.edges[tensor_name]["mem_matrix"][edge_idx])
                if self.edges[tensor_name]["up_node"] is None:
                    down_node_key, idx_for_down = self.edges[tensor_name]["down_node"][
                        0], self.edges[tensor_name]["idx_for_down"][0]
                    var_size = self.edges[tensor_name]["edge"].get_var_size()
                    down_strategy = self.nodes[down_node_key]["strategy"]
                    __mem_cost = _mem_cost(var_size, down_strategy, idx_for_down)
                    mip_var_list.append([self.nodes[down_node_key]["mip_var"]])
                    mem_matrix_list.append([__mem_cost])

            for var in op_.invars:
                if var.name not in self.edges:
                    continue
                if var.name not in idx_record:
                    idx_record[var.name] = 0
                idx_record[var.name] += 1

            if len(mip_var_list) >= 1:
                need_sum = []
                for mip_var, mem_matrix in zip(mip_var_list, mem_matrix_list):
                    shape_1 = len(mip_var)
                    shape_2 = len(mip_var[0])
                    for i in range(shape_1):
                        for j in range(shape_2):
                            need_sum.append(mip_var[i][j] * mem_matrix[i][j])
                self.m += mip.xsum(i for i in need_sum) <= self.max_memory_constrain

        for node in self.nodes.values():
            mip_var = node["mip_var"]
            shape_1 = len(mip_var)
            self.m += mip.xsum(mip_var[i] for i in range(shape_1)) == 1

        self.m.objective = mip.minimize(comm_cost + 0.00000001 * mem_cost)

        self.m.verbose = 0
        status = self.m.optimize(max_seconds_same_incumbent=mdconfig.max_seconds_same_incumbent)
        logger.info(f'[AutoFlowSolver.status]:\t {status}')
        logger.info(f'[AutoFlowSolver.solution_cost]:\t {self.m.objective_value}')

        return self.get_optimal_stratey()

    # Lansong(TODO) get_optimal_stratey is deprecated
    def get_optimal_stratey(self):
        optimal_stratey = {}
        for unique_key_ in self.nodes:
            node = self.nodes[unique_key_]['node']
            opt_ = None
            strategy_list = self.nodes[unique_key_]['strategy']
            mip_var = self.nodes[unique_key_]['mip_var']
            assert len(strategy_list) == len(mip_var)
            for s_, mip_var_s in zip(strategy_list, mip_var):
                if mip_var_s.x == 1:
                    opt_ = s_
            optimal_stratey[unique_key_] = {'node': node, 'strategy': opt_}

        if mdconfig.log_level <= logging.DEBUG:
            print("optimal_stratey:")
            pprint(optimal_stratey)

        return optimal_stratey

    def ilp_solve(self, count_invars=False):
        comm_cost, mem_cost = 0, 0
        for edge_mip_info in self.cluster_edges.values():
            for idx in range(len(edge_mip_info.mip_var)):
                mip_var = edge_mip_info.mip_var[idx]
                comm_matrix = edge_mip_info.comm_matrix[idx]
                mem_matrix = edge_mip_info.mem_matrix[idx]
                shape_1 = len(mip_var)
                shape_2 = len(mip_var[0])
                comm_cost = comm_cost + mip.xsum(mip_var[i][j] * comm_matrix[i][j]
                                                 for i in range(shape_1) for j in range(shape_2))
                mem_cost = mem_cost + mip.xsum(mip_var[i][j] * mem_matrix[i][j]
                                               for i in range(shape_1) for j in range(shape_2))

            def _mem_cost(var_size, down_strategy: 'list[VarSPMDStrategy]'):
                memory_cost_list = []
                for i in range(len(down_strategy)):
                    strategy = down_strategy[i]

                    # FIXME: only use out_strategy here for shard_size is ok?
                    s1, s2 = strategy[0], strategy[1]
                    shard_size = 1
                    if s1.state == SPMD.SHARD:
                        shard_size *= self.device_mesh[0]
                    if s2.state == SPMD.SHARD:
                        shard_size *= self.device_mesh[1]

                    memory_cost_list.append(var_size // shard_size)

                return memory_cost_list

            if count_invars and edge_mip_info.up_node is None:
                var_size = edge_mip_info.edge.get_var_size()
                for down_node, idx_for_down in zip(edge_mip_info.down_nodes,
                                                   edge_mip_info.idx_for_down):
                    down_cluster_mip_info = self.clusters[down_node.cluster_id]
                    down_cluster_strtg_pool = down_cluster_mip_info.cluster_mip_info
                    down_strategy = down_cluster_strtg_pool.get_invar_strtg_list(
                        down_node.unique_id, idx_for_down)
                    __mem_cost = _mem_cost(var_size, down_strategy)
                    down_cluster_mip_var = self.clusters[down_node.cluster_id].mip_var
                    mem_cost = mem_cost + mip.xsum(down_cluster_mip_var[i] * __mem_cost[i]
                                                   for i in range(len(down_cluster_mip_var)))

        for edge_mip_info in self.cluster_edges.values():
            for idx in range(len(edge_mip_info.mip_var)):
                mip_var = edge_mip_info.mip_var[idx]
                shape_1 = len(mip_var)
                shape_2 = len(mip_var[0])
                self.m += mip.xsum(mip_var[i][j] for i in range(shape_1)
                                   for j in range(shape_2)) == 1

                up_cluster_id = edge_mip_info.up_node.cluster_id
                up_cluster_mip_var = self.clusters[up_cluster_id].mip_var

                down_node = edge_mip_info.down_nodes[idx]
                down_cluster_mip_var = self.clusters[down_node.cluster_id].mip_var

                for i in range(shape_1):
                    for j in range(shape_2):
                        self.m += mip_var[i][j] <= up_cluster_mip_var[i]
                        self.m += mip_var[i][j] <= down_cluster_mip_var[j]
                        self.m += mip_var[i][
                            j] >= up_cluster_mip_var[i] + down_cluster_mip_var[j] - 1


#        Lansong(TODO) support memory limit
#        idx_record = {}
#        for idx_ in range(len(self.liveness)):
#            mem_live = self.liveness[idx_]
#            op_ = self.graph.op_list[idx_]
#
#            mip_var_list = []
#            mem_matrix_list = []
#            for tensor_name in mem_live:
#                if tensor_name not in self.cluster_edges:
#                    continue
#                if tensor_name not in idx_record:
#                    idx_record[tensor_name] = 0
#                if len(self.cluster_edges[tensor_name]["mip_var"]) > 0:
#                    edge_idx = min(
#                        len(self.cluster_edges[tensor_name]["mip_var"]) - 1, idx_record[tensor_name])
#                    mip_var_list.append(self.cluster_edges[tensor_name]["mip_var"][edge_idx])
#                    mem_matrix_list.append(self.cluster_edges[tensor_name]["mem_matrix"][edge_idx])
#                if self.cluster_edges[tensor_name]["up_node"] is None:
#                    down_node_key, idx_for_down = self.cluster_edges[tensor_name]["down_nodes"][
#                        0], self.cluster_edges[tensor_name]["idx_for_down"][0]
#                    var_size = self.cluster_edges[tensor_name]["edge"].get_var_size()
#                    down_strategy = self.clusters[down_node_key]["strategy"]
#                    __mem_cost = _mem_cost(var_size, down_strategy, idx_for_down)
#                    mip_var_list.append([self.clusters[down_node_key]["mip_var"]])
#                    mem_matrix_list.append([__mem_cost])
#
#            for var in op_.invars:
#                if var.name not in self.cluster_edges:
#                    continue
#                if var.name not in idx_record:
#                    idx_record[var.name] = 0
#                idx_record[var.name] += 1
#
#            if len(mip_var_list) >= 1:
#                need_sum = []
#                for mip_var, mem_matrix in zip(mip_var_list, mem_matrix_list):
#                    shape_1 = len(mip_var)
#                    shape_2 = len(mip_var[0])
#                    for i in range(shape_1):
#                        for j in range(shape_2):
#                            need_sum.append(mip_var[i][j] * mem_matrix[i][j])
#                self.m += mip.xsum(i for i in need_sum) <= self.max_memory_constrain

        for cluster_mip_info in self.clusters.values():
            mip_var = cluster_mip_info.mip_var
            shape_1 = len(mip_var)
            self.m += mip.xsum(
                mip_var[i]
                for i in range(shape_1)) == 1  # one and only one cluster strategy is active

        self.m.objective = mip.minimize(comm_cost + 0.00000001 * mem_cost)

        self.m.verbose = 0
        status = self.m.optimize(max_seconds_same_incumbent=mdconfig.max_seconds_same_incumbent)
        logger.info(f'[AutoFlowSolver.status]:\t {status}')
        logger.info(f'[AutoFlowSolver.solution_cost]:\t {self.m.objective_value}')

        return self.get_strategies()

    def get_strategies(self):
        optimal_strategies = {}

        for cluster_mip_info in self.clusters.values():
            cluster = cluster_mip_info.cluster
            opt_strtg_idx = -1
            for strtg_idx, mip_var_s in enumerate(cluster_mip_info.mip_var):
                if mip_var_s.x == 1:
                    opt_strtg_idx = strtg_idx
                    break
            assert opt_strtg_idx >= 0

            for node in cluster.nodes.values():
                nd_strtg = cluster_mip_info.get_node_strtg(node.unique_id, opt_strtg_idx)
                optimal_strategies[node.unique_key()] = {'node': node, 'strategy': nd_strtg}

        if mdconfig.log_level <= logging.DEBUG:
            print("optimal_strategies:")
            pprint(optimal_strategies)

        return optimal_strategies

    def beam_search(self, candidate_num=100):

        def get_new_cost(strategy, node, strategy_idx):
            cost = 0.
            edge_list = [self.edges[invar.name] for invar in node['node'].invars]
            for edge in edge_list:
                up_node_key = edge["up_node"]

                idx = edge["down_node"].index(node['node'].unique_key())
                if up_node_key in strategy:
                    up_node_strategy_idx = strategy[up_node_key]['strategy_idx']
                    mem_cost = edge["mem_matrix"][idx][up_node_strategy_idx][strategy_idx]
                    comm_cost = edge["comm_matrix"][idx][up_node_strategy_idx][strategy_idx]
                    cost += comm_cost

            return cost

        def add_candidate(strategy_candidate, accumulate_cost, node):
            new_strategy_candidate = []
            new_accumulate_cost = []
            key_ = node['node'].unique_key()
            if len(strategy_candidate) == 0:
                for idx in range(len(node['strategy'])):
                    stratey = {key_: {'node': node['node'], 'strategy_idx': idx}}
                    new_strategy_candidate.append(stratey)
                    new_accumulate_cost.append(0.)
            else:
                for idx, strategy in enumerate(strategy_candidate):
                    old_cost = accumulate_cost[idx]
                    for idx in range(len(node['strategy'])):
                        new_cost = get_new_cost(strategy, node, idx)
                        new_strategy = {key: strategy[key] for key in strategy}
                        new_strategy[key_] = {'node': node['node'], 'strategy_idx': idx}
                        new_strategy_candidate.append(new_strategy)
                        new_accumulate_cost.append(old_cost + new_cost)

            return new_strategy_candidate, new_accumulate_cost

        def select_candidate(strategy_candidate, accumulate_cost, candidate_num):
            assert len(strategy_candidate) == len(accumulate_cost)
            if len(accumulate_cost) <= candidate_num:
                return strategy_candidate, accumulate_cost

            accumulate_cost, strategy_candidate = shuffle_list(accumulate_cost, strategy_candidate)
            accumulate_cost, strategy_candidate = zip(
                *sorted(zip(accumulate_cost, strategy_candidate), key=lambda x: x[0]))

            return strategy_candidate[:candidate_num], accumulate_cost[:candidate_num]

        strategy_candidate = []
        accumulate_cost = []
        for unique_key_ in self.nodes:
            node = self.nodes[unique_key_]

            strategy_candidate, accumulate_cost = add_candidate(strategy_candidate,
                                                                accumulate_cost, node)

            strategy_candidate, accumulate_cost = select_candidate(strategy_candidate,
                                                                   accumulate_cost, candidate_num)

            accumulate_cost, strategy_candidate = zip(
                *sorted(zip(accumulate_cost, strategy_candidate), key=lambda x: x[0]))

        strategy = strategy_candidate[0]

        optimal_stratey = {}
        for key in strategy:
            node = strategy[key]['node']
            node_strategy_list = self.nodes[node.unique_key()]['strategy']
            optimal_stratey[key] = {
                "node": node,
                "strategy": node_strategy_list[strategy[key]['strategy_idx']]
            }

        logger.info(f'=========== solution cost:\t {accumulate_cost[0]}')

        return optimal_stratey
