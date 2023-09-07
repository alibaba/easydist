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

from __future__ import annotations
import copy
import os
import logging
from typing import List
from functools import reduce

from metadist.metashard.combination import ReduceOp
from metadist.platform import get_backend
import metadist.config as mdconfig

logger = logging.getLogger(__name__)

DEVICE_MESH_1D = -1


class SPMD:

    REPLICATE = "REPLICATE"
    SHARD = "SHARD"
    PARTIAL = "PARTIAL"

    def __init__(self, state, args=None) -> None:
        self.state = state
        self.args = args

    def is_shard(self) -> bool:
        return self.state == SPMD.SHARD

    def is_replicate(self) -> bool:
        return self.state == SPMD.REPLICATE

    def is_partial(self) -> bool:
        return self.state == SPMD.PARTIAL

    def __eq__(self, other) -> bool:
        if self.state == other.state and self.args == other.args:
            return True
        return False

    def __str__(self) -> str:
        str_ = f"{self.state}"
        if self.args:
            str_ += f"({self.args})"
        return str_

    def __repr__(self) -> str:
        return self.__str__()


class VarSPMDStrategy:

    def __init__(self, *var_spmd_strategy) -> None:
        self.var_spmd_strategy = list(var_spmd_strategy)

    def __getitem__(self, idx) -> SPMD:
        return self.var_spmd_strategy[idx]

    def __add__(self, other):
        return VarSPMDStrategy(*self.var_spmd_strategy, *other.var_spmd_strategy)

    def __eq__(self, other) -> bool:
        if len(self.var_spmd_strategy) == len(other.var_spmd_strategy):
            for i, j in zip(self.var_spmd_strategy, other.var_spmd_strategy):
                if i != j:
                    return False
            return True
        return False

    def __str__(self) -> str:
        return f"VarSPMDStrategy({self.var_spmd_strategy.__str__()})"

    def __repr__(self) -> str:
        return self.__str__()


class VarSPMDStrategyGroup:

    def __init__(self, *var_spmd_strategy_group) -> None:
        self.var_spmd_strategy_group = list(var_spmd_strategy_group)

    def append(self, var_spmd_strategy: VarSPMDStrategy) -> None:
        self.var_spmd_strategy_group.append(var_spmd_strategy)

    def get_var_strtg(self, idx: int) -> VarSPMDStrategy:
        assert idx < len(self.var_spmd_strategy_group)
        return self.var_spmd_strategy_group[idx]

    def __eq__(self, other) -> bool:
        if len(self.var_spmd_strategy_group) == len(other.var_spmd_strategy_group):
            for i, j in zip(self.var_spmd_strategy_group, other.var_spmd_strategy_group):
                if i != j:
                    return False
            return True
        return False

    def __getitem__(self, idx) -> VarSPMDStrategy:
        return self.var_spmd_strategy_group[idx]

    def __str__(self) -> str:
        return f"VarSPMDStrategyGroup({self.var_spmd_strategy_group.__str__()})"

    def __repr__(self) -> str:
        return self.__str__()


class NodeSPMDStrategy:

    def __init__(self, in_strtg_group: VarSPMDStrategyGroup,
                 out_strtg_group: VarSPMDStrategyGroup):
        self.in_strtg_group = in_strtg_group
        self.out_strtg_group = out_strtg_group

    def get_invar_strtg(self, invar_idx: int) -> VarSPMDStrategy:
        return self.in_strtg_group.get_var_strtg(invar_idx)

    def get_outvar_strtg(self, outvar_idx: int) -> VarSPMDStrategy:
        return self.out_strtg_group.get_var_strtg(outvar_idx)

    def __str__(self) -> str:
        return (f"NodeSPMDStrategy(in_strtg_group: {self.in_strtg_group}, "
                f"out_strtg_group: {self.out_strtg_group})")

    def __repr__(self) -> str:
        return self.__str__()


class NodeSPMDStrategyPool:

    def __init__(self):
        self.strategies = []

    def add_strategy(self, strtg: NodeSPMDStrategy):
        self.strategies.append(strtg)

    def find_matched_out(self, out_var_idx: int, expected_strtg: VarSPMDStrategy):
        for strtg_idx, nd_strtg in enumerate(self.strategies):
            if nd_strtg.out_strtg_group.get_var_strtg(out_var_idx) == expected_strtg:
                return strtg_idx

        return -1

    def get_strtg(self, idx: int) -> NodeSPMDStrategy:
        assert idx < len(self.strategies)
        return self.strategies[idx]

    def strtg_num(self) -> int:
        return len(self.strategies)

    def __str__(self) -> str:
        return f"{self.strategies})"

    def __repr__(self) -> str:
        return self.__str__()


def get_sharding_strategy(sharding_anns, shard_dim_id):
    spmd_strategy = VarSPMDStrategyGroup()

    for tensor_ann in sharding_anns.annotation:
        shard_dim_ = None
        for dim_idx, dim_ann in enumerate(tensor_ann):
            if dim_ann.shard_dim_id == shard_dim_id:
                shard_dim_ = dim_idx
        if shard_dim_ is not None:
            spmd_strategy.append(VarSPMDStrategy(SPMD(SPMD.SHARD, {"dim": shard_dim_})))
        else:
            spmd_strategy.append(VarSPMDStrategy(SPMD(SPMD.REPLICATE)))

    return spmd_strategy


def combination_to_sharding_strategy(comm_anns, all_replicate=False):
    # example of comm_anns:
    # functools.partial(<function CombinationFunc.gather at 0x7fab788efd30>, dim=0)

    spmd_strategy = VarSPMDStrategyGroup()

    if not (isinstance(comm_anns, list) or isinstance(comm_anns, tuple)):
        comm_anns = [comm_anns]
    for comm_ann in comm_anns:
        func_name = comm_ann.func.__name__
        if all_replicate or func_name == "identity":
            spmd_strategy.append(VarSPMDStrategy(SPMD(SPMD.REPLICATE)))
        elif func_name == "gather":
            spmd_strategy.append(VarSPMDStrategy(SPMD(SPMD.SHARD, comm_ann.keywords)))
        elif func_name == "reduce":
            spmd_strategy.append(VarSPMDStrategy(SPMD(SPMD.PARTIAL, comm_ann.keywords)))

    return spmd_strategy


_dtype2byte = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bool": 0.125,
    "int32": 4,
    "int64": 8,
    "uint32": 4,
    "uint8": 1,
    "complex64": 16,
}


class MetaVar:

    id_counter: int = 0

    @staticmethod
    def generate_unique_id():
        unique_id = MetaVar.id_counter
        MetaVar.id_counter += 1
        return unique_id

    @staticmethod
    def clear_id_counter():
        MetaVar.id_counter = 0

    def __init__(self, name, shape, dtype) -> None:
        self.unique_id = self.generate_unique_id()
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.up_node = None
        self.idx_for_up = -1
        self.down_nodes = []
        self.indice_for_down = []

    def get_var_size(self):
        if len(self.shape) == 0:
            return _dtype2byte[self.dtype]
        num_ele = reduce((lambda x, y: x * y), self.shape)
        return _dtype2byte[self.dtype] * num_ele

    def __str__(self, details=False) -> str:
        return self.name.__str__()

    def debug_str(self) -> str:
        return f"{self.name}({self.shape}, {self.dtype})"

    def __repr__(self) -> str:
        return self.__str__()


# (TODO) need to automatic find the heavy ops during MetaSPMD annotation.
_heavy_ops = [
    "convolution",
    "scaled_dot_product",
    "aten.baddbmm",
    "aten.mm",
    "aten.bmm",
    "dot",
]


class MetaNode:

    id_counter: int = 0

    @staticmethod
    def generate_unique_id():
        unique_id = MetaVar.id_counter
        MetaVar.id_counter += 1
        return unique_id

    @staticmethod
    def clear_id_counter():
        MetaVar.id_counter = 0

    def __init__(self, name, op_name, invars: List[MetaVar], outvars: List[MetaVar],
                 sharding_info, is_placeholder=False) -> None:
        self.unique_id = self.generate_unique_id()
        self.cluster_id = -1
        self.name = name
        self.op_name = op_name
        self.invars = invars
        for idx, in_var in enumerate(invars):
            if in_var:
                if isinstance(in_var, MetaVar):  # Lansong(TODO): should remove this line later
                    in_var.down_nodes.append(self)
                    in_var.indice_for_down.append(idx)
        self.compact_out_idx_tbl = []

        self.outvars = outvars
        for idx, out_var in enumerate(outvars):
            if out_var:
                out_var.up_node = self
                out_var.idx_for_up = idx
        self.compact_in_idx_tbl = []

        self.strtg_pool = None
        self.unique_key_ = None

        self.sharding_info = sharding_info
        self.is_placeholder = is_placeholder

    def set_in_var(self, in_var: MetaVar, idx: int):
        assert idx < len(self.compact_in_idx_tbl)
        compact_idx = self.compact_in_idx_tbl[idx]

        assert compact_idx < len(self.invars)
        assert not self.invars[compact_idx]
        self.invars[compact_idx] = in_var
        in_var.down_nodes.append(self)
        in_var.indice_for_down.append(compact_idx)

    def set_out_var(self, out_var: MetaVar, idx: int):
        assert idx < len(self.compact_out_idx_tbl)
        compact_idx = self.compact_out_idx_tbl[idx]

        assert compact_idx < len(self.outvars)
        assert not self.outvars[compact_idx]
        self.outvars[compact_idx] = out_var
        out_var.up_node = self
        out_var.idx_for_up = compact_idx

    def unique_key(self):
        if self.unique_key_ is None:
            self.unique_key_ = str(self.name)
        return self.unique_key_

    def _replicate_strategy(self):
        invars_strategy = VarSPMDStrategyGroup()
        for _ in self.invars:
            invars_strategy.append(VarSPMDStrategy(SPMD(SPMD.REPLICATE)))
        outvars_strategy = VarSPMDStrategyGroup()
        for _ in self.outvars:
            outvars_strategy.append(VarSPMDStrategy(SPMD(SPMD.REPLICATE)))
        return NodeSPMDStrategy(invars_strategy, outvars_strategy)

    def get_strtg(self, idx: int) -> NodeSPMDStrategy:
        assert self.strtg_pool
        return self.strtg_pool.get_strtg(idx)

    def get_strtg_pool(self) -> NodeSPMDStrategyPool:
        if self.strtg_pool is not None:
            return self.strtg_pool

        strategy_list_1d = []

        sharding_anns = self.sharding_info['sharding_ann']
        comm_anns = self.sharding_info['combination_ann']

        # some ops like torch.ops.aten.scalar_tensor.default have no invars
        if len(comm_anns) == 0 and len(self.invars) == 0:
            self.strtg_pool = NodeSPMDStrategyPool()
            return self.strtg_pool

        for comm_ann in comm_anns:
            in_strtg_group = get_sharding_strategy(sharding_anns, comm_ann)
            out_strtg_group = combination_to_sharding_strategy(comm_anns[comm_ann])
            strategy_list_1d.append(NodeSPMDStrategy(in_strtg_group, out_strtg_group))

        if all(op not in self.op_name for op in _heavy_ops) or len(comm_anns) == 0:
            strategy_list_1d.append(self._replicate_strategy())

        # (FIXME) modify stratgy here, need remove
        #     if "batch_norm" in self.name:
        #         dp_strategy = {
        #             "invars_sharding":
        #             get_sharding_strategy(sharding_anns, None),
        #             "outvars_sharding":
        #             combination_to_sharding_strategy(comm_anns[comm_ann], all_replicate=True)
        #         }
        #         dp_strategy["invars_sharding"][0] = SPMD(SPMD.SHARD, {'dim': 0})
        #         dp_strategy["outvars_sharding"][0] = SPMD(SPMD.SHARD, {'dim': 0})
        #         if "backward" in self.name:
        #             dp_strategy["invars_sharding"][1] = SPMD(SPMD.SHARD, {'dim': 0})
        #             dp_strategy["outvars_sharding"][1] = SPMD(SPMD.PARTIAL, {'ops': ReduceOp.SUM})
        #             dp_strategy["outvars_sharding"][2] = SPMD(SPMD.PARTIAL, {'ops': ReduceOp.SUM})
        #         strategy_list = [dp_strategy, replicate_strategy]

        # if "convolution.default" in self.name:
        #     del strategy_list[1]
        # if "convolution_backward.default" in self.name:
        #     del_idx = None
        #     for idx, s in enumerate(strategy_list):
        #         if s["invars_sharding"][-1] == SPMD(SPMD.SHARD, {'dim': 1}):
        #             del_idx = idx
        #             break
        #     if del_idx is not None:
        #         del strategy_list[del_idx]

        self.strtg_pool = NodeSPMDStrategyPool()
        if DEVICE_MESH_1D == -1:
            for idx1, s1 in enumerate(strategy_list_1d):
                for idx2, s2 in enumerate(strategy_list_1d):
                    # [Shard(i), Shard(i)] is not support for pytorch dtensor runtime
                    if get_backend() == "torch":
                        if any(i.state == SPMD.SHARD for i in s1.in_strtg_group) and idx1 == idx2:
                            continue
                    invars_strategy = VarSPMDStrategyGroup()
                    for i, j in zip(s1.in_strtg_group, s2.in_strtg_group):
                        invars_strategy.append(i + j)
                    outvars_strategy = VarSPMDStrategyGroup()
                    for i, j in zip(s1.out_strtg_group, s2.out_strtg_group):
                        outvars_strategy.append(i + j)
                    self.strtg_pool.add_strategy(
                        NodeSPMDStrategy(invars_strategy, outvars_strategy))
        else:
            replicate_ = VarSPMDStrategy(SPMD(SPMD.REPLICATE))
            for s in strategy_list_1d:
                if DEVICE_MESH_1D == 0:
                    invars_strategy = VarSPMDStrategyGroup()
                    for i in s.in_strtg_group:
                        invars_strategy.append(replicate_ + i)
                    outvars_strategy = VarSPMDStrategyGroup()
                    for i in s.out_strtg_group:
                        outvars_strategy.append(replicate_ + i)
                    self.strtg_pool.add_strategy(
                        NodeSPMDStrategy(invars_strategy, outvars_strategy))
                elif DEVICE_MESH_1D == 1:
                    invars_strategy = VarSPMDStrategyGroup()
                    for i in s.in_strtg_group:
                        invars_strategy.append(i + replicate_)
                    outvars_strategy = VarSPMDStrategyGroup()
                    for i in s.out_strtg_group:
                        outvars_strategy.append(i + replicate_)
                    self.strtg_pool.add_strategy(
                        NodeSPMDStrategy(invars_strategy, outvars_strategy))
                else:
                    exit(-1)

        if mdconfig.log_level <= logging.DEBUG:
            print("node: %s" % self.name.__str__())
            print(self.strtg_pool)
        return self.strtg_pool

    def __str__(self) -> str:
        return self.name.__str__()

    def debug_str(self) -> str:
        return_str = f"{self.name} ({self.op_name})\n"
        return_str += "invars: " + ",".join([var.debug_str() for var in self.invars]) + "\n"
        return_str += "outvars: " + ",".join(
            [var.debug_str() if var is not None else "None" for var in self.outvars]) + "\n"
        return_str += f"sharding_info: {self.sharding_info}"
        return return_str

    def __repr__(self) -> str:
        return self.__str__()


class ClusterStrategy:

    def __init__(self):
        self.in_strtg_group = None
        self.out_strtg_group = None
        self.inner_strtg_group = None
        self.node_strategies = {}

    def set_node_strategy(self, nd_id: int, nd_strtg_id: int, nd_strtg: NodeSPMDStrategy):
        assert nd_id not in self.node_strategies

        self.node_strategies[nd_id] = [nd_strtg_id, nd_strtg]

    def __str__(self) -> str:
        return (f"ClusterStrategies(in_strtg_group: {self.in_strtg_group}, "
                f"out_strtg_group: {self.out_strtg_group}, "
                f"inner_strtg_group: {self.inner_strtg_group})")

    def __repr__(self) -> str:
        return self.__str__()


class NodeIOStrategies:

    def __init__(self, node: MetaNode):
        self.node = node
        self.in_strategies = [[] for _ in range(len(node.invars))]
        if self.node.is_placeholder:
            self.in_strategies.append([])
        self.out_strategies = [[] for _ in range(len(node.outvars))]

    def add_in_strategy(self, in_idx: int, var_strtg: VarSPMDStrategy):
        if not self.node.is_placeholder:
            assert in_idx < len(self.node.invars)
        self.in_strategies[in_idx].append(var_strtg)

    def add_out_strategy(self, out_idx: int, var_strtg: VarSPMDStrategy):
        assert out_idx < len(self.node.outvars)
        self.out_strategies[out_idx].append(var_strtg)

    def get_invar_strtg(self, invar_idx: int, strtg_idx: int) -> VarSPMDStrategy:
        return self.in_strategies[invar_idx][strtg_idx]

    def get_outvar_strtg(self, outvar_idx: int, strtg_idx: int) -> VarSPMDStrategy:
        return self.out_strategies[outvar_idx][strtg_idx]

    def get_invar_strtg_list(self, invar_idx: int):  # return list of VarSPMDStrategy
        return self.in_strategies[invar_idx]

    def get_outvar_strtg_list(self, outvar_idx: int):  # return list of VarSPMDStrategy
        return self.out_strategies[outvar_idx]

    def __str__(self) -> str:
        res = str(self.node)
        res += f"\nin strategies: {self.in_strategies}"
        res += f"\noutput strategies: {self.out_strategies}"
        return res

    def __repr__(self) -> str:
        return self.__str__()


class ClusterStrategyPool:

    def __init__(self, cluster: MetaNodeCluster):
        self.cluster = cluster
        self.node_io_strategies = {}
        self.node_strategies = {}
        self.strtg_num = 0

        for nd in cluster.nodes.values():
            self.node_io_strategies[nd.unique_id] = NodeIOStrategies(nd)
            self.node_strategies[nd.unique_id] = []

    def get_invar_strtg(self, nd_id: int, invar_idx: int, strtg_idx: int) -> VarSPMDStrategy:
        assert strtg_idx < self.strtg_num
        nd_io_strtg = self.node_io_strategies[nd_id]
        return nd_io_strtg.get_invar_strtg(invar_idx, strtg_idx)

    def get_outvar_strtg(self, nd_id: int, outvar_idx: int, strtg_idx: int) -> VarSPMDStrategy:
        assert strtg_idx < self.strtg_num
        nd_io_strtg = self.node_io_strategies[nd_id]
        return nd_io_strtg.get_outvar_strtg(outvar_idx, strtg_idx)

    def get_invar_strtg_list(self, nd_id: int, invar_idx: int) -> list[VarSPMDStrategy]:
        nd_io_strtg = self.node_io_strategies[nd_id]
        return nd_io_strtg.get_invar_strtg_list(invar_idx)

    def get_outvar_strtg_list(self, nd_id: int, outvar_idx: int) -> list[VarSPMDStrategy]:
        nd_io_strtg = self.node_io_strategies[nd_id]
        return nd_io_strtg.get_outvar_strtg_list(outvar_idx)

    def get_node_strtg(self, nd_id: int, strtg_idx: int) -> NodeSPMDStrategy:
        assert nd_id in self.node_strategies
        assert strtg_idx < self.strtg_num
        return self.node_strategies[nd_id][strtg_idx]

    def add_strategy(self, strtg: ClusterStrategy):
        self.strtg_num += 1
        for nd_id, nd_strtg_info in strtg.node_strategies.items():
            nd_strtg_id = nd_strtg_info[0]
            nd_strtg = nd_strtg_info[1]
            nd = self.cluster.nodes[nd_id]

            # store node strategies with original format
            self.node_strategies[nd.unique_id].append(nd_strtg)

            # store io strategies
            nd_io_strtg = self.node_io_strategies[nd.unique_id]
            if nd.is_placeholder:
                outvar_strtg = nd_strtg.get_outvar_strtg(0)
                nd_io_strtg.add_in_strategy(0, outvar_strtg)
            else:
                for invar_idx in range(len(nd.invars)):
                    invar_strtg = nd_strtg.get_invar_strtg(invar_idx)
                    nd_io_strtg.add_in_strategy(invar_idx, invar_strtg)

                
            for outvar_idx in range(len(nd.outvars)):
                outvar_strtg = nd_strtg.get_outvar_strtg(outvar_idx)
                nd_io_strtg.add_out_strategy(outvar_idx, outvar_strtg)

    def get_strtg_num(self) -> int:
        return self.strtg_num

    def __str__(self) -> str:
        res = f"node strategies: {self.node_strategies}"
        res += f"\nnode io strategies: {self.node_io_strategies}"
        return res

    def __repr__(self) -> str:
        return self.__str__()


class ClusterArgs:

    def __init__(self) -> None:
        self.descs = []

    def add_arg(self, input_node: MetaNode, idx: int) -> None:
        self.descs.append([input_node, idx])

    def __str__(self) -> str:
        return_str = f"{self.descs.__str__()}"
        return return_str


class MetaNodeCluster:

    def __init__(self, unique_id: int) -> None:
        self.unique_id = unique_id
        self.args = ClusterArgs()
        self.output_node = None
        self.nodes = {}
        self.strategy_pool = None

    def add_node(self, meta_node: MetaNode) -> None:
        assert meta_node.unique_id not in self.nodes
        assert meta_node.unique_id >= 0
        self.nodes[meta_node.unique_id] = meta_node
        meta_node.cluster_id = self.unique_id

    def back_build_strategy(self, nd: MetaNode, nd_strtg_idx: int,
                            cluster_strtg: ClusterStrategy) -> bool:
        succ = True
        for invar_idx, invar in enumerate(nd.invars):
            if not invar:
                continue

            up_node = invar.up_node
            if not up_node:
                continue

            if up_node.unique_id not in self.nodes:
                # up_node is not in the cluster
                continue

            # we should guarantee no multi-output node inside cluster
            idx_for_up = invar.idx_for_up
            assert len(up_node.outvars) == 1
            assert idx_for_up == 0

            nd_strtg = nd.strtg_pool.get_strtg(nd_strtg_idx)
            expected_var_strtg = nd_strtg.get_invar_strtg(invar_idx)

            up_nd_strtg_pool = up_node.get_strtg_pool()
            up_nd_strtg_idx = up_nd_strtg_pool.find_matched_out(idx_for_up, expected_var_strtg)
            if up_nd_strtg_idx >= 0:
                up_nd_strtg = up_nd_strtg_pool.get_strtg(up_nd_strtg_idx)
                cluster_strtg.set_node_strategy(up_node.unique_id, up_nd_strtg_idx, up_nd_strtg)
                if not self.back_build_strategy(up_node, up_nd_strtg_idx, cluster_strtg):
                    succ = False
            else:
                succ = False

        return succ

    def finalize(self) -> None:
        for node in self.nodes.values():
            if node.invars:
                for idx, invar in enumerate(node.invars):
                    if not invar.up_node:
                        self.args.add_arg(node, idx)
                    elif invar.up_node.unique_id not in self.nodes:
                        self.args.add_arg(node, idx)
            else:
                # placeholder or get_attr node
                self.args.add_arg(node, 0)

        all_node_outs = []
        for node in self.nodes.values():
            all_node_outs.extend(node.outvars)

        # collect all variables which are read by outside of cluster
        cluster_out_vars = []
        for var in all_node_outs:
            if var:
                if not var.down_nodes:
                    # output MetaVar of graph has an empty down_nodes list
                    cluster_out_vars.append(var)
                else:
                    for node in var.down_nodes:
                        if node.unique_id not in self.nodes:
                            cluster_out_vars.append(var)

        # get or build strategy info for out node
        out_node = None
        for outvar in cluster_out_vars:
            if not out_node:
                out_node = outvar.up_node
            else:
                # be sure only one output node in a cluster
                assert out_node == outvar.up_node

        self.output_node = out_node

        # build strategy candidates for cluster
        self.strategy_pool = ClusterStrategyPool(self)
        out_strtg_pool = out_node.get_strtg_pool()

        for out_strtg_idx in range(out_strtg_pool.strtg_num()):
            cluster_strtg = ClusterStrategy()
            out_strtg = out_node.get_strtg(out_strtg_idx)
            cluster_strtg.set_node_strategy(out_node.unique_id, out_strtg_idx, out_strtg)

            if self.back_build_strategy(out_node, out_strtg_idx, cluster_strtg):
                assert len(cluster_strtg.node_strategies) == len(self.nodes)
                self.strategy_pool.add_strategy(cluster_strtg)
            else:
                # failed to search sync free cluster strategy
                # Lansong(TODO) we should decompose the cluster into multiple clusters
                logger.debug(
                    "failed to search sync free cluster strategy for node %s, strategy %s",
                    str(out_node), str(out_strtg_pool.get_strtg(out_strtg_idx)))
                continue

    def get_strtg_pool(self) -> ClusterStrategyPool:
        assert self.strategy_pool
        return self.strategy_pool

    def __str__(self) -> str:
        cluster_info = "\ncluster id: " + str(self.unique_id) + "\n  nodes: "
        for node in self.nodes.values():
            cluster_info += str(node) + ", "

        cluster_info += "\n  inputs: " + str(self.args)
        cluster_info += "\n  output: " + str(self.output_node)

        cluster_info += "\nstrategies:\n" + str(self.strategy_pool)
        return cluster_info

    def __repr__(self) -> str:
        return self.__str__()


class MetaGraph:

    def __init__(self, ori_struct) -> None:
        self.ori_struct = ori_struct
        self.input_list = []
        self.op_list = []
        self.output_list = []

        self.node_clusters = []

        self.state_io_map = {}

    def add_input(self, placeholder: MetaNode) -> None:
        self.input_list.append(placeholder)

    def add_node(self, meta_node: MetaNode) -> None:
        self.op_list.append(meta_node)

    def add_output(self, outvars: MetaVar) -> None:
        self.output_list.append(outvars)

    def __str__(self) -> str:
        return_str = f"=====================\n[MetaIR]\n\ninput_list: {self.input_list.__str__()}\n\n"
        for op in self.op_list:
            return_str += f"{op.outvars} <--- [{op.op_name}] --- {op.invars}\n"
        return_str += f"\noutput_list: {self.output_list.__str__()}\n=====================\n"
        return_str += f"\nnode clusters:"
        for cluster in self.node_clusters:
            return_str += str(cluster)
        return_str += "\n=====================\n"
        return return_str

    def __repr__(self) -> str:
        return self.__str__()

    def liveness(self, reserve_input=False):
        liveness_set = set([var.name for var in self.output_list])

        liveness_list = []

        for op in reversed(self.op_list):
            for var in op.invars:
                liveness_set.add(var.name)
            for var in op.outvars:
                if var:  # skip if an output is dangling
                    liveness_set.add(var.name)

            liveness_set_line = copy.deepcopy(liveness_set)
            if reserve_input:
                liveness_set_line.union(set([var.name for var in self.input_list]))

            liveness_list.insert(0, liveness_set_line)

            for var in op.outvars:
                if var:  # skip if an output is dangling
                    liveness_set.remove(var.name)

        return liveness_list

    def build_fine_grain_clusters(self):
        cluster_id = 0
        for node in self.op_list:
            cluster = MetaNodeCluster(unique_id=cluster_id)
            cluster.add_node(node)
            cluster.finalize()
            self.node_clusters.append(cluster)
            cluster_id += 1

    def find_cone_roots(self):
        cone_roots = []
        logger.debug("node num: %s" % len(self.op_list))
        for node in self.op_list:
            down_node_num = 0
            valid_outvars = []
            for out_var in node.outvars:
                if out_var:
                    for down_nd in out_var.down_nodes:
                        if down_nd:
                            valid_outvars.append(out_var)
                            down_node_num += 1

            if down_node_num > 1 or down_node_num == 0:
                cone_roots.append(node)
                continue

            up_node_num = 0
            valid_invars = []
            for in_var in node.invars:
                if in_var and in_var.up_node:
                    valid_invars.append(in_var)
                    up_node_num += 1

            if up_node_num > 1:
                cone_roots.append(node)
                continue
            elif up_node_num == 0:
                continue

            # single up node and single down node
            # Lansong(TODO) we should allow multi-input node as inner node of cone
            assert len(valid_outvars) == 1 and len(valid_invars) == 1
            out_size = valid_outvars[0].get_var_size()
            in_size = valid_invars[0].get_var_size()

            if out_size < in_size:
                cone_roots.append(node)
                continue

        return cone_roots

    def build_cone_cluster(self, nd: MetaNode, root_ids, cluster: MetaNodeCluster):
        cluster.add_node(nd)
        for in_var in nd.invars:
            if in_var and in_var.up_node:
                if in_var.up_node.unique_id not in root_ids:
                    #print("recursive build cone from %s" % str(in_var.up_node))
                    self.build_cone_cluster(in_var.up_node, root_ids, cluster)

    def build_cone_clusters(self):
        cone_roots = self.find_cone_roots()
        logger.debug("root num: %d" % len(cone_roots))
        logger.debug(cone_roots)

        root_ids = set()
        for cone_root in cone_roots:
            root_ids.add(cone_root.unique_id)

        cluster_id = 0
        for cone_root in cone_roots:
            cluster = MetaNodeCluster(unique_id=cluster_id)
            self.build_cone_cluster(cone_root, root_ids, cluster)
            cluster.finalize()
            self.node_clusters.append(cluster)
            cluster_id += 1

    def coarsen(self, coarsen_level: int):
        if coarsen_level == 0:
            self.build_fine_grain_clusters()
        elif coarsen_level == 1:
            self.build_cone_clusters()
        else:
            print("Lansong 2do: support more aggressive coarsening")
            # call cone cluster building instead
            self.build_cone_clusters()

        if mdconfig.log_level <= logging.DEBUG:
            for cluster in self.node_clusters:
                print(str(cluster))

    def dump(self):
        if mdconfig.dump_dir is not None:
            os.makedirs(mdconfig.dump_dir, exist_ok=True)
            filename = os.path.join(mdconfig.dump_dir, "metair.txt")
            with open(filename, "w") as f:
                f.write(self.__str__())
            logger.info(f"MetaIR dump into {filename}")

    def get_input_strategy(self, opt_strategy):

        partial_strategy = {}
        for op in self.op_list:
            if op.is_placeholder:
                if op.unique_key() in opt_strategy:
                    strategy = opt_strategy[op.unique_key()]['strategy'].out_strtg_group[0]
                    partial_strategy[op.outvars[0]] = strategy
                else:
                    logger.warning(f"{op.unique_key()} not found in opt_strategy. (maybe scalar tensor)")

        partial_strategy_list = []

        for var in self.input_list:
            if var in partial_strategy:
                partial_strategy_list.append(partial_strategy[var])
            else:
                partial_strategy_list.append(
                    [SPMD(SPMD.REPLICATE),
                     SPMD(SPMD.REPLICATE)])

        return partial_strategy_list