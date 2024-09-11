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

from collections import defaultdict
from ctypes import cast
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import permutations
import operator
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed._functional_collectives import _expand_group
from torch.fx.node import Node, _get_qualified_name, map_arg
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._tensor.ops.view_ops import (view_groups, normalize_sizes, expand,
                                                    propagate_shape_and_sharding,
                                                    compute_local_shape)

import easydist.config as mdconfig
from easydist.torch.device_mesh import get_device_mesh
from easydist.torch.experimental.pp.split_utils import ANNOTATION_OPS
from easydist.torch.utils import to_torch_spmd, EDInfo, EDNodeType, create_meta_from_node
from easydist.utils.testing.mock import MockDeviceMesh
from easydist.metashard.metair import VarSPMDStrategy, SPMD, VarSPMDStrategyGroup, NodeSPMDStrategy
from easydist.metashard.combination import ReduceOp

logger = logging.getLogger(__name__)

reduce_map = {
    ReduceOp.SUM: "sum",
    ReduceOp.MAX: "max",
    ReduceOp.MIN: "min",
    ReduceOp.AVG: "avg",
}

CREATE_ATEN_OP = [
    torch.ops.aten.empty.memory_format, torch.ops.aten.zeros.default, torch.ops.aten.ones.default,
    torch.ops.aten.scalar_tensor.default, torch.ops.aten.arange.default,
    torch.ops.aten.zeros.default, torch.ops.aten.arange.start
]


view_op_map = {
    torch.ops.aten.view.default:
    lambda input_shape, shape: view_groups(input_shape, shape),
    torch.ops.aten._unsafe_view.default:
    lambda input_shape, shape: view_groups(input_shape, shape),
    torch.ops.aten.reshape.default:
    lambda input_shape, shape: view_groups(input_shape, shape),
    torch.ops.aten.expand.default:
    lambda input_shape, sizes: expand(input_shape, normalize_sizes(sizes)),
}


def all_reduce_start(self: torch.Tensor, reduceOp: str, group: List[int], tag: str = ""):
    tag, rankset, group_size = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_reduce(self, reduceOp, tag, rankset,
                                                  group_size)  # type: ignore[attr-defined]
    return tensor


def all_reduce_end(self: torch.Tensor, reduceOp: str, group: List[int], tag: str = ""):
    return torch.ops.c10d_functional.wait_tensor(self)


def all_gather_start(self: torch.Tensor, gather_dim: int, group: List[int], tag: str = ""):
    if not self.is_contiguous():
        self = self.contiguous()
    tag, rankset, group_size = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_gather_into_tensor(
        self, tag, rankset, group_size)  # type: ignore[attr-defined]
    return tensor


def all_gather_end(self: torch.Tensor, gather_dim: int, group: List[int], tag: str = ""):
    self = torch.ops.c10d_functional.wait_tensor(self)
    tag, rankset, group_size = _expand_group(group, tag)
    if gather_dim != 0:
        self = torch.cat(torch.chunk(self, group_size, dim=0), dim=gather_dim)
    return self


def scatter_wrapper(tensor, num_chunks, dim, indice):
    return torch.ops.aten.chunk(tensor, num_chunks, dim)[indice].contiguous()


def copy_wrapper(self, other):
    return torch.ops.aten.copy_.default(self, other)


def reduce_scatter_start(self: torch.Tensor,
                         reduceOp: str,
                         scatter_dim: int,
                         group: List[int],
                         tag: str = ""):
    tag, rankset, group_size = _expand_group(group, tag)
    assert (self.size(scatter_dim) % group_size == 0
            ), f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size}"
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops.c10d_functional.reduce_scatter_tensor(
        self, reduceOp, tag, rankset, group_size)  # type: ignore[attr-defined]
    return tensor


def reduce_scatter_end(self: torch.Tensor,
                       reduceOp: str,
                       scatter_dim: int,
                       group: List[int],
                       tag: str = ""):
    return torch.ops.c10d_functional.wait_tensor(self)


def all_to_all_start(tensor, gather_dim, scatter_dim, num_chunks, indice, ranks, tag: str = ""):
    # (TODO) call all_gather for all_to_all, need to use all_to_all for less comm size
    gathered_tensor = all_gather_start(tensor, gather_dim, ranks, tag)
    return gathered_tensor


def all_to_all_end(tensor, gather_dim, scatter_dim, num_chunks, indice, ranks, tag: str = ""):
    tensor = all_gather_end(tensor, gather_dim, ranks, tag)
    return scatter_wrapper(tensor, num_chunks, scatter_dim, indice)


COMM_FUNCS = [all_reduce_start, all_gather_start, reduce_scatter_start, all_to_all_start]
COMM_SYNC_FUNCS = [all_reduce_end, all_gather_end, reduce_scatter_end, all_to_all_end]
CUSTOM_FUNCS = COMM_FUNCS + COMM_SYNC_FUNCS + [scatter_wrapper, copy_wrapper]


def materialize(x, device):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bool:
            return torch.rand(x.size(), dtype=torch.float, device=device) > 0.5
        elif torch.is_floating_point(x):
            return torch.rand(x.size(), dtype=x.dtype, device=device)
        else:
            return torch.randint(high=8, size=x.size(), dtype=x.dtype, device=device)
    return x


def redist_tensor_func(input_tensor, specs):
    if isinstance(input_tensor, DTensor) and input_tensor.size() != torch.Size([0]):
        spmd_mesh = get_device_mesh('spmd')
        if specs != input_tensor._spec.placements:
            return input_tensor.redistribute(spmd_mesh, specs).contiguous()
    return input_tensor.contiguous()


def insert_spmd_head(body):
    return ["from torch.distributed._tensor import Shard, Replicate\n", *body]


@torch.no_grad()
def to_dtensor(input_tensor, placements=None, global_size=None):
    if isinstance(input_tensor, torch.Tensor) and not isinstance(input_tensor, DTensor):
        spmd_mesh = get_device_mesh('spmd')
        if placements is None:
            placements = [Replicate()] * spmd_mesh.ndim
        if global_size is None:
            return DTensor.from_local(input_tensor, spmd_mesh, placements)
        else:
            return DTensor(input_tensor, spmd_mesh, placements, size=global_size)
    return input_tensor


def fix_in_gragh_tensor(fx_module: torch.fx.GraphModule, sharding_strategy):

    for node in fx_module.graph.nodes:
        if node.op == 'get_attr':
            with fx_module.graph.inserting_after(node):
                to_dtensor_node = fx_module.graph.call_function(to_dtensor, args=(node, ))

                node.replace_all_uses_with(to_dtensor_node)

                to_dtensor_node.update_arg(0, node)

        create_ops = [
            "torch.ops.aten.scalar_tensor.", "torch.ops.aten.ones.", "torch.ops.aten.empty.",
            "torch.ops.aten.arange.", "torch.ops.aten.zeros."
        ]

        if node.op == 'call_function':
            if node.target in CREATE_ATEN_OP:
                strategy, global_shape = None, None
                if node.name in sharding_strategy and len(sharding_strategy[node.name]) > 0:
                    strategy = sharding_strategy[node.name][0]
                    # modify the arg of creat ops
                    # (CAUTION!!) device_mesh maybe mock now,
                    # but the local shape need to be consistent of runtime
                    global_shape = node.args[0]
                    local_shape = list(global_shape)
                    spmd_mesh = get_device_mesh('spmd')
                    if isinstance(spmd_mesh, MockDeviceMesh):
                        logger.warning("maybe wrong shape in MockDeviceMesh for create ops.")
                    for idx, placement in enumerate(strategy):
                        if placement.is_shard():
                            shard_dim = placement.dim
                            split_size, pad_idx = divmod(global_shape[shard_dim],
                                                         spmd_mesh.size(idx))
                            local_shape[shard_dim] = split_size
                            if spmd_mesh.get_coordinate()[idx] < pad_idx:
                                local_shape[shard_dim] = split_size + 1

                    node.update_arg(0, local_shape)

                with fx_module.graph.inserting_after(node):
                    to_dtensor_node = fx_module.graph.call_function(to_dtensor,
                                                                    args=(node, strategy,
                                                                          global_shape))

                    node.replace_all_uses_with(to_dtensor_node)

                    to_dtensor_node.update_arg(0, node)

    fx_module.recompile()

    return fx_module


def replace_subsequence_use(node, arg_, redist_node):
    users_node = list(node.users.keys())
    node_next = node.next

    def maybe_replace_node(n: Node) -> Node:
        if n == arg_:
            return redist_node
        else:
            return n

    while node_next.name != "":
        if node_next in users_node:
            new_args = map_arg(node_next.args, maybe_replace_node)
            new_kwargs = map_arg(node_next.kwargs, maybe_replace_node)
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            node_next.args = new_args
            node_next.kwargs = new_kwargs
        node_next = node_next.next


def sharding_transform_dtensor(fx_module: torch.fx.GraphModule, sharding_strategy):
    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if node.name in sharding_strategy:
                node_strategy = sharding_strategy[node.name]

                # skip for create ops
                if node.target in CREATE_ATEN_OP:
                    continue

                node_args_flatten = pytree.tree_flatten(node.args)[0]
                invars = [arg for arg in node_args_flatten if isinstance(arg, Node)]
                assert (len(invars) == len(node_strategy))

                for arg_, arg_strategy in zip(invars, node_strategy):
                    with fx_module.graph.inserting_before(node):
                        redist_node = fx_module.graph.call_function(redist_tensor_func,
                                                                    args=(arg_, arg_strategy))

                        # FIXME: may introduce redundancy communication, update_arg for all subsequence use
                        node.replace_input_with(arg_, redist_node)
                        #replace_subsequence_use(node, arg_, redist_node)

    fx_module.graph.on_generate_code(lambda _: insert_spmd_head)

    fx_module.recompile()

    # (fix) %_tensor_constant0 : [#users=1] = get_attr[target=_tensor_constant0]
    fx_module = fix_in_gragh_tensor(fx_module, sharding_strategy)

    return fx_module


def _replicate_then_shard(tup: Tuple[int, SPMD, SPMD]) -> int:
    """
    This is a helper function to allow reordering _TransformInfo list. The high level
    idea is that we want to reorder the sharding redistributions so that the DTensor
    redistribution is consistent with its full tensor. This is built on top of two simple
    assumptions:
    1. Replication happens from inner to outer dimension. i.e. Shard -> Replicate
    2. Sharding happens from outer to inner dimension, i.e. Replicate -> Shard

    So we always put the replication first and put sharding later.
    """
    mesh_dim, src, dst = tup
    if (dst.is_replicate() or dst.is_partial()) and src.is_shard():
        return -mesh_dim
    elif (src.is_replicate() or src.is_partial()) and dst.is_shard():
        return mesh_dim
    else:
        return 0


@dataclass
class Partition:
    '''
        Tensor partition from a logical view, and the rank that holds it.
    '''
    start_coord: Tuple[int, ...]  # start coord of this partition
    end_coord: Tuple[int, ...]  # end coord of this partition
    rank: int  # rank that hold this partition
    partial_coord: Tuple[int, ...]  # partial coord of this partition

    def __post_init__(self):
        self._hash: Optional[int] = None

    def _hash_impl(self) -> int:
        return hash((self.start_coord, self.end_coord, self.rank, self.partial_coord))

    def __hash__(self) -> int:
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def __repr__(self) -> str:
        return f"{{{self.start_coord}->{self.end_coord} partial {self.partial_coord} on rank{self.rank}}}"

    def shard(self, tensor_dim: int, shard_num: int, shard_idx: int) -> "Partition":
        block_size = (self.end_coord[tensor_dim] - self.start_coord[tensor_dim] + shard_num - 1) // shard_num
        return Partition(
            self.start_coord[:tensor_dim] + (min(self.start_coord[tensor_dim] + block_size * shard_idx, self.end_coord[tensor_dim]),) + self.start_coord[tensor_dim+1:],
            self.end_coord[:tensor_dim] + (min(self.start_coord[tensor_dim] + block_size * (shard_idx + 1), self.end_coord[tensor_dim]),) + self.end_coord[tensor_dim+1:],
            self.rank,
            self.partial_coord
        )

    def partial(self, new_partial_idx: int) -> "Partition":
        return Partition(self.start_coord, self.end_coord, self.rank, self.partial_coord + (new_partial_idx,))

    @staticmethod
    def from_tensor_spec(placements: VarSPMDStrategy, global_shape: torch.Size) -> Tuple["Partition"]:
        tensor_mesh = get_device_mesh('spmd').mesh

        unravel = torch.unravel_index(torch.arange(tensor_mesh.numel()), tensor_mesh.shape)
        rank_to_coord = []
        for i in range(unravel[0].shape[0]):
            rank_to_coord.append(
                tuple(unravel[j][i].item() for j in range(tensor_mesh.ndim))
            )
        partitions = [
            Partition((0,) * len(global_shape), tuple(global_shape), rank, tuple()) for rank in tensor_mesh.flatten().tolist()
        ]

        for mesh_dim, placement in enumerate(placements):
            if placement.is_shard():
                tensor_dim = placement.args['dim']
                shard_num = tensor_mesh.size(mesh_dim)
                for partition in partitions:
                    partitions[partition.rank] = partition.shard(tensor_dim, shard_num, rank_to_coord[partition.rank][mesh_dim])
            elif placement.is_partial():
                for partition in partitions:
                    partitions[partition.rank] = partition.partial(rank_to_coord[partition.rank][mesh_dim])
        return tuple(partitions)

    def from_src(self, src_partition: "Partition") -> Optional["Partition"]:

        start_coord = tuple(max(s1, s2) for s1, s2 in zip(self.start_coord, src_partition.start_coord))
        end_coord = tuple(min(e1, e2) for e1, e2 in zip(self.end_coord, src_partition.end_coord))
        if any(s >= e for s, e in zip(start_coord, end_coord)) or src_partition.partial_coord != self.partial_coord:
            return None

        return Partition(start_coord, end_coord, src_partition.rank, src_partition.partial_coord)

    @staticmethod
    @lru_cache(maxsize=None)
    def gen_recv_meta(src_partitions: Tuple["Partition"], tgt_partitions: Tuple["Partition"]) -> Dict[int, List["Partition"]]:
        recv_meta = defaultdict(list)
        for tgt in tgt_partitions:
            cache = defaultdict(int)
            # TODO better load balance strategy
            for src in sorted(src_partitions, key=lambda x: abs(x.rank - tgt.rank)):
                intersection = tgt.from_src(src)
                if intersection is None:
                    continue
                cache_str = f"{intersection.start_coord}{intersection.end_coord}"
                if cache_str not in cache:
                    recv_meta[tgt.rank].append(intersection)
                cache[cache_str] += 1

        return recv_meta

    @staticmethod
    def gen_p2p_utils(src_tensor: torch.Tensor, rank: int, local_src_partition: "Partition", local_dst_partition: "Partition", recv_meta: Dict[int, List["Partition"]]) -> Tuple[List[dist.P2POp], List[dist.P2POp], torch.Tensor, List[Tuple[slice]]]:
        send_ops = []
        recv_ops = []
        recv_slices = []
        buffer_shape = tuple(e - s for s, e in zip(local_dst_partition.start_coord, local_dst_partition.end_coord))

        recv_buffer = torch.empty(buffer_shape, dtype=src_tensor.dtype, device=src_tensor.device)

        for recv_rank, intersections in recv_meta.items():
            for intersection in intersections:
                send_rank = intersection.rank
                src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))

                if rank == send_rank == recv_rank:
                    # assign the static intersection to the buffer
                    recv_buffer[tgt_slice] = src_tensor[src_slice]
                    continue

                if rank == send_rank:
                    src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    # TODO: possible to eliminate contiguous call to reduce mem usage?
                    send_ops.append(dist.P2POp(dist.isend, src_tensor[src_slice].contiguous(), recv_rank))

                if rank == recv_rank:
                    tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    # TODO: possible to eliminate contiguous call to reduce mem usage?
                    recv_ops.append(dist.P2POp(dist.irecv, recv_buffer[tgt_slice].contiguous(), send_rank))
                    recv_slices.append(tgt_slice)

        # TODO: handle the case where there is no communication, currently sending a dummy tensor
        if len(send_ops + recv_ops) == 0:
            # NOTE: recv_buffer, recv_slices will still be empty
            dummy_tensor = torch.zeros(1, dtype=src_tensor.dtype, device=src_tensor.device)
            send_ops.append(dist.P2POp(dist.isend, dummy_tensor, rank))
            recv_ops.append(dist.P2POp(dist.irecv, dummy_tensor, rank))

        return send_ops, recv_ops, recv_buffer, recv_slices

    @staticmethod
    def fill_recv_buffer(recv_ops: List[dist.P2POp], recv_buffer: torch.Tensor, recv_slices: List[Tuple[slice]]) -> torch.Tensor:
        for op, slice in zip(recv_ops, recv_slices):
            if op.op is not dist.irecv:
                raise ValueError("recv_ops must be irecv")
            recv_buffer[slice] = op.tensor

        return recv_buffer


def _gen_immediate_transform_infos(
    src_placements: VarSPMDStrategy,
    dst_placements: VarSPMDStrategy,
) -> Tuple[List[Tuple[int, SPMD, SPMD]], VarSPMDStrategy]:
    """
    Similar to _gen_transform_infos, but only perform simple transformation by dim if it can be achieved by one single collective operation.
    Consider device dim k:
        1. R -> R: No-op
        2. R -> S(i): Safe when num of S(i) are the same in cur_spec[:k] and dst_spec[:k] and no S(i) in cur_spec[k+1:]
        3. R -> P: Always safe

        4. S(i) -> R: Safe when no S(i) in cur_spec[k+1:]
        5. S(i) -> S(j): Safe when S(i) meets 4 and S(j) meets 2
        6. S(i) -> P: Used as S(i) -> R during backward

        7. P -> R: Always safe
        8. P -> S(i): P -> R -> S(i), safe when 7 and 2 are satisfied
        9. P -> P: No-op

        10. if P -> S(i) is not executed, do P -> R since P2P cannot handle reduce_scatter
    """
    cur_placements = deepcopy(src_placements.var_spmd_strategy)
    transform_infos: List[Tuple[int, SPMD, SPMD]] = []

    def count_shards(placements: List[SPMD], shard: SPMD) -> int:
        return sum(1 for p in placements if p.is_shard() and p.args['dim'] == shard.args['dim'])

    max_immediate_transform_cnt = 0
    best_transform_infos = []
    best_cur_placements = deepcopy(list(src_placements))

    for seq in permutations(reversed(range(len(src_placements)))):
        immediate_transform_cnt = 0
        cur_placements = deepcopy(list(src_placements))
        transform_infos = []
        def to_shard(k: int, tgt_shard: SPMD):
            assert tgt_shard.is_shard()
            return count_shards(cur_placements[:k], tgt_shard) == count_shards(cur_placements[:k], tgt_shard) and count_shards(cur_placements[k + 1:], tgt_shard) == 0

        def from_shard(k: int, cur_shard: SPMD):
            assert cur_shard.is_shard()
            return count_shards(cur_placements[k + 1:], cur_shard) == 0

        for k in seq:
            cur = cur_placements[k]
            tgt = dst_placements[k]

            if cur == tgt:
                continue

            if cur.is_replicate():
                if tgt.is_shard():
                    if to_shard(k, tgt):
                        transform_infos.append(
                            (k, cur, tgt)
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
                elif tgt.is_partial():
                    transform_infos.append(
                        (k, cur, tgt)
                    )
                    cur_placements[k] = tgt
                    immediate_transform_cnt += 1
            elif cur.is_shard():
                if tgt.is_replicate():
                    if from_shard(k, cur):
                        transform_infos.append(
                            (k, cur, tgt)
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
                elif tgt.is_shard():
                    if cur.args['dim'] != tgt.args['dim'] and from_shard(k, cur) and to_shard(k, tgt):
                        transform_infos.append(
                            (k, cur, tgt)
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
                elif tgt.is_partial():
                    if from_shard(k, cur):
                        transform_infos.append(
                            (k, cur, tgt)
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
            else:
                if tgt.is_replicate():
                    transform_infos.append(
                        (k, cur, tgt)
                    )
                    cur_placements[k] = tgt
                    immediate_transform_cnt += 1
                elif tgt.is_shard():
                    if to_shard(k, tgt):
                        transform_infos.append(
                            (k, cur, tgt)
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1

        if immediate_transform_cnt > max_immediate_transform_cnt:
            max_immediate_transform_cnt = immediate_transform_cnt
            best_transform_infos = transform_infos
            best_cur_placements = cur_placements

    # transform rest P -> S to R -> S since P2P cannot handle reduce_scatter
    for k, (cur, tgt) in enumerate(zip(best_cur_placements, dst_placements)):
        if cur.is_partial() and tgt.is_shard():
            tgt = SPMD(SPMD.REPLICATE)
            best_transform_infos.append(
                (k, cur, tgt),
            )
            best_cur_placements[k] = tgt

    return best_transform_infos, VarSPMDStrategy(*best_cur_placements)


def do_p2p_comm_wrapper(global_shape: torch.Size, src_placements: VarSPMDStrategy, tgt_placements: VarSPMDStrategy, rank: int) -> Callable:
    src_partitions = Partition.from_tensor_spec(src_placements, global_shape)
    tgt_partitions = Partition.from_tensor_spec(tgt_placements, global_shape)
    recv_info = Partition.gen_recv_meta(src_partitions, tgt_partitions)

    def inner(local_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(local_tensor, FakeTensor):
            local_tensor = torch.empty(local_tensor.shape, dtype=local_tensor.dtype, device=local_tensor.device)
            assert not isinstance(local_tensor, FakeTensor)

        send_ops, recv_ops, local_tensor, recv_slices = Partition.gen_p2p_utils(local_tensor, rank, src_partitions[rank], tgt_partitions[rank], recv_info)
        works = dist.batch_isend_irecv(send_ops + recv_ops)
        for work in works:
            work.wait()
        local_tensor = Partition.fill_recv_buffer(recv_ops, local_tensor, recv_slices)

        return local_tensor
    return inner


def _gen_transform_infos(
    src_placements: VarSPMDStrategy,
    dst_placements: VarSPMDStrategy,
) -> List[Tuple[int, SPMD, SPMD]]:
    cur_placements = deepcopy(src_placements.var_spmd_strategy)
    transform_infos: List[Tuple[int, SPMD, SPMD]] = []

    for i in reversed(range(len(cur_placements))):
        cur = cur_placements[i]
        dst = dst_placements[i]

        if dst.is_shard():
            current_mesh_sharding, target_mesh_sharding = [], []
            for j, (s, p) in enumerate(zip(cur_placements, dst_placements)):
                if j >= i:
                    break
                if s.is_shard() and s.args['dim'] == dst.args['dim']:
                    current_mesh_sharding.append(j)
                if p.is_shard() and p.args['dim'] == dst.args['dim']:
                    target_mesh_sharding.append(j)

            if current_mesh_sharding != target_mesh_sharding:
                dst = SPMD(SPMD.REPLICATE)

            if cur != dst:
                transform_infos.append((i, cur, dst))
                cur_placements[i] = dst

    for i, (cur, dst) in enumerate(zip(cur_placements, dst_placements)):
        if cur != dst:
            transform_infos.append(
                (i, cur, dst)
            )
            cur_placements[i] = dst

    return transform_infos


def insert_comm_node(fx_module: torch.fx.GraphModule,
                     node,
                     var_,
                     src_specs: List[SPMD],
                     tgt_specs: List[SPMD],
                     copy_innode=None):

    spmd_mesh = get_device_mesh('spmd')
    spmd_axis_namelist = get_device_mesh()._binding['spmd']
    rank = spmd_mesh.get_rank()
    my_coordinate = spmd_mesh.get_coordinate()
    global_shape = var_.meta['val'].shape

    if mdconfig.experimental_p2p_sharding_transform:
        transform_infos, src_specs = _gen_immediate_transform_infos(src_specs, tgt_specs)
    else:
        transform_infos = _gen_transform_infos(src_specs, tgt_specs)

    for i, current, target in transform_infos:
        num_chunks = spmd_mesh.size(mesh_dim=i)
        spmd_axis_name = spmd_axis_namelist[i]

        submesh = get_device_mesh(spmd_axis_name)
        ranks = submesh.mesh.flatten().tolist()

        if current == target:
            continue

        with fx_module.graph.inserting_before(node):
            if target.is_shard():
                if current.is_replicate():
                    # insert split_tensor here
                    scatter_node = fx_module.graph.call_function(scatter_wrapper,
                                                                    args=(var_, num_chunks,
                                                                        target.args["dim"],
                                                                        my_coordinate[i]))

                    node.replace_input_with(var_, scatter_node)
                    var_ = scatter_node
                elif current.is_shard():
                    if current.args["dim"] != target.args["dim"]:
                        # all_to_all
                        all_to_all_start_node = fx_module.graph.call_function(
                            all_to_all_start,
                            args=(var_, current.args["dim"], target.args["dim"], num_chunks,
                                    my_coordinate[i], ranks))
                        all_to_all_end_node = fx_module.graph.call_function(
                            all_to_all_end,
                            args=(all_to_all_start_node, current.args["dim"], target.args["dim"],
                                    num_chunks, my_coordinate[i], ranks))

                        node.replace_input_with(var_, all_to_all_end_node)
                        var_ = all_to_all_end_node
                elif current.is_partial():
                    # reduce_scatter
                    reduceOp = reduce_map[current.args["ops"]]
                    # make sure contiguous
                    reduce_scatter_start_node = fx_module.graph.call_function(
                        reduce_scatter_start, args=(var_, reduceOp, target.args["dim"], ranks))
                    reduce_scatter_end_node = fx_module.graph.call_function(
                        reduce_scatter_end,
                        args=(reduce_scatter_start_node, reduceOp, target.args["dim"], ranks))

                    node.replace_input_with(var_, reduce_scatter_end_node)
                    var_ = reduce_scatter_end_node
            elif target.is_replicate():
                if current.is_shard():
                    # insert all_gather here
                    # make sure contiguous
                    all_gather_start_node = fx_module.graph.call_function(
                        all_gather_start, args=(var_, current.args["dim"], ranks))
                    all_gather_end_node = fx_module.graph.call_function(
                        all_gather_end, args=(all_gather_start_node, current.args["dim"], ranks))

                    node.replace_input_with(var_, all_gather_end_node)
                    var_ = all_gather_end_node
                elif current.is_partial():
                    # insert all_reduce here
                    reduceOp = reduce_map[current.args["ops"]]
                    all_reduce_start_node = fx_module.graph.call_function(all_reduce_start,
                                                                            args=(var_, reduceOp,
                                                                                ranks))
                    all_reduce_end_node = fx_module.graph.call_function(
                        all_reduce_end, args=(all_reduce_start_node, reduceOp, ranks))

                    node.replace_input_with(var_, all_reduce_end_node)
                    var_ = all_reduce_end_node

    if mdconfig.experimental_p2p_sharding_transform and src_specs != tgt_specs:
        with fx_module.graph.inserting_before(node):
            p2p_node = fx_module.graph.call_function(
                do_p2p_comm_wrapper(global_shape, src_specs, tgt_specs, rank),
                args=(var_,)
            )
            node.replace_input_with(var_, p2p_node)
            var_ = p2p_node

    if copy_innode is not None:
        with fx_module.graph.inserting_before(node):
            copy_node = fx_module.graph.call_function(copy_wrapper, args=(copy_innode, var_))
            node.replace_input_with(var_, copy_node)

    return fx_module

# some of this part from torch/distributed/_tensor/ops/view_ops.py
def override_args(node, invars_strategy):
    spmd_mesh = get_device_mesh('spmd')

    if node.target in view_op_map:
        global_in_shape = node.args[0].meta['val'].shape
        shape_argnum = 1

        input_dtensor_spec = [to_torch_spmd(i) for i in invars_strategy[0]]
        rules = view_op_map[node.target](global_in_shape, node.args[shape_argnum])

        (
            global_out_shape,
            shard_out,
            shardable_dims,
        ) = propagate_shape_and_sharding(
            input_dtensor_spec,
            tuple(global_in_shape),
            rules,
            tuple(spmd_mesh.mesh.shape),
        )

        if shard_out is None:  # no need to reshard
            shard_out = input_dtensor_spec
        local_out_shape = compute_local_shape(list(global_out_shape), spmd_mesh, shard_out)
        
        node.update_arg(shape_argnum, local_out_shape)


def sharding_transform(fx_module: torch.fx.GraphModule, opt_strategy, state_io_map):

    shard_env = {}

    spme_mesh = get_device_mesh('spmd')

    # (TODO) move this part out of this pass
    for node in fx_module.graph.nodes:
        if node.op == 'call_function' and 'val' not in node.meta:
            node.meta = create_meta_from_node(node)

    # the last element in fx_module output is the return tensor for user function
    # we need to make it replicate before
    num_return_value = fx_module._out_spec.children_specs[-1].num_leaves

    all_placeholder_node = {}

    for node in fx_module.graph.nodes:
        if node.op == 'placeholder':
            if node.name in opt_strategy:
                shard_env[node.name] = opt_strategy[node.name]['strategy'].out_strtg_group[0]
            else:
                # TODO: only support 2d device mesh here
                shard_env[node.name] = VarSPMDStrategy(*[SPMD(SPMD.REPLICATE) for _ in range(spme_mesh.ndim)])

            all_placeholder_node[node.name] = node
        elif node.op == 'call_function':

            # skip for create ops
            ops_name = _get_qualified_name(node.target)
            if node.target in CREATE_ATEN_OP:
                # TODO: only support 2d device mesh here
                shard_env[node.name] = VarSPMDStrategy(*[SPMD(SPMD.REPLICATE) for _ in range(spme_mesh.ndim)])
                continue

            if node.target == operator.getitem:
                shard_env[node.name] = shard_env[node.args[0].name][node.args[1]]
                opt_strategy[node.name] = {
                    'node':
                    node.name,
                    'strategy':
                    NodeSPMDStrategy(VarSPMDStrategyGroup(shard_env[node.name]),
                                     VarSPMDStrategyGroup(shard_env[node.name]))
                }
                continue

            node_args_flatten = pytree.tree_flatten(node.args)[0]
            invars = [arg for arg in node_args_flatten if isinstance(arg, Node)]

            invars_strategy = opt_strategy[node.name]['strategy'].in_strtg_group

            override_args(node, invars_strategy)

            assert len(invars) == len(invars_strategy)

            visited = set()
            for var_, tgt_specs in zip(invars, invars_strategy):
                if var_ in visited:  # same var may appear multiple times but only need to process once
                    continue
                visited.add(var_)
                src_specs = shard_env[var_.name]
                if tgt_specs != src_specs:
                    # mismatch need to insert communication node
                    fx_module = insert_comm_node(fx_module, node, var_, src_specs, tgt_specs)

            shard_env[node.name] = opt_strategy[node.name]['strategy'].out_strtg_group
            if len(shard_env[node.name]) == 1 and node.target not in ANNOTATION_OPS:  # ANNOTATION_OPS returns tensor list, no need to unwrap
                shard_env[node.name] = shard_env[node.name][0]

        elif node.op == 'output':
            need_replicate_node = node.args[0][-1 * num_return_value:]
            need_replicate_node = [i for i in need_replicate_node if i is not None]
            for o_node in need_replicate_node:
                src_specs = shard_env[o_node.name]
                tgt_specs = [SPMD(SPMD.REPLICATE)] * len(src_specs)
                tgt_specs = VarSPMDStrategy(*tgt_specs)
                if tgt_specs != src_specs:
                    # mismatch need to insert communication node
                    fx_module = insert_comm_node(fx_module, node, o_node, src_specs, tgt_specs)

            for in_node, out_node in state_io_map.items():
                if in_node.name in shard_env:
                    src_specs = shard_env[out_node.name]
                    tgt_specs = shard_env[in_node.name]
                    o_node = None
                    for node_ in node.args[0]:
                        if node_ is not None and node_.name == out_node.name:
                            o_node = node_
                            break
                    assert o_node is not None
                    if tgt_specs != src_specs:
                        # mismatch need to insert communication node
                        fx_module = insert_comm_node(
                            fx_module,
                            node,
                            o_node,
                            src_specs,
                            tgt_specs,
                            copy_innode=all_placeholder_node[in_node.name])

    fx_module.recompile()

    # (TODO) move this part out of this pass
    for node in fx_module.graph.nodes:
        if not hasattr(node, "ed_info"):
            node.ed_info = EDInfo(ori_meta=node.meta)

        if node.name in opt_strategy:
            node.ed_info.strategy = opt_strategy[node.name]['strategy']

        node.meta = node.ed_info.get_sharded_meta()

        if node.op == 'placeholder':
            node.ed_info.node_type = EDNodeType.AUXILIARY
        elif node.op == 'call_function':
            # create meta for custom function
            # if node.target in CUSTOM_FUNCS:
            node.meta = create_meta_from_node(node)
            # annotate node type
            if node.target in COMM_FUNCS:
                node.ed_info.node_type = EDNodeType.COMMUNICATION
            elif node.target == operator.getitem:
                node.ed_info.node_type = EDNodeType.AUXILIARY
            else:
                node.ed_info.node_type = EDNodeType.COMPUTATION
        elif node.op == 'output':
            node.ed_info.node_type = EDNodeType.AUXILIARY

    return fx_module
