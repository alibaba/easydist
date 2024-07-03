# torchrun --nproc_per_node 4 --master-port 12345 pseudo.py

import os
import random

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import combinations
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import numpy as np

import torch
import torch.distributed as dist

from torch import Tensor
from torch.distributed._tensor import (DeviceMesh, Replicate, Shard, distribute_tensor)
from torch.distributed._tensor.placement_types import _Partial
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states

from easydist.torch.passes.sharding import (all_gather_end, all_gather_start, all_reduce_end,
                                            all_reduce_start)

LOG = False


class DummyFile:

    def write(*args, **kwargs):
        pass

    def flush(*args, **kwargs):
        pass


rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device_mesh_global = DeviceMesh(mesh=torch.arange(4).reshape(2, 2), device_type='cpu')
device_mesh_1d = DeviceMesh(mesh=device_mesh_global.mesh.flatten(), device_type='cpu')

def reduce(self: torch.Tensor, reduceOp: str, group: List[int], tag: str = ""):
    t = all_reduce_start(self, reduceOp, group, tag)
    return all_reduce_end(t, reduceOp, group, tag)


def reduce_sum(self: torch.Tensor, group: List[int], tag: str = ""):
    return reduce(self, "sum", group, tag)


def reduce_max(self: torch.Tensor, group: List[int], tag: str = ""):
    return reduce(self, "max", group, tag)


def reduce_min(self: torch.Tensor, group: List[int], tag: str = ""):
    return reduce(self, "min", group, tag)


def gather(self: torch.Tensor, gather_dim: int, group: List[int], tag: str = ""):
    t = all_gather_start(self, gather_dim, group, tag)
    return all_gather_end(t, gather_dim, group, tag)

def identity(self: torch.Tensor, group: List[int]):
    return self

def getDevAnn(comb_func: partial,
              gather_dim=None) -> Union[Shard, Replicate]:
    if comb_func.func == identity:
        assert gather_dim is None
        return Replicate()
    elif comb_func.func == reduce_sum:
        assert gather_dim is None
        return _Partial(reduce_op=dist.ReduceOp.SUM)
    elif comb_func.func == reduce_max:
        assert gather_dim is None
        return _Partial(reduce_op=dist.ReduceOp.MAX)
    elif comb_func.func == reduce_min:
        assert gather_dim is None
        return _Partial(reduce_op=dist.ReduceOp.MIN)
    else:
        assert gather_dim is not None and comb_func.func == gather
        return Shard(gather_dim)

class TensorDimAnn:

    def __init__(self, device_dims: Optional[List[int]] = None):
        if isinstance(device_dims, tuple):
            device_dims = list(device_dims)
        self.device_dims = device_dims

    def __repr__(self):
        return "S(" + "".join([str(d)
                               for d in self.device_dims]) + ")" if self.device_dims else "R"

    def __add__(self, other):
        if self.device_dims is None:
            return other
        if other.device_dims is None:
            return self
        return TensorDimAnn(self.device_dims + other.device_dims)

TensorAnn = List[TensorDimAnn]
DeviceAnn = List[Union[Replicate, Shard]]
CombSeq = List[Callable]

class Constant:

    def __repr__(self):
        return "C"

def getLocalTensor(tensor: Tensor, anns: TensorAnn, device_mesh: DeviceMesh) -> Tensor:

    def to_dtensor_ann(anns: List[TensorDimAnn]):
        ret = [Replicate() for _ in range(len(device_mesh.mesh.shape))]
        for tensor_dim, ann in enumerate(anns):
            if ann.device_dims:
                for device_dim in ann.device_dims:
                    ret[device_dim] = Shard(tensor_dim)
        return ret

    dtensor = distribute_tensor(tensor, device_mesh, to_dtensor_ann(anns))
    return dtensor._local_tensor


def genDimComb(dims: Set[int]) -> List[Tuple[int]]:
    return [x for xx in [combinations(dims, i) for i in range(1, len(dims) + 1)] for x in xx]


def search1DShardSingle(tensor: Tensor) -> List[TensorAnn]:
    ans = []

    def _searchShardSingle(left_dims: Set[int], tensor: Tensor, cur_dim: int,
                           cur_shard: List[TensorDimAnn]):
        if cur_dim >= tensor.dim() or len(left_dims) == 0:
            # if len(left_dims) == 0:
            cur_shard_cp = deepcopy(cur_shard)
            for i in range(cur_dim, tensor.dim()):
                cur_shard_cp.append(TensorDimAnn())
            ans.append(cur_shard_cp)
            return

        # shard in cur_dim
        for comb in genDimComb(left_dims):
            _searchShardSingle(left_dims - set(comb), tensor, cur_dim + 1,
                               cur_shard + [TensorDimAnn(comb)])

        # no shard
        _searchShardSingle(left_dims, tensor, cur_dim + 1, cur_shard + [TensorDimAnn()])

    _searchShardSingle(set(i for i in range(device_mesh_1d.mesh.dim())), tensor, 0, [])

    return ans


def getRanks(device_dims: Tuple[int], device_mesh: DeviceMesh) -> Tuple[int]:
    local_coord = deepcopy(device_mesh.get_coordinate())
    for dim in device_dims:
        local_coord[dim] = slice(None)
    return device_mesh.mesh[tuple(local_coord)].flatten().tolist()


def getCombFuncs(t: Tensor) -> List[partial]:
    base_funcs = [partial(identity), partial(reduce_sum), partial(reduce_max), partial(reduce_min)]
    for i in range(t.dim()):
        base_funcs.append(partial(gather, gather_dim=i))
    return base_funcs


def search1DCombSingle(comb_funcs: List[partial]) -> List[Tuple[CombSeq, DeviceAnn]]:
    ans = []

    def _searchCombSingle(left_dims: Set[int], cur_comb: CombSeq, cur_dev: DeviceAnn,
                          prev_comb: Optional[partial]):
        if len(left_dims) == 0:
            ans.append((deepcopy(cur_comb), deepcopy(cur_dev)))
            return

        for dims in genDimComb(left_dims):
            for comb_func in comb_funcs:
                ranks = getRanks(dims, device_mesh_1d)
                fn = partial(comb_func, group=ranks)
                cur_dev_cp = deepcopy(cur_dev)
                for dim in dims:
                    dev_ann = getDevAnn(fn,
                                        gather_dim=fn.keywords['gather_dim']
                                        if 'gather_dim' in fn.keywords else None)
                    cur_dev_cp[dim] = dev_ann
                _searchCombSingle(left_dims - set(dims), cur_comb + [fn], cur_dev_cp, comb_func)

    _searchCombSingle(set(i for i in range(device_mesh_1d.mesh.dim())), [],
                      [Replicate() for _ in range(device_mesh_1d.mesh.dim())], None)
    return ans


def search1DSolution(op,
                   op_args: List[Any],
                   args_anns: List[Union[TensorAnn, Constant]],
                   quick_sol=True) -> List[Tuple[TensorAnn, CombSeq, DeviceAnn]]:
    global_output = op(*op_args)
    assert isinstance(global_output, Tensor), "only support single output for now"
    op_legit_anns_and_comb = []
    handle_all_replicate_flag = False

    for i, args_ann in enumerate(args_anns):
        with open(f"rank{rank}.txt", "w") if LOG else nullcontext(DummyFile()) as f:
            f.write(f"{rank=} {i=}\n")

            local_op_args = []
            tensor_ann = []

            for arg, ann in zip(op_args, args_ann):
                if isinstance(ann, Constant):
                    local_op_args.append(deepcopy(arg))
                    continue
                local_op_args.append(getLocalTensor(arg, ann, device_mesh_1d))
                tensor_ann.append(ann)

            def get_global_flag(local_flag: float):
                scalar_t = torch.tensor(local_flag)
                dist.all_reduce(scalar_t, op=dist.ReduceOp.SUM)
                return scalar_t.item()

            if all(dim_ann.device_dims is None for ann in tensor_ann for dim_ann in ann):
                if handle_all_replicate_flag:
                    continue
                handle_all_replicate_flag = True
                all_dim = [i for i in range(device_mesh_1d.mesh.dim())]
                op_legit_anns_and_comb.append(
                    (args_ann, [partial(identity, group=getRanks(all_dim, device_mesh_1d))],
                     [Replicate() for _ in range(device_mesh_1d.mesh.dim())]))
                continue

            # test op is executable or not (pre-prune)
            local_flag = 1.0
            try:
                local_output = op(*local_op_args)
            except Exception as e:
                if rank == 0:
                    print(str(e))
                local_flag = 0.0

            if get_global_flag(local_flag) != world_size:
                continue

            assert isinstance(local_output, Tensor), "only support single output for now"

            possible_combination = search1DCombSingle(getCombFuncs(local_output))

            for comb_seq in possible_combination:
                f.write(str(comb_seq) + "\n")

            for j, (comb_seq, output_ann) in enumerate(possible_combination):
                f.write(f"{j=}\n")
                cur_output = None
                for k, comb_func in enumerate(comb_seq):
                    f.write(f"\t{k=} {comb_func}\n")
                    f.flush()

                    local_flag = 1.0
                    try:
                        if cur_output is None:
                            cur_output = comb_func(local_output)
                        else:
                            cur_output = comb_func(cur_output)
                    except Exception as e:
                        if rank == 0:
                            print(str(e))
                        local_flag = 0.0

                    if get_global_flag(local_flag) != world_size:
                        break

                if cur_output is not None and cur_output.shape == global_output.shape and torch.allclose(
                        cur_output, global_output):
                    op_legit_anns_and_comb.append((args_ann, comb_seq, output_ann))
                    if quick_sol:
                        break

    return op_legit_anns_and_comb

def searchNDSolution(solutions_1d: List[Tuple[TensorAnn, CombSeq, DeviceAnn]]) -> List[Tuple[TensorAnn, CombSeq, DeviceAnn]]:
    ans = []

    def _flattenSolutions(cur_sol: List[Tuple[TensorAnn, CombSeq, DeviceAnn]]) -> Tuple[TensorAnn, CombSeq, DeviceAnn]:
        def _add_args_ann(cur, args_ann):
            cur = deepcopy(cur)
            for cur_arg_ann, arg_ann in zip(cur, args_ann):
                for i, (cur_dim_ann, dim_ann) in enumerate(zip(cur_arg_ann, arg_ann)):
                    cur_arg_ann[i] = cur_dim_ann + dim_ann
            return cur

        def _add_comb_seq(comb_seq1, comb_seq2):
            return comb_seq1 + comb_seq2

        assert len(cur_sol) == device_mesh_global.mesh.dim()

        flatten_sol, *left_sol = cur_sol
        flatten_sol = deepcopy(flatten_sol)
        flatten_args_ann, flatten_comb_seq, flatten_dev_ann = flatten_sol
        assert len(flatten_comb_seq) == 1
        flatten_comb_seq[0].keywords['group'] = getRanks((0, ), device_mesh_global)

        cur_device_dim = 1
        for tmp in left_sol:
            args_ann, comb_seq, _ = deepcopy(tmp)
            for arg_ann in args_ann:
                for dim_ann in arg_ann:
                    if dim_ann.device_dims is not None:
                        assert len(dim_ann.device_dims) == 1, f"{dim_ann=}"
                        dim_ann.device_dims = [cur_device_dim]
            flatten_args_ann = _add_args_ann(flatten_args_ann, args_ann)
            assert len(comb_seq) == 1
            comb_seq[0].keywords['group'] = getRanks((cur_device_dim, ), device_mesh_global)
            flatten_comb_seq = _add_comb_seq(flatten_comb_seq, comb_seq)
            cur_device_dim += 1

        if rank == 0:
            print(f"{flatten_args_ann=} {flatten_comb_seq=}")

        flatten_dev_ann = [getDevAnn(comb_func, gather_dim= comb_func.keywords['gather_dim'] if 'gather_dim' in comb_func.keywords else None) for comb_func in flatten_comb_seq]

        return flatten_args_ann, flatten_comb_seq, flatten_dev_ann


    def _searchNDSolution(device_dim: int, cur_sol: List):
        if device_dim >= device_mesh_global.mesh.dim():
            ans.append(_flattenSolutions(cur_sol))
            return

        for sol in solutions_1d:
            cur_sol.append(sol)
            _searchNDSolution(device_dim + 1, cur_sol)
            cur_sol.pop()

    _searchNDSolution(0, [])
    return ans


def searchShard(arg_ann: List[Union[List[TensorDimAnn], Constant]]):
    tensor_ann = []
    for ann_comb in arg_ann:
        if not isinstance(ann_comb, Constant):
            tensor_ann.append(ann_comb)

    constant = [i for i, ann in enumerate(arg_ann) if isinstance(ann, Constant)]
    ans = []

    def _searchShard(i, cur):
        if i >= len(tensor_ann):
            ans.append(deepcopy(cur))
            return

        for j in range(len(tensor_ann[i])):
            cur.append(tensor_ann[i][j])
            _searchShard(i + 1, cur)
            cur.pop()

    _searchShard(0, [])

    for a in ans:
        for c in constant:
            a.insert(c, Constant())

    return ans


def test_single_tensor_shard():
    tensor = torch.ones(16, 4, 8)
    shardAnn = search1DShardSingle(tensor)
    if rank == 0:
        print(len(shardAnn))
        for ann in shardAnn:
            print(ann)

@torch.no_grad
def test_matmul():
    tensor1 = torch.rand(32, 16, 8)
    tensor2 = torch.rand(32, 8, 16)
    shardAnn1 = search1DShardSingle(tensor1)
    shardAnn2 = search1DShardSingle(tensor2)
    args_combs = searchShard([shardAnn1, shardAnn2])
    ans = search1DSolution(torch.matmul, [tensor1, tensor2], args_combs)
    ans = searchNDSolution(ans)
    if rank == 0:
        with open(f'device{tuple(device_mesh_global.mesh.shape)}-matmul_sol.txt', 'w') as f:
            f.write(f"ans len: {len(ans)}\n")
            for x in ans:
                f.write(f"input spec: {str(x[0]):<40}output DTensor spec: {str(x[2]):<60}\n")


@torch.no_grad
def test_conv():
    layer = torch.nn.Conv2d(32, 64, 8, stride=1)
    tensor = torch.rand(16, 32, 32, 32)
    weight, bias, stride, padding, dilation, groups = layer.weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups
    tensorAnn = search1DShardSingle(tensor)
    weightAnn = search1DShardSingle(weight)
    biasAnn = search1DShardSingle(bias)

    args_combs = searchShard(
        [tensorAnn, weightAnn, biasAnn,
         Constant(), Constant(),
         Constant(), Constant()])

    ans = search1DSolution(torch.conv2d, [tensor, weight, bias, stride, padding, dilation, groups],
                         args_combs)
    ans = searchNDSolution(ans)
    if rank == 0:
        with open(f'device{tuple(device_mesh_global.mesh.shape)}-conv_sol.txt', 'w') as f:
            f.write(f"ans len: {len(ans)}\n")
            for x in ans:
                f.write(f"input spec: {str(x[0]):<60}output DTensor spec: {str(x[2]):<60}, combine funcs: {str(x[1])}\n")



def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for built-in Python
    random.seed(seed)
    # Set(seed) for each of the random number generators in python:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed()
    test_matmul()
    # test_conv()