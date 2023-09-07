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

import functools
from enum import Enum
import logging

from metadist import platform
import metadist.config as mdconfig

logger = logging.getLogger(__name__)

TRY_COMBINATION_FUNC = dict()


def try_combination_func(func):
    """Register a function as a try_combination_func"""
    TRY_COMBINATION_FUNC[func.__name__] = func
    return func


class ReduceOp(Enum):
    SUM = 1
    MAX = 2
    MIN = 3


class HaloHint:

    def __init__(self, halo, dim, idx_=None):
        self.halo = halo
        self.dim = dim
        self.out_idx = idx_


def aligned_prefix(t1: platform.Tensor, t2: platform.Tensor, dim_idx: int):
    """calculate the max prefix for t1 and t2 on dimension(dim_idx)

    For example: aligned_prefix([1, 2, 3], [1, 2, 5]) -> 2
    """
    dim_size = min(t1.shape[dim_idx], t2.shape[dim_idx])
    for i in range(1, dim_size + 1):
        if not platform.allclose(platform.narrow(t1, dim_idx, 0, i),
                                 platform.narrow(t2, dim_idx, 0, i)):
            return i - 1
    return i


def shape_aligned_otherdim(shape_1, shape_2, dim_idx):
    """return True when shape_1 and shape_2 only different on dimension(dim_idx)"""
    if len(shape_1) != len(shape_2):
        return False

    diff_dim = []
    for idx in range(len(shape_1)):
        if shape_1[idx] != shape_2[idx]:
            diff_dim.append(idx)

    if diff_dim == [dim_idx]:
        return True
    return False


class CombinationFunc:

    @staticmethod
    def identity(sharded_tensor):
        identity_tensor = sharded_tensor[0]
        for tensor_ in sharded_tensor:
            if not platform.equal(identity_tensor, tensor_):
                logger.debug("not all tensor same as identity")
                return None
        return identity_tensor

    @staticmethod
    def reduce(sharded_tensor, ops=ReduceOp.SUM):
        init = platform.zeros_like(sharded_tensor[0])
        assert ops in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN]
        if ops == ReduceOp.SUM:
            reduce_func_ = platform.add
        elif ops == ReduceOp.MAX:
            reduce_func_ = platform.max
        elif ops == ReduceOp.MIN:
            reduce_func_ = platform.min
        return functools.reduce(reduce_func_, sharded_tensor, init)

    @staticmethod
    def gather(sharded_tensor, dim, halowidth=0, chunk=1):
        if halowidth == 0:
            if chunk == 1:
                return platform.concatenate(sharded_tensor, dim=dim)
            else:
                chunk_sharded_tensor = [platform.chunk(t, chunk, dim) for t in sharded_tensor]
                reorder_list = []
                for chunk_idx in range(chunk):
                    reorder_list += [chunk_list[chunk_idx] for chunk_list in chunk_sharded_tensor]
                return platform.concatenate(reorder_list, dim=dim)

        shard_world_size = len(sharded_tensor)
        gathered_tensor = sharded_tensor[0]
        for idx_ in range(1, shard_world_size):
            width_tensor_1 = int(gathered_tensor.shape[dim])
            width_tensor_2 = int(sharded_tensor[idx_].shape[dim])
            if halowidth > 0:
                gathered_tensor = platform.concatenate([
                    platform.narrow(gathered_tensor, dim, 0, width_tensor_1 - halowidth),
                    platform.add(
                        platform.narrow(gathered_tensor, dim, width_tensor_1 - halowidth,
                                        halowidth),
                        platform.narrow(sharded_tensor[idx_], dim, 0, halowidth)),
                    platform.narrow(sharded_tensor[idx_], dim, halowidth,
                                    width_tensor_2 - halowidth)
                ],
                                                       dim=dim)
            else:
                gathered_tensor = platform.concatenate([
                    platform.narrow(gathered_tensor, dim, 0, width_tensor_1 + halowidth),
                    platform.narrow(sharded_tensor[idx_], dim, -1 * halowidth,
                                    width_tensor_2 + halowidth)
                ],
                                                       dim=dim)

        return gathered_tensor


@try_combination_func
def try_combination_identity(sharded_tensor_, global_tensor_):
    """try combination through identity, return combination func if success, otherwise None"""

    for t_ in sharded_tensor_:
        if t_.shape != global_tensor_.shape:
            return None

    result = CombinationFunc.identity(sharded_tensor_)
    if result is not None and platform.allclose(result, global_tensor_):
        return functools.partial(CombinationFunc.identity)

    return None


@try_combination_func
def try_combination_reduce(sharded_tensor_, global_tensor_):
    """try combination through reduce, return combination func if success, otherwise None"""

    for t_ in sharded_tensor_:
        if t_.shape != global_tensor_.shape:
            return None

    for reduce_op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN]:
        reduce_func = functools.partial(CombinationFunc.reduce, ops=reduce_op)
        if platform.allclose(reduce_func(sharded_tensor_), global_tensor_):
            return reduce_func

    return None


@try_combination_func
def try_combination_gather(sharded_output_, global_output_):
    """try combination through gather, return combination func if success, otherwise None"""

    # 1) all sharded tensor shape should be equal to global tensor except the gather_dim
    sharded_output_shape = sharded_output_[0].shape
    global_output_shape = global_output_.shape

    # return if global_output is only one element.
    if len(global_output_shape) == 0:
        return None

    shard_size = len(sharded_output_)

    for dim_idx in range(len(sharded_output_shape)):
        if sharded_output_shape[dim_idx] != global_output_shape[dim_idx]:
            break

    for tensor_ in sharded_output_:
        if not shape_aligned_otherdim(tensor_.shape, global_output_shape, dim_idx):
            return None

    combination_size = 0
    for out_ in sharded_output_:
        combination_size += out_.shape[dim_idx]
    gap_size = combination_size - global_output_shape[dim_idx]
    gather_func = None
    if gap_size == 0:
        gather_func = functools.partial(CombinationFunc.gather, dim=dim_idx)
        gathered_tensor = gather_func(sharded_output_)
        if gathered_tensor.shape == global_output_.shape:
            if platform.allclose(gathered_tensor, global_output_):
                return gather_func
            if mdconfig.extend_space:
                reference_shard_output = platform.chunk(global_output_,
                                                        chunks=shard_size,
                                                        dim=dim_idx)[0]
                aligned_prefix_len = aligned_prefix(sharded_output_[0], reference_shard_output,
                                                    dim_idx)

                # explore gather with chunk > 1
                if aligned_prefix_len != 0 and sharded_output_shape[
                        dim_idx] % aligned_prefix_len == 0:
                    guess_chunk = sharded_output_shape[dim_idx] // aligned_prefix_len
                    gather_func = functools.partial(CombinationFunc.gather,
                                                    dim=dim_idx,
                                                    chunk=guess_chunk)
                    gathered_tensor_chunk = gather_func(sharded_output_)
                    if platform.allclose(gathered_tensor_chunk, global_output_):
                        return gather_func

                # return HaloHint to explore the chance of halo sharding
                if aligned_prefix_len > (sharded_output_shape[dim_idx] // 2):
                    return HaloHint(sharded_output_shape[dim_idx] - aligned_prefix_len, dim_idx)

    if mdconfig.extend_space:
        # halo gather with halowidth > 0
        if gap_size > 0 and gap_size % (shard_size - 1) == 0:
            halowidth = int(gap_size // (shard_size - 1))
            if halowidth >= combination_size // shard_size:
                return None
            gather_func = functools.partial(CombinationFunc.gather,
                                            dim=dim_idx,
                                            halowidth=halowidth)
            gathered_tensor = gather_func(sharded_output_)
            if gathered_tensor.shape == global_output_.shape:
                if platform.allclose(gathered_tensor, global_output_):
                    return gather_func

        # halo gather with halowidth < 0
        if gap_size > 0 and gap_size % shard_size == 0:
            halowidth = -1 * int(gap_size // shard_size)
            if -1 * halowidth >= combination_size // (2 * shard_size):
                return None
            gather_func = functools.partial(CombinationFunc.gather,
                                            dim=dim_idx,
                                            halowidth=halowidth)
            gathered_tensor = gather_func(sharded_output_)
            if gathered_tensor.shape == global_output_.shape:
                if platform.allclose(gathered_tensor, global_output_):
                    return gather_func

        # raise HaloHint when convolution without padding
        if gap_size < 0 and gap_size % (shard_size - 1) == 0:
            halowidth = int(gap_size // (shard_size - 1)) // 2
            if -1 * halowidth >= combination_size // shard_size:
                return None
            return HaloHint(halowidth, dim_idx)


def try_combination_single(sharded_output_, global_output_):

    # check all sharded tensor have equal dimension of global_output
    for sharded_tensor in sharded_output_:
        if len(sharded_tensor.shape) != len(global_output_.shape):
            return None

    for func_name in TRY_COMBINATION_FUNC:
        combination_func = TRY_COMBINATION_FUNC[func_name](sharded_output_, global_output_)
        if combination_func:
            return combination_func

    return None


def try_combination(sharded_output_, global_output_):

    if isinstance(global_output_, platform.Tensor):
        return try_combination_single(sharded_output_, global_output_)

    if isinstance(global_output_, tuple) or isinstance(global_output_, list):
        output_num = [len(i) for i in sharded_output_]
        if len(global_output_) == min(output_num) == max(output_num):
            return_combination_ann = []
            for idx_ in range(len(global_output_)):
                if isinstance(global_output_[idx_], platform.Tensor):
                    single_ann = try_combination_single([i[idx_] for i in sharded_output_],
                                                        global_output_[idx_])
                    if single_ann is None:
                        return None
                    if isinstance(single_ann, HaloHint):
                        single_ann.out_idx = idx_
                        return single_ann
                    return_combination_ann.append(single_ann)
                else:
                    for out_ in sharded_output_:
                        if global_output_[idx_] != out_[idx_]:
                            return None

            if len(return_combination_ann) > 0:
                return return_combination_ann

    return None
