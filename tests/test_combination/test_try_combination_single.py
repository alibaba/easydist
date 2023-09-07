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

import numpy
import pytest
import functools

from metadist import platform
import metadist.config as mdconfig
from metadist.metashard.combination import CombinationFunc, ReduceOp, try_combination_single
from metadist.utils.testing import ALL_PLATFORM, assert_partial_func_equal
from metadist import metadist_setup


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_reduce(backend):
    metadist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.random.uniform(size=(3, 4))) for _ in range(4)]

    for op_type in [ReduceOp.MAX, ReduceOp.MIN, ReduceOp.SUM]:
        comb_func = functools.partial(CombinationFunc.reduce, ops=op_type)
        global_tensor = comb_func(shard_tensor)

        return_func = try_combination_single(shard_tensor, global_tensor)

        assert_partial_func_equal(comb_func, return_func)


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_gather(backend):
    metadist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.random.uniform(size=(3, 4))) for _ in range(4)]

    for dim_ in [0, 1]:
        comb_func = functools.partial(CombinationFunc.gather, dim=dim_)
        global_tensor = comb_func(shard_tensor)

        return_func = try_combination_single(shard_tensor, global_tensor)

        assert_partial_func_equal(comb_func, return_func)


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_gather_halo(backend):
    metadist_setup(backend)
    mdconfig.extend_space = True

    shard_tensor = [platform.from_numpy(numpy.random.uniform(size=(3, 4))) for _ in range(3)]

    for dim_, halo_ in zip([0, 1], [1, 2]):
        comb_func = functools.partial(CombinationFunc.gather, dim=dim_, halowidth=halo_)
        global_tensor = comb_func(shard_tensor)

        return_func = try_combination_single(shard_tensor, global_tensor)

        assert_partial_func_equal(comb_func, return_func)


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_gather_chunk(backend):
    metadist_setup(backend)
    mdconfig.extend_space = True

    shard_tensor = [platform.from_numpy(numpy.random.uniform(size=(3, 4))) for _ in range(3)]

    for dim_, chunk_ in zip([0, 1], [3, 2]):
        comb_func = functools.partial(CombinationFunc.gather, dim=dim_, chunk=chunk_)
        global_tensor = comb_func(shard_tensor)

        return_func = try_combination_single(shard_tensor, global_tensor)

        assert_partial_func_equal(comb_func, return_func)
