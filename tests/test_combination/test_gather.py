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

from easydist import platform
from easydist.metashard.combination import CombinationFunc
from easydist.utils.testing import ALL_PLATFORM
from easydist import easydist_setup


@pytest.mark.all_platform
@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_gather(backend):
    easydist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.ones((3, 4)))] * 4
    global_tensor_1 = platform.from_numpy(numpy.ones((12, 4)))
    gather_dim1 = CombinationFunc.gather(shard_tensor, dim=0)

    assert platform.allclose(global_tensor_1, gather_dim1)

    global_tensor_2 = platform.from_numpy(numpy.ones((3, 16)))
    gather_dim2 = CombinationFunc.gather(shard_tensor, dim=1)

    assert platform.allclose(global_tensor_2, gather_dim2)


@pytest.mark.all_platform
@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_gather_halo(backend):
    easydist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.array([1, 1, 1]))] * 3
    global_tensor_1 = platform.from_numpy(numpy.array([1, 1, 2, 1, 2, 1, 1]))
    gather_halo_1 = CombinationFunc.gather(shard_tensor, dim=0, halowidth=1)

    assert platform.allclose(global_tensor_1, gather_halo_1)

    global_tensor_2 = platform.from_numpy(numpy.array([1, 1, 1, 1, 1]))
    gather_halo_2 = CombinationFunc.gather(shard_tensor, dim=0, halowidth=-1)

    assert platform.allclose(global_tensor_2, gather_halo_2)


@pytest.mark.all_platform
@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_gather_chunk(backend):
    easydist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.array([1, 2, 3]))] * 3
    global_tensor = platform.from_numpy(numpy.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
    gather_chunk = CombinationFunc.gather(shard_tensor, dim=0, chunk=3)

    assert platform.allclose(global_tensor, gather_chunk)
