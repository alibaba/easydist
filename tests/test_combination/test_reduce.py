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
from easydist.metashard.combination import CombinationFunc, ReduceOp
from easydist.utils.testing import ALL_PLATFORM
from easydist import easydist_setup


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_reduce(backend):
    easydist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.array([i, i, i])) for i in range(4)]
    max_tensor = platform.from_numpy(numpy.array([3, 3, 3]))
    combination_max = CombinationFunc.reduce(shard_tensor, ops=ReduceOp.MAX)

    assert platform.allclose(max_tensor, combination_max)

    min_tensor = platform.from_numpy(numpy.array([0, 0, 0]))
    combination_min = CombinationFunc.reduce(shard_tensor, ops=ReduceOp.MIN)

    assert platform.allclose(min_tensor, combination_min)

    sum_tensor = platform.from_numpy(numpy.array([6, 6, 6]))
    combination_sum = CombinationFunc.reduce(shard_tensor, ops=ReduceOp.SUM)

    assert platform.allclose(sum_tensor, combination_sum)
