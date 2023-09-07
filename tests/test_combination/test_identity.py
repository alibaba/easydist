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

from metadist import platform
from metadist.metashard.combination import CombinationFunc
from metadist.utils.testing import ALL_PLATFORM
from metadist import metadist_setup


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_identity(backend):
    metadist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.array([1, 2, 3]))] * 4
    global_tensor = platform.from_numpy(numpy.array([1, 2, 3]))
    combination_tensor = CombinationFunc.identity(shard_tensor)

    assert platform.allclose(global_tensor, combination_tensor)


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_identity_2(backend):
    metadist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.array([1, 2, 3]))] * 4
    global_tensor = platform.from_numpy(numpy.array([1, 2, 4]))
    combination_tensor = CombinationFunc.identity(shard_tensor)

    assert not platform.allclose(global_tensor, combination_tensor)


@pytest.mark.parametrize("backend", ALL_PLATFORM)
def test_identity_3(backend):
    metadist_setup(backend)
    shard_tensor = [platform.from_numpy(numpy.array([1, 2, 3]))] * 3 + [
        platform.from_numpy(numpy.array([1, 2, 4]))
    ]
    combination_tensor = CombinationFunc.identity(shard_tensor)

    assert combination_tensor is None
