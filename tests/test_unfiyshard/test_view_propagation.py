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

import pytest

from easydist.metashard import ShardAnnotation, ShardDim
from easydist.metashard.combination import CombinationFunc
from easydist.metashard.view_propagation import view_propagation_preset
from easydist.utils.testing.mock import assert_partial_func_equal

@pytest.mark.torch
def test_view_propagation_preset():
    preset_anno = ShardAnnotation([[ShardDim(1, chunk=5), ShardDim(0)]])
    comb_func = view_propagation_preset([10, 8], [5, 2, 8], preset_anno)
    answer = functools.partial(CombinationFunc.gather, dim=1)
    assert_partial_func_equal(comb_func, answer)

    preset_anno = ShardAnnotation([[ShardDim(0), ShardDim(1, chunk=2)]])
    comb_func = view_propagation_preset([10, 8], [10, 2, 2, 2], preset_anno)
    answer = functools.partial(CombinationFunc.gather, dim=2)
    assert_partial_func_equal(comb_func, answer)

    preset_anno = ShardAnnotation([[ShardDim(0), ShardDim(1, chunk=4)]])
    comb_func = view_propagation_preset([10, 8], [10, 2, 2, 2], preset_anno)
    answer = functools.partial(CombinationFunc.gather, dim=3)
    assert_partial_func_equal(comb_func, answer)

    preset_anno = ShardAnnotation([[ShardDim(1, chunk=3), ShardDim(0)]])
    comb_func = view_propagation_preset([10, 8], [5, 2, 8], preset_anno)
    assert comb_func is None

    preset_anno = ShardAnnotation([[ShardDim(0), ShardDim(1, chunk=2)]])
    comb_func = view_propagation_preset([10, 8], [5, 2, 8], preset_anno)
    assert comb_func is None
