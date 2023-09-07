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

import torch

from metadist.metashard.combination import CombinationFunc
from metadist.metashard import ShardAnnotation, ShardDim, MetaOp
from metadist.utils.testing import assert_partial_func_equal
from metadist import metadist_setup


def test_metaop_preset():
    metadist_setup("torch")
    input_args = (torch.rand((3, 4, 768)), 3, 2), {}
    meta_op = MetaOp(torch.ops.aten.chunk, input_args)
    preset_anno = ShardAnnotation([[ShardDim(0), ShardDim(0), ShardDim(1, chunk=3)]])
    comb_func = meta_op.sharding_discovery_with_preset(preset_anno)

    right_answer = [functools.partial(CombinationFunc.gather, dim=2)] * 3

    assert comb_func != None
    assert len(comb_func) == len(right_answer)

    for func1, func2 in zip(comb_func, right_answer):
        assert_partial_func_equal(func1, func2)
