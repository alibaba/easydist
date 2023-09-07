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
from metadist import platform
from metadist.metashard.combination import aligned_prefix, shape_aligned_otherdim


def test_aligned_prefix():
    t1 = platform.from_numpy(numpy.array([1, 2, 3, 4]))
    t2 = platform.from_numpy(numpy.array([1, 2, 3, 4]))
    assert 4 == aligned_prefix(t1, t2, dim_idx=0)

    t1 = platform.from_numpy(numpy.array([1, 2, 3, 4]))
    t2 = platform.from_numpy(numpy.array([2, 2, 3, 4]))
    assert 0 == aligned_prefix(t1, t2, dim_idx=0)

    t1 = platform.from_numpy(numpy.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
    t2 = platform.from_numpy(numpy.array([[1, 2, 3, 4], [1, 2, 3, 5]]))
    assert 1 == aligned_prefix(t1, t2, dim_idx=0)
    assert 3 == aligned_prefix(t1, t2, dim_idx=1)


def test_aligned_otherdim():
    shape_1 = (10, 11, 12)
    shape_2 = (10, 13, 12)
    assert shape_aligned_otherdim(shape_1, shape_2, 1) == True
    assert shape_aligned_otherdim(shape_1, shape_2, 2) == False

    shape_1 = (10, 11, 12)
    shape_2 = (10, 13, 13)
    assert shape_aligned_otherdim(shape_1, shape_2, 1) == False
    assert shape_aligned_otherdim(shape_1, shape_2, 2) == False

    shape_1 = (10, 11, 12)
    shape_2 = (10, 11, 12, 13)
    assert shape_aligned_otherdim(shape_1, shape_2, 2) == False
    assert shape_aligned_otherdim(shape_1, shape_2, 3) == False
