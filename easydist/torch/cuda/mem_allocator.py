# Copyright (c) 2024, Alibaba Group;
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

import ctypes

from easydist.torch._C.profiling_allocator import _cuda_changeCurrentAllocator, \
                                                  _cuda_customEffectiveAllocator, \
                                                  _cuda_CUDAAllocator


class _CUDAAllocator:
    def __init__(self, allocator: _cuda_CUDAAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class EffectiveCUDAAllocator(_CUDAAllocator):
    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = _cuda_customEffectiveAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _CUDAAllocator) -> None:
    _cuda_changeCurrentAllocator(allocator.allocator())


