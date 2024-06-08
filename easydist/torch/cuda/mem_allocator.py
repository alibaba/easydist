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

import easydist
from easydist.torch.meta_allocator import profiling_allocator


class _CUDAAllocator:
    def __init__(self, allocator):
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
        self._allocator = profiling_allocator._cuda_customEffectiveAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _CUDAAllocator) -> None:
    profiling_allocator._cuda_changeCurrentAllocator(allocator.allocator())

def init_meta_allocator():
    if not easydist.config.enable_memory_opt:
        return
    swap_to_profiling_allocator()

def swap_to_profiling_allocator():
    # swap from caching allocator to profiling allocator

    profiling_allocator._compile_if_needed()

    path_to_profiling_allocator = profiling_allocator.module.__file__
    raw_allocator = ctypes.CDLL(path_to_profiling_allocator)
    init_fn = ctypes.cast(getattr(raw_allocator, 'init_fn'), ctypes.c_void_p).value
    new_alloc = EffectiveCUDAAllocator(
        path_to_profiling_allocator, 'meta_malloc', 'meta_free')
    profiling_allocator._save_back_allocator()
    change_current_allocator(new_alloc)
    new_alloc.allocator().set_init_fn(init_fn)

