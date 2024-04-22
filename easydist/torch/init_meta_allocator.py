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

import torch
import ctypes
import os
import __main__
import easydist
from profiling_allocator import _save_back_allocator

def init_meta_allocator():
    if not easydist.config.enable_memory_opt:
        return
    swap_to_profiling_allocator()

def swap_to_profiling_allocator():
    # swap from caching allocator to profiling allocator
    path_to_profiling_allocator = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                        "profiling_allocator.cpython-38-x86_64-linux-gnu.so")
    raw_allocator = ctypes.CDLL(path_to_profiling_allocator)
    init_fn = ctypes.cast(getattr(raw_allocator, 'init_fn'), ctypes.c_void_p).value
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        path_to_profiling_allocator, 'meta_malloc', 'meta_free')
    _save_back_allocator()
    torch.cuda.memory.change_current_allocator(new_alloc)
    new_alloc.allocator().set_init_fn(init_fn)