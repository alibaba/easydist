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


def init_meta_allocator():
    if not easydist.config.enable_memory_opt:
        return
    register_global_variables()
    swap_to_profiling_allocator()

def register_global_variables():
    # setting global variables
    __main__.start_recording = False
    __main__.op_name = 'N/A'
    __main__.allocator_profiling_info = []
    __main__.allocator_mode = 'profile'
    __main__.memory_plan = []

def swap_to_profiling_allocator():
    # swap from caching allocator to profiling allocator
    path_to_profiling_allocator = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                        "build",
                        "lib.linux-x86_64-cpython-38",
                        "profiling_allocator.cpython-38-x86_64-linux-gnu.so")
    raw_allocator = ctypes.CDLL(path_to_profiling_allocator)
    init_fn = ctypes.cast(getattr(raw_allocator, 'init_allocator'), ctypes.c_void_p).value
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        path_to_profiling_allocator, 'meta_malloc', 'meta_free')
    torch.cuda.memory.change_current_allocator(new_alloc)
    new_alloc.allocator().set_init_fn(init_fn)
