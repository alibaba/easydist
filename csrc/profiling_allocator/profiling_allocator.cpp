/* Copyright (c) 2023, Alibaba Group;
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
==============================================================================*/
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "cupti_callback_api.h"
#include "profiling_allocator.h"

//#define DEBUG_MEMORY

#ifdef DEBUG_MEMORY
#include <cupti.h>
#endif

void* reserved_space_for_memory_plan = nullptr;
uintptr_t mem_start = 0;
uintptr_t mem_end = 0;
std::unordered_set<void*> profiling_ptrs;
std::unordered_set<void*> assigned_ptrs;
bool memory_plan_initialized = false;
bool runtime_shortcut = false;

// Pybind var
ssize_t mem_size = 0;
ssize_t temp_mem_size = 0;
bool start_recording = false;
std::vector<std::tuple<std::string, uintptr_t, ssize_t>> allocator_profiling_info;
AllocatorMode allocator_mode = PROFILE;
std::vector<std::string> memory_plan;
bool customized_flag = false;
std::vector<std::tuple<long, long, bool, std::string>> raw_mem_allocs;

c10::cuda::CUDACachingAllocator::CUDAAllocator* back_allocator = nullptr;

class GraphMemoryPlan {
  int malloc_counter = 0;
  std::vector<void *> mem_addresses_;
  std::vector<uintptr_t> mem_sizes_;    // Lansong(TODO): may only debug?
#ifdef DEBUG_MEMORY
  std::vector<void *> mem_ends_;        // for debug
  std::vector<std::string> op_names_;   // for debug
  std::vector<bool> temp_flags_;        // for debug
#endif

public:
#ifdef DEBUG_MEMORY
  std::tuple<void*, uintptr_t, std::string, bool> get_mem_address(int device) {
    if (malloc_counter >= mem_addresses_.size()) {
      std::cerr << "allocation number exceeds limit!" << std::endl;
      exit(-1);
    }

    void* ptr = mem_addresses_[malloc_counter];
    uintptr_t size = mem_sizes_[malloc_counter];
    std::string& op_name = op_names_[malloc_counter];
    bool is_temp = temp_flags_[malloc_counter];

    ++malloc_counter;
    if (malloc_counter == mem_addresses_.size()) {
        malloc_counter = 0;
    }
    return std::make_tuple(ptr, size, op_name, is_temp);
  }
#else
  std::tuple<void*, uintptr_t> get_mem_address(int device) {
    if (malloc_counter >= mem_addresses_.size()) {
      std::cerr << "allocation number exceeds limit!" << std::endl;
      exit(-1);
    }

    void* ptr = mem_addresses_[malloc_counter];
    uintptr_t size = mem_sizes_[malloc_counter];

    ++malloc_counter;
    if (malloc_counter == mem_addresses_.size()) {
        malloc_counter = 0;
    }
    return std::make_tuple(ptr, size);
  }
#endif

  void reset_malloc_counter() {
    malloc_counter = 0;
  }

  void set_mem_addresses(std::vector<void *>& mem_addresses) {
    mem_addresses_ = std::move(mem_addresses);
  }

  void set_mem_sizes(std::vector<uintptr_t>& mem_sizes) {
    mem_sizes_ = std::move(mem_sizes);
  }

#ifdef DEBUG_MEMORY
  void set_mem_ends(std::vector<void *>& mem_ends) {
    mem_ends_ = std::move(mem_ends);
  }

  void set_op_names(std::vector<std::string>& op_names) {
    op_names_ = std::move(op_names);
  }

  void set_temp_flags(std::vector<bool>& temp_flags) {
    temp_flags_ = std::move(temp_flags);
  }

  std::string memory_info_str() {
    // print mem_addresses
    std::string res = "mem_addresses: [\n";
    for (int i=0; i<mem_addresses_.size(); ++i) {
      std::string temp_flag = temp_flags_[i] ? "True" : "False";
      res += std::to_string(i) + "(" + op_names_[i] + "): "
             + std::to_string((uintptr_t)mem_addresses_[i]) + " ~ "
             + std::to_string((uintptr_t)mem_ends_[i]) + ", size: "
             + std::to_string(mem_sizes_[i])
             + ", is temp: " + temp_flag + "\n";
    }
    res += "]\n";

    return res;
  }
#endif
};

GraphMemoryPlan graph_memory_plan;

void init_memory_plan(int device) {
  if (memory_plan_initialized) {
    std::cerr << "Error: Memory plan already initialized!" << std::endl;
    exit(-1);
  }

  mem_start = reinterpret_cast<uintptr_t>(reserved_space_for_memory_plan);
  mem_end = mem_start + mem_size + temp_mem_size;
  uintptr_t temp_mem_start = mem_start + mem_size;

  // read memory addresses in plan and convert them to physical memory addresses
  std::vector<void *> mem_addresses;
  std::vector<uintptr_t> mem_sizes;   // Lansong(TODO): may only debug?

#ifdef DEBUG_MEMORY
  std::vector<void *> mem_ends;       // for debug
  std::vector<std::string> op_names;  // for debug
  std::vector<bool> temp_flags;       // for debug
  std::cout << "rank: " << device << ", mem_start: " << mem_start << std::endl
            << std::flush;
  std::cout << "rank: " << device << ", temp_mem_start: " << temp_mem_start
            << std::endl << std::flush;
#endif

  for (auto& current_alloc: raw_mem_allocs) {
    uintptr_t addr = std::get<0>(current_alloc);
    bool is_temp_mem = std::get<2>(current_alloc);
    if (is_temp_mem) {
      addr += temp_mem_start;
    } else {
      addr += mem_start;
    }
    mem_addresses.push_back(reinterpret_cast<void *>(addr));

    uintptr_t size = std::get<1>(current_alloc);
    mem_sizes.push_back(size);
#ifdef DEBUG_MEMORY
    mem_ends.push_back(reinterpret_cast<void *>(addr+size));

    std::string cur_op_name = std::get<3>(current_alloc);
    op_names.push_back(cur_op_name);
    temp_flags.push_back(is_temp_mem);
#endif
  }

  graph_memory_plan.set_mem_addresses(mem_addresses);
  graph_memory_plan.set_mem_sizes(mem_sizes);
#ifdef DEBUG_MEMORY
  graph_memory_plan.set_op_names(op_names);
  graph_memory_plan.set_mem_ends(mem_ends);
  graph_memory_plan.set_temp_flags(temp_flags);

  if (device == 0) {
    std::cout << "rank: " << device << std::endl
              << graph_memory_plan.memory_info_str() << std::flush;
  }
#endif

  // initialization finished
  memory_plan_initialized = true;
}

void init_fn(int device_count) {
  if(back_allocator) {
    back_allocator->init(device_count);
  }
}

void init_reserved_space() {
  reserved_space_for_memory_plan = back_allocator->raw_alloc(mem_size+temp_mem_size);
}

void* profiling_malloc(ssize_t size, int device, cudaStream_t stream) {
  void *ptr = back_allocator->raw_alloc(size);
  if (!start_recording) {
    return ptr;
  }
  auto ptr_int_value = reinterpret_cast<uintptr_t>(ptr);
  std::tuple<std::string, uintptr_t, ssize_t> profiling_info_tuple = std::make_tuple(g_cur_op_name, ptr_int_value, size);
  allocator_profiling_info.push_back(profiling_info_tuple);
  return ptr;
}

void profiling_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
#ifdef DEBUG_MEMORY
  if (device == 0) {
    std::cout << "[profiler] (real free) rank: " << device << ", ptr: " << (uintptr_t)ptr
              << ", size: " << size << std::endl << std::flush;
  }
#endif
  back_allocator->raw_delete(ptr);
}

void* runtime_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef DEBUG_MEMORY
  uint32_t stream_id = 0;
  cuptiGetStreamIdEx(nullptr, stream, 1, &(stream_id));
#endif
  if (customized_flag) {
    if (!memory_plan_initialized) {
      std::cerr << "Error: Memory plan is not initialized!" << std::endl;
      exit(-1);
    }
    void* ptr = 0;
    if (size == 0) {
      return ptr;
    }

#ifdef DEBUG_MEMORY
    std::tuple<void*, uintptr_t, std::string, bool> addr_size = graph_memory_plan.get_mem_address(device);
    if (device == 0) {
      bool is_temp = std::get<3>(addr_size);
      std::string malloc_str = "[runtime] (fake malloc) rank: " + std::to_string(device);
      malloc_str += ", ptr: " + std::to_string((uintptr_t)std::get<0>(addr_size));
      malloc_str += ", alloced size: " + std::to_string(std::get<1>(addr_size));
      malloc_str += ", expected size: " + std::to_string(size) + ", stream: ";
      malloc_str += std::to_string(stream_id);
      malloc_str += ", op: " + std::get<2>(addr_size);
      if (is_temp) {
        malloc_str += ", is temp: True";
      } else {
        malloc_str += ", is temp: False";
      }
      malloc_str += "\n";
      std::cout << malloc_str << std::flush;
    }
#else
    std::tuple<void*, uintptr_t> addr_size = graph_memory_plan.get_mem_address(device);
#endif
    return std::get<0>(addr_size);
  } else {
    void *ptr = back_allocator->raw_alloc(size);
#ifdef DEBUG_MEMORY
    if (device == 0 && size > 0) {
      std::string malloc_str = "[runtime] (real malloc) rank: " + std::to_string(device);
      malloc_str += ", ptr: " + std::to_string((uintptr_t)ptr);
      malloc_str += ", alloced size: " + std::to_string(size) + ", stream: ";
      malloc_str += std::to_string(stream_id) + "\n";
      std::cout << malloc_str << std::flush;
    }
#endif
    return ptr;
  }
}

void runtime_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  if (reinterpret_cast<uintptr_t>(ptr) >= mem_end ||
      reinterpret_cast<uintptr_t>(ptr) < mem_start)
  {
#ifdef DEBUG_MEMORY
    if (device == 0) {
      std::cout << "[runtime] (real free) rank: " << device << ", ptr: " << (uintptr_t)ptr
                << ", size: " << size << std::endl << std::flush;
    }
#endif
    back_allocator->raw_delete(ptr);
  } else {
#ifdef DEBUG_MEMORY
    if (device == 0) {
      std::cout << "[runtime] (fake free) rank: " << device << ", ptr: " << (uintptr_t)ptr
                << ", size: " << size << std::endl << std::flush;
    }
#endif
  }
}

void* meta_malloc(ssize_t size, int device, cudaStream_t stream) {
  // after switching mode to runtime, skip mode routing
  if (runtime_shortcut) return runtime_malloc(size, device, stream);
  // mode routing
  if (allocator_mode==PROFILE) {
    void* ptr = profiling_malloc(size, device, stream);
#ifdef DEBUG_MEMORY
    if (device == 0 && size>0) {
      std::cout << "[profiler] malloc: " << (uintptr_t)ptr << std::endl << std::flush;
    }
#endif
    profiling_ptrs.insert(ptr);
    return ptr;
  }
  else if (allocator_mode==RUNTIME) {
    runtime_shortcut = true;
    init_reserved_space();
    init_memory_plan(device);
    return runtime_malloc(size, device, stream);
  }
  else {
    std::cerr << "allocator mode unknown!" << std::endl;
    exit(-1);
  }
}

void meta_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   if (profiling_ptrs.find(ptr) != profiling_ptrs.end()) {
#ifdef DEBUG_MEMORY
      if (device == 0) {
        std::cout << "[profiler] free: " << (uintptr_t)ptr << std::endl << std::flush;
      }
#endif
      profiling_free(ptr, size, device, stream);
      profiling_ptrs.erase(ptr);
   } else {
#ifdef DEBUG_MEMORY
      if (device == 0) {
        std::cout << "[runtime] free: " << (uintptr_t)ptr << std::endl << std::flush;
      }
#endif
      runtime_free(ptr, size, device, stream);
   }
}

//pybind link func
void set_start_recording(bool flag){
    start_recording = flag;
}

void set_allocator_mode(AllocatorMode mode){
    allocator_mode = mode;
}

void set_customized_flag(bool flag){
    customized_flag = flag;
}

void set_mem_size(long memory_size){
    mem_size = memory_size;
}

void set_temp_mem_size(long temp_memory_size){
    temp_mem_size = temp_memory_size;
}

void set_raw_mem_allocs(std::vector<std::tuple<long, long, bool, std::string>> py_mem_allocs){
    raw_mem_allocs = py_mem_allocs;
}

std::vector<std::tuple<std::string, uintptr_t, ssize_t>> get_allocator_profiling_info(){
  return allocator_profiling_info;
}

void save_back_allocator(){
  back_allocator = c10::cuda::CUDACachingAllocator::get();
}

