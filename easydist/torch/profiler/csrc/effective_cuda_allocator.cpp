/* Copyright (c) 2024, Alibaba Group;
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

#include <mutex>
#include <iostream>

#include "effective_cuda_allocator.h"

extern std::string g_cur_op_name;
extern uintptr_t mem_start;
extern uintptr_t mem_end;

namespace torch::cuda::CUDAPluggableAllocator {
extern int device_count;

EffectiveCUDAAllocator::EffectiveCUDAAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn)
    : CUDAPluggableAllocator(alloc_fn, free_fn)
    , addr_op_map_() {
  enable_runtime_trace_ = false;
  const char* enable_trace_str = std::getenv("ENABLE_RUNTIME_TRACE");
  if (enable_trace_str) {
    std::string enable(enable_trace_str);
    if (enable == "True" || enable == "TRUE" || enable == "true") {
      enable_runtime_trace_ = true;
    }
  }
}

EffectiveCUDAAllocator::EffectiveCUDAAllocator(EffectiveCUDAAllocator& other)
    : CUDAPluggableAllocator(other)
    , enable_runtime_trace_(other.enable_runtime_trace_)
    , addr_op_map_(other.addr_op_map_) {
}

void* EffectiveCUDAAllocator::malloc(
    size_t size,
    int device,
    cudaStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  if (reinterpret_cast<uintptr_t>(r) >= mem_end ||
      reinterpret_cast<uintptr_t>(r) < mem_start) {
    {
      const std::lock_guard<std::mutex> lock(allocator_mutex_);
      allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
    }
  }

  if (enable_runtime_trace_ && device == 0) {
    std::string mem_type;
    if (reinterpret_cast<uintptr_t>(r) >= mem_end ||
        reinterpret_cast<uintptr_t>(r) < mem_start) {
      mem_type = "dynamic";
    } else {
      mem_type = "static";
    }

    if (addr_op_map_.find(r) != addr_op_map_.end()) {
      auto& op_name = addr_op_map_[r];
      std::cout << mem_type << " address " << r << " is reassigned from "
                << op_name << " to " << g_cur_op_name << std::endl << std::flush;
      op_name = g_cur_op_name;
    } else {
      std::cout << mem_type << " address " << r << " is allocated for "
                << g_cur_op_name << std::endl << std::flush;
      addr_op_map_.emplace(r, g_cur_op_name);
    }
  }

  return r;
}

#if TORCH_VERSION_MAJOR>=2 && TORCH_VERSION_MINOR>=2
c10::DataPtr EffectiveCUDAAllocator::allocate(size_t size) {
  c10::DeviceIndex device;
#else
c10::DataPtr EffectiveCUDAAllocator::allocate(size_t size) const {
  int device;
#endif
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  void* r =
      const_cast<EffectiveCUDAAllocator*>(this)->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::CUDA, device)};
  return data_ptr;
}

void EffectiveCUDAAllocator::raw_delete(void* ptr) {
  cudaStream_t stream;
  int device_idx;
  size_t size;
  if (reinterpret_cast<uintptr_t>(ptr) >= mem_end ||
      reinterpret_cast<uintptr_t>(ptr) < mem_start) {
    // back allocator handle
    {
      const std::lock_guard<std::mutex> lock(allocator_mutex_);
      TORCH_CHECK(
          allocation_metadata_.count(ptr),
          "Trying to free a pointer not allocated here");
      _AllocationMetadata& metadata = allocation_metadata_[ptr];
      size = metadata.size;
      device_idx = metadata.device_idx;
      stream = metadata.stream;
      allocation_metadata_.erase(ptr);
    }
  }

  if (enable_runtime_trace_ && device_idx == 0) {
    std::string mem_type;
    if (reinterpret_cast<uintptr_t>(ptr) >= mem_end ||
        reinterpret_cast<uintptr_t>(ptr) < mem_start) {
      mem_type = "dynamic";
    } else {
      mem_type = "static";
    }

    if (addr_op_map_.find(ptr) == addr_op_map_.end()) {
      std::cout << mem_type << " address " << ptr << " doesn't exist."
                << std::endl << std::flush;
    } else {
      auto& op_name = addr_op_map_[ptr];
      std::cout << mem_type << " address " << ptr << " is delete for "
                << op_name << std::endl << std::flush;
      addr_op_map_.erase(ptr);
    }
  }

  free_fn_(ptr, size, device_idx, stream);
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomEffectiveAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn) {
  std::shared_ptr<EffectiveCUDAAllocator> allocator(
      new EffectiveCUDAAllocator(alloc_fn, free_fn));
  allocator->init(device_count);
  return allocator;
}

} // namespace torch::cuda::CUDAPluggableAllocator

