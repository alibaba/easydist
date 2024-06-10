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

#pragma once

#include <unordered_map>
#include <string>
#include <set>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>


namespace torch::cuda::CUDAPluggableAllocator {

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomEffectiveAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn);

struct EffectiveCUDAAllocator : public CUDAPluggableAllocator {
  EffectiveCUDAAllocator(
      std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
      std::function<void(void*, size_t, int, cudaStream_t)> free_fn);

  EffectiveCUDAAllocator(EffectiveCUDAAllocator& other);

  void* malloc(size_t size, int device, cudaStream_t stream);

#if TORCH_VERSION_MAJOR>=2 && TORCH_VERSION_MINOR>=2
  c10::DataPtr allocate(size_t size) override;
#else
  c10::DataPtr allocate(size_t size) const override;
#endif
  virtual void raw_delete(void* ptr) override;

  bool enable_runtime_trace_ = false;                   // debug only
  std::unordered_map<void*, std::string> addr_op_map_;  // debug only
};

} // namespace torch::cuda::CUDAPluggableAllocator

