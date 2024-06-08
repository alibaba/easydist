#pragma once
#include <vector>
#include <string>
#include <tuple>
#include "c10/cuda/CUDACachingAllocator.h"

enum AllocatorMode {
  PROFILE,
  RUNTIME
};
// CType Func
extern "C"{
void init_fn(int device_count);

void* meta_malloc(ssize_t size, int device, cudaStream_t stream);

void meta_free(void* ptr, ssize_t size, int device, cudaStream_t stream);
}
// Pybind Func

void set_start_recording(bool flag);

void set_allocator_mode(AllocatorMode mode);

void set_customized_flag(bool flag);

void set_mem_size(long memory_size);

void set_temp_mem_size(long temp_memory_size);

void set_raw_mem_allocs(std::vector<std::tuple<long, long, bool, std::string>> py_mem_allocs);

std::vector<std::tuple<std::string, uintptr_t, ssize_t>> get_allocator_profiling_info();

void save_back_allocator();