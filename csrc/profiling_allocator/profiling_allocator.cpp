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
#include <Python.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>


extern "C" {

PyObject *op_name = nullptr;
PyObject *start_recording = nullptr;
PyObject *global_dict = nullptr;
PyObject *main_module = nullptr;
PyObject *allocator_profiling_info_queue = nullptr;
PyObject *allocator_mode = nullptr;

void* reserved_space_for_memory_plan = nullptr;
uintptr_t mem_start = 0;
uintptr_t mem_end = 0;
std::unordered_set<void*> profiling_ptrs;
std::unordered_set<void*> assigned_ptrs;
bool memory_plan_initialized = false;
bool runtime_shortcut = false;
ssize_t mem_size = 0;
ssize_t temp_mem_size = 0;

class GraphMemoryPlan {
  int malloc_counter = 0;
  std::vector<void *> mem_addresses_;
  std::vector<void *> mem_ends_;        // for debug
  std::vector<std::string> op_names_;   // for debug
  std::vector<uintptr_t> mem_sizes_;   // for debug
public:
  std::pair<void*, uintptr_t> get_mem_address(int device) {
    if (malloc_counter >= mem_addresses_.size()) {
      std::cerr << "allocation number exceeds limit!" << std::endl;
      exit(-1);
    }

    //std::cout << "rank: " << device << ", malloc_counter: " << malloc_counter
    //          << ", op: " << op_names_[malloc_counter] << std::endl;
    void* ptr = mem_addresses_[malloc_counter];
    uintptr_t size = mem_sizes_[malloc_counter];
    ++malloc_counter;
    if (malloc_counter == mem_addresses_.size()) {
        malloc_counter = 0;
    }
    return std::make_pair(ptr, size);
  }

  void reset_malloc_counter() {
    malloc_counter = 0;
  }

  void set_mem_addresses(std::vector<void *>& mem_addresses) {
    mem_addresses_ = std::move(mem_addresses);
  }

  void set_mem_ends(std::vector<void *>& mem_ends) {
    mem_ends_ = std::move(mem_ends);
  }

  void set_mem_sizes(std::vector<uintptr_t>& mem_sizes) {
    mem_sizes_ = std::move(mem_sizes);
  }

  void set_op_names(std::vector<std::string>& op_names) {
    op_names_ = std::move(op_names);
  }

  std::string memory_info_str() {
    // print mem_addresses
    std::string res = "mem_addresses: [\n";
    for (int i=0; i<mem_addresses_.size(); ++i) {
       res += std::to_string(i) + "(" + op_names_[i] + "): "
              + std::to_string((uintptr_t)mem_addresses_[i]) + " ~ "
              + std::to_string((uintptr_t)mem_ends_[i]) + ", size: "
              + std::to_string(mem_sizes_[i]) + "\n";
    }
    res += "]\n";

    return res;
  }
};

GraphMemoryPlan graph_memory_plan;

void init_memory_plan(int device) {
  if (memory_plan_initialized) {
    std::cerr << "Error: Memory plan already initialized!" << std::endl;
    exit(-1);
  }

  // read memory addresses in plan and convert them to physical memory addresses
  std::vector<void *> mem_addresses;
  std::vector<void *> mem_ends;       // for debug
  std::vector<std::string> op_names;  // for debug
  std::vector<uintptr_t> mem_sizes;   // for debug
  PyObject *py_raw_mem_allocs = PyDict_GetItemString(global_dict, "raw_mem_allocs");
  mem_start = reinterpret_cast<uintptr_t>(reserved_space_for_memory_plan);
  mem_end = mem_start + mem_size + temp_mem_size;
  uintptr_t temp_mem_start = mem_start + mem_size;
  //std::cout << "rank: " << device << ", mem_start: " << mem_start << std::endl;
  //std::cout << "rank: " << device << ", temp_mem_start: " << temp_mem_start
  //          << std::endl;

  for (Py_ssize_t i = 0; i < PyList_Size(py_raw_mem_allocs); i++) {
    PyObject* current_alloc = PyList_GetItem(py_raw_mem_allocs, i);
    PyObject* py_addr = PyTuple_GetItem(current_alloc, 0);
    uintptr_t addr = PyLong_AsLong(py_addr);
    PyObject* py_is_temp_mem = PyTuple_GetItem(current_alloc, 2);
    if (PyObject_IsTrue(py_is_temp_mem)) {
      addr += temp_mem_start;
    } else {
      addr += mem_start;
    }
    mem_addresses.push_back(reinterpret_cast<void *>(addr));

    PyObject* py_size = PyTuple_GetItem(current_alloc, 1);
    uintptr_t size = PyLong_AsLong(py_size);
    mem_sizes.push_back(size);
    mem_ends.push_back(reinterpret_cast<void *>(addr+size));

    PyObject* py_op_name = PyTuple_GetItem(current_alloc, 3);
    const char* op_name = PyBytes_AsString(PyUnicode_AsUTF8String(py_op_name));
    op_names.push_back(std::string(op_name));
  }

  graph_memory_plan.set_mem_addresses(mem_addresses);
  graph_memory_plan.set_op_names(op_names);
  graph_memory_plan.set_mem_ends(mem_ends);
  graph_memory_plan.set_mem_sizes(mem_sizes);

  //std::cout << "rank: " << device << std::endl
  //          << graph_memory_plan.memory_info_str();

  // initialization finished
  memory_plan_initialized = true;
}

void init_reserved_space() {
  PyObject *py_mem_size = PyDict_GetItemString(global_dict, "mem_size");
  mem_size = PyLong_AsLong(py_mem_size);
  PyObject *py_temp_mem_size = PyDict_GetItemString(global_dict, "temp_mem_size");
  temp_mem_size = PyLong_AsLong(py_temp_mem_size);
  cudaMalloc(&reserved_space_for_memory_plan, mem_size+temp_mem_size);
}

void init_allocator(int device_count) {
   // initialize Python interpreter
   Py_Initialize();
   if (!Py_IsInitialized()) {
      PyErr_SetString(PyExc_RuntimeError, "Python initialization failed!");
      exit(-1);
   }

   // import Python __main__ module
   // main_module is a New reference.
   main_module = PyImport_ImportModule("__main__");
   if (main_module == nullptr) {
      PyErr_SetString(PyExc_ImportError, "Failed to import __main__ module!");
      exit(-1);
   }

   // get Python global variables
   // global_dict is a Borrowed reference.
   global_dict = PyModule_GetDict(main_module);
   if (global_dict == nullptr) {
      PyErr_SetString(PyExc_AttributeError, "Failed to get global dict!");
      exit(-1);
   }

   // get allocator info queue from global variables
   // allocator_profiling_info_queue is a Borrowed reference.
   allocator_profiling_info_queue = PyDict_GetItemString(global_dict, "allocator_profiling_info");
   if (allocator_profiling_info_queue == nullptr) {
      PyErr_SetString(PyExc_AttributeError, "Failed to get 'allocator_profiling_info' from __main__ module!");
      exit(-1);
   }
}

void* profiling_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMalloc(&ptr, size);

   // read start_recording flag from global variables
   // start_recording is a Borrowed reference.
   start_recording = PyDict_GetItemString(global_dict, "start_recording");
   if (start_recording == nullptr) {
      std::cerr << "start_recording is null!" << std::endl;
      exit(-1);
   }

   // if recording not started, skip profiling
   if (!PyObject_IsTrue(start_recording)) {
      return ptr;
   }

   // read op_name from global variables
   // op_name is a Borrowed reference.
   op_name = PyDict_GetItemString(global_dict, "op_name");
   if (op_name == nullptr){
      std::cerr << "op_name is null!" << std::endl;
      exit(-1);
   }

   auto ptr_int_value = reinterpret_cast<uintptr_t>(ptr);
   PyObject* py_ptr_int = Py_BuildValue("k", ptr_int_value);
   PyObject* py_size_int = Py_BuildValue("k", size);
   PyObject* profiling_info_tuple = PyTuple_Pack(3, op_name, py_ptr_int, py_size_int);
   Py_DECREF(py_ptr_int);
   Py_DECREF(py_size_int);
   if (profiling_info_tuple == nullptr) {
      std::cerr << "Runtime Error: Can't create Python tuple" << std::endl;
      exit(-1);
   }

   int succ = PyList_Append(allocator_profiling_info_queue, profiling_info_tuple);
   if (succ == -1) {
      std::cerr << "Runtime Error: Failed to append to allocator_profiling_info_queue!" << std::endl;
      exit(-1);
   }
   Py_DECREF(profiling_info_tuple);

   return ptr;
}

void profiling_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   cudaFree(ptr);
}

void* runtime_malloc(ssize_t size, int device, cudaStream_t stream) {
  PyObject *py_is_customized = PyDict_GetItemString(global_dict, "is_customized");
  if (PyObject_IsTrue(py_is_customized)) {
    if (!memory_plan_initialized) {
      std::cerr << "Error: Memory plan is not initialized!" << std::endl;
      exit(-1);
    }
    void* ptr = 0;
    if (size == 0) {
      return ptr;
    }

    std::pair<void*, uintptr_t> addr_size = graph_memory_plan.get_mem_address(device);
//#define DEBUG_MEMORY
#ifdef DEBUG_MEMORY
    if (device == 0) {
      std::string malloc_str = "(fake malloc) rank: " + std::to_string(device) + ", ptr: ";
      malloc_str += std::to_string((uintptr_t)addr_size.first) + ", alloced size: ";
      malloc_str += std::to_string(addr_size.second) + ", expected size: ";
      malloc_str += std::to_string(size) + ", stream: ";
      malloc_str += std::to_string(reinterpret_cast<uintptr_t>(stream)) + "\n";
      std::cout << malloc_str;
    }
#endif
    return addr_size.first;
  } else {
    void *ptr = nullptr;
    cudaMalloc(&ptr, size);
#ifdef DEBUG_MEMORY
    if (device == 0) {
      std::string malloc_str = "(real malloc) rank: " + std::to_string(device) + ", ptr: ";
      malloc_str += std::to_string((uintptr_t)ptr) + ", alloced size: ";
      malloc_str += std::to_string(size) + ", stream: ";
      malloc_str += std::to_string(reinterpret_cast<uintptr_t>(stream)) + "\n";
      std::cout << malloc_str;
    }
#endif
    return ptr;
  }
}

void runtime_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  if (reinterpret_cast<uintptr_t>(ptr) >= mem_end ||
      reinterpret_cast<uintptr_t>(ptr) < mem_start)
  {
    //std::cout << "(real free) rank: " << device << ", ptr: " << (uintptr_t)ptr
    //          << ", size: " << size << std::endl;
    cudaFree(ptr);
  } else {
    //std::cout << "(fake free) rank: " << device << ", ptr: " << (uintptr_t)ptr
    //          << ", size: " << size << std::endl;
  }
}

void* meta_malloc(ssize_t size, int device, cudaStream_t stream) {
   // after switching mode to runtime, skip mode routing
   if (runtime_shortcut) return runtime_malloc(size, device, stream);

   // mode routing
   allocator_mode = PyDict_GetItemString(global_dict, "allocator_mode");
   if (allocator_mode == nullptr) {
      std::cerr << "allocator mode is null!" << std::endl;
      exit(-1);
   }
   if (!PyUnicode_Check(allocator_mode)) {
      std::cerr << "Not a unicode: allocator_mode!" << std::endl;
      exit(-1);
   }
   std::string allocator_mode_string = PyBytes_AsString(PyUnicode_AsUTF8String(allocator_mode));
   if (allocator_mode_string == "profile") {
      void* ptr = profiling_malloc(size, device, stream);
      profiling_ptrs.insert(ptr);
      return ptr;
   } else if (allocator_mode_string == "runtime") {
      runtime_shortcut = true;
      init_reserved_space();
      init_memory_plan(device);
      return runtime_malloc(size, device, stream);
   } else {
      std::cerr << "allocator mode: " << allocator_mode_string << " unknown!" << std::endl;
      exit(-1);
   }
}

void meta_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   if (profiling_ptrs.find(ptr) != profiling_ptrs.end()) {
      profiling_free(ptr, size, device, stream);
      profiling_ptrs.erase(ptr);
   } else {
      runtime_free(ptr, size, device, stream);
   }
}
}
