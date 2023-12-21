#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <Python.h>
#include <unordered_set>


extern "C" {

PyObject *op_name = nullptr;
PyObject *start_recording = nullptr;
PyObject *global_dict = nullptr;
PyObject *main_module = nullptr;
PyObject *allocator_profiling_info_queue = nullptr;
PyObject *allocator_mode = nullptr;

std::unordered_set<void*> profiling_ptrs;
bool runtime_shortcut = false;

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

   // only profile on device 0
   if (device != 0) return ptr;

   // read start_recording flag from global variables
   // start_recording is a Borrowed reference.
   start_recording = PyDict_GetItemString(global_dict, "start_recording");
   if (start_recording == nullptr) {
      std::cerr << "start_recording is null!" << std::endl;
      exit(-1);
   }

   // if recording not started, skip profiling
   if (PyObject_IsFalse(start_recording)) {
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

   auto op_name_string = std::string(PyBytes_AsString(PyUnicode_AsUTF8String(op_name)));
   if (op_name_string == "native_layer_norm") {
      std::cout << "op_name: " << op_name_string << std::endl;
      std::cout << "int of ptr: " << ptr_int_value << std::endl;
      std::cout << "size: " << size << std::endl;
   }

   return ptr;
}

void profiling_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   cudaFree(ptr);
}

void* runtime_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMalloc(&ptr, size);
   return ptr;
}

void runtime_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   cudaFree(ptr);
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