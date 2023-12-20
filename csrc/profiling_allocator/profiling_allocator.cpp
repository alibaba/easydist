#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <Python.h>
#include <unordered_set>


extern "C" {

PyObject *op_name = nullptr;
PyObject *ignore = nullptr;
PyObject *global_dict = nullptr;
PyObject *main_module = nullptr;
PyObject *allocator_profiling_info_queue = nullptr;
PyObject *allocator_mode = nullptr;
bool verbose = false;
std::unordered_set<void*> profiling_ptrs;

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

   // read ignore flag from global variables
   // ignore is a Borrowed reference.
   ignore = PyDict_GetItemString(global_dict, "ignore");
   if (ignore == nullptr) {
      std::cerr << "ignore is null!" << std::endl;
      exit(-1);
   }

   // ignore: skip profiling
   if (PyObject_IsTrue(ignore)) {
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
   PyObject* py_int = Py_BuildValue("k", ptr_int_value);
   PyObject* profiling_info_tuple = PyTuple_Pack(2, op_name, py_int);
   Py_DECREF(py_int);
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

   if (verbose) {
      auto op_name_string = PyBytes_AsString(PyUnicode_AsUTF8String(op_name));
      std::cout << "alloc "<< ptr << " " << "size: " << size << std::endl;
      std::cout << "op_name: " << op_name_string << std::endl;
      std::cout << "int of ptr: " << ptr_int_value << std::endl;
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