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

std::unordered_set<void*> profiling_ptrs;
std::unordered_set<void*> assigned_ptrs;
bool memory_plan_initialized = false;
bool runtime_shortcut = false;

class NodeMemoryPlan {
   int malloc_counter = 0;
public:
   std::vector<int> output_indexes;
   std::vector<uintptr_t> assigned_addresses;
   int total_mallocs;

   NodeMemoryPlan() : total_mallocs(0) {}

   NodeMemoryPlan(
      std::vector<int>&& output_indexes,
      std::vector<uintptr_t>&& assigned_addresses,
      int total_mallocs) noexcept
      :  output_indexes(std::move(output_indexes)),
         assigned_addresses(std::move(assigned_addresses)),
         total_mallocs(total_mallocs) {}

   void* get_assigned_address() {
      // 1. return assigned address if current malloc is for output
      // 2. return nullptr if not for output
      void* ptr = nullptr;
      if (std::find(output_indexes.begin(), output_indexes.end(), malloc_counter) != output_indexes.end()) {
         ptr = reinterpret_cast<void*>(assigned_addresses[malloc_counter]);
      }
      malloc_counter++;
      malloc_counter %= total_mallocs;
      return ptr;
   }
};

class GraphMemoryPlan {
   std::unordered_map<std::string, NodeMemoryPlan> m_plan;
   std::vector<std::string> graph_execution_order;
   std::vector<int> global_output_indexes;
   std::vector<uintptr_t> global_assigned_addresses;
   int global_malloc_counter = 0;
   bool is_graph_execution_order_set = false;
   bool is_global_info_ready = false;
   bool is_global_output_indexes_initialized = false;
   bool is_global_assigned_addresses_initialized = false;
public:
   NodeMemoryPlan& operator[](const std::string& key) {
        return m_plan[key];
   }

   auto find(const std::string& key) {
      return m_plan.find(key);
   }

   auto end() {
      return m_plan.end();
   }

   void set_graph_execution_order(std::vector<std::string>&& new_order) {
      graph_execution_order = std::move(new_order);
      is_graph_execution_order_set = true;
   }

   void init_global_info() {
      build_global_output_indexes();
      build_global_assigned_addresses();
      is_global_info_ready = is_global_output_indexes_initialized && is_global_assigned_addresses_initialized;
   }

   void* get_global_assigned_address() {
      void* ptr = nullptr;
      if (std::find(
         global_output_indexes.begin(),
         global_output_indexes.end(),
         global_malloc_counter) != global_output_indexes.end()) {
         ptr = reinterpret_cast<void*>(global_assigned_addresses[global_malloc_counter]);
      }
      global_malloc_counter++;
      return ptr;
   }

   void print_global_info() {
      if (!is_global_info_ready){
         std::cerr << "global_info not ready yet!" << std::endl;
         return;
      }

      // print graph_execution_order
      std::cout << "graph_execution_order: [";
      for (auto i: graph_execution_order) {
         std::cout << i << ", ";
      }
      std::cout << "]" << std::endl;

      // print global_output_indexes
      std::cout << "global_output_indexes: [";
      for (auto i: global_output_indexes) {
         std::cout << i << ", ";
      }
      std::cout << "]" << std::endl;

      // print global_assigned_addresses
      std::cout << "global_assigned_addresses: [";
      for (auto i: global_assigned_addresses) {
         std::cout << i << ", ";
      }
      std::cout << "]" << std::endl;
   }

private:
   void build_global_output_indexes() {
      if (is_global_output_indexes_initialized) {
         std::cerr << "global_output_indexes already initialized!" << std::endl;
         exit(-1);
      }

      if (!is_graph_execution_order_set) {
         std::cerr << "graph_execution_order not ready!" << std::endl;
         exit(-1);
      }

      int current_index_offset = 0;
      for (const auto& node_name: graph_execution_order) {
         for (const auto& index: m_plan[node_name].output_indexes) {
            global_output_indexes.push_back(current_index_offset + index);
         }
         current_index_offset += m_plan[node_name].total_mallocs;
      }
      is_global_output_indexes_initialized = true;
   }

   void build_global_assigned_addresses() {
      if (is_global_assigned_addresses_initialized) {
         std::cerr << "global_assigned_addresses already initialized!" << std::endl;
         exit(-1);
      }

      if (!is_graph_execution_order_set) {
         std::cerr << "graph_execution_order not ready!" << std::endl;
         exit(-1);
      }

      for (const auto& node_name: graph_execution_order) {
         const auto& next_assigned_addresses = m_plan[node_name].assigned_addresses;
         global_assigned_addresses.insert(
            global_assigned_addresses.end(),
            next_assigned_addresses.begin(),
            next_assigned_addresses.end());
      }
      is_global_assigned_addresses_initialized = true;
   }
};

GraphMemoryPlan graph_memory_plan;

void init_memory_plan() {
   if (memory_plan_initialized) {
      std::cerr << "Error: Memory plan already initialized!" << std::endl;
      exit(-1);
   }
   // read memory plan starts
   PyObject *memory_plan = PyDict_GetItemString(global_dict, "memory_plan");
   PyObject *keys = PyDict_Keys(memory_plan);

   // read node memory plan from each node
   for (Py_ssize_t i = 0; i < PyList_Size(keys); i++) {
      // get dict keys
      PyObject *key = PyList_GetItem(keys, i);

      // get values according to the key
      PyObject *node_memory_plan = PyDict_GetItem(memory_plan, key);

      // read attributes from node_memory_plan
      //
      // 1. read output indexes
      PyObject *output_indexes = PyObject_GetAttrString(node_memory_plan, "output_indexes");
      std::vector<int> new_output_indexes;
      Py_ssize_t len_output_indexes = PyList_Size(output_indexes);
      for (Py_ssize_t i = 0; i < len_output_indexes; i++) {
         PyObject *index = PyList_GetItem(output_indexes, i);
         int index_value = PyLong_AsLong(index);
         new_output_indexes.push_back(index_value);
      }

      // 2. read assigned addresses
      PyObject *assigned_addresses = PyObject_GetAttrString(node_memory_plan, "assigned_addresses");
      std::vector<uintptr_t> new_assigned_addresses;
      Py_ssize_t len_assigned_addresses = PyList_Size(assigned_addresses);
      for (Py_ssize_t i = 0; i < len_assigned_addresses; i++) {
         PyObject *address = PyList_GetItem(assigned_addresses, i);
         uintptr_t address_value = PyLong_AsUnsignedLong(address);
         new_assigned_addresses.push_back(address_value);
      }

      // 3. read total mallocs
      PyObject *total_mallocs = PyObject_GetAttrString(node_memory_plan, "total_mallocs");
      int new_total_mallocs = PyLong_AsLong(total_mallocs);

      // insert node memory plan into graph memory plan
      graph_memory_plan[PyUnicode_AsUTF8(key)] = NodeMemoryPlan(
         std::move(new_output_indexes),
         std::move(new_assigned_addresses),
         new_total_mallocs);
   }

   // read graph_execution_order
   std::vector<std::string> new_graph_execution_order;
   PyObject *py_graph_execution_order = PyDict_GetItemString(global_dict, "graph_execution_order");
   for (Py_ssize_t i = 0; i < PyList_Size(py_graph_execution_order); i++) {
      PyObject *current_node_name = PyList_GetItem(py_graph_execution_order, i);
      new_graph_execution_order.push_back(PyUnicode_AsUTF8(current_node_name));
   }
   graph_memory_plan.set_graph_execution_order(std::move(new_graph_execution_order));

   graph_memory_plan.init_global_info();

   graph_memory_plan.print_global_info();

   // initialization finished
   memory_plan_initialized = true;
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
   if (!memory_plan_initialized) {
      std::cerr << "Error: Memory plan is not initialized!" << std::endl;
      exit(-1);
   }
   void *ptr = nullptr;
   op_name = PyDict_GetItemString(global_dict, "op_name");
   std::string op_name_key = std::string(PyBytes_AsString(PyUnicode_AsUTF8String(op_name)));

   // find if op_name is in memory plan
   if (graph_memory_plan.find(op_name_key) != graph_memory_plan.end()) {
      // if op_name in memory plan, set ptr to assigned_address
      ptr = graph_memory_plan[op_name_key].get_assigned_address();
   }

   // allocate memory if is not an assigned address
   if (ptr == nullptr) {
      cudaMalloc(&ptr, size);
   }
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
      init_memory_plan();
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