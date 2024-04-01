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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stream_tracer.h"

namespace py = pybind11;

PYBIND11_MODULE(profiling_allocator, m) {
    m.doc() = "memory tracer";
    m.def("_enable_stream_tracer", enableStreamTracer);
    m.def("_disable_stream_tracer", disableStreamTracer);
    m.def("_prepare_stream_tracer", prepareStreamTracer);
    m.def("_set_cur_op_name", setCurOpName);
    m.def("_activate_stream_tracer", activateStreamTracer);
    m.def("_inactivate_stream_tracer", inactivateStreamTracer);
    m.def("_enter_op_core", enterOpCore);
    m.def("_leave_op_core", leaveOpCore);
    m.def("_get_stream_trace_data", getStreamTraceData);

    py::class_<StreamTraceData>(m, "StreamTraceData")
        .def(py::init<>())
#ifdef TRACER_VERBOSE
        .def_readwrite("op_kernels", &StreamTraceData::op_kernels_)
        .def_readwrite("op_extra_kernels", &StreamTraceData::op_extra_kernels_)
#endif
        .def_readwrite("op_streams", &StreamTraceData::op_streams_)
        .def_readwrite("op_extra_streams", &StreamTraceData::op_extra_streams_);
}


