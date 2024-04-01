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

#include <cassert>

#include "stream_tracer.h"
#include "cupti_callback_api.h"

std::string g_cur_op_name("N/A");
bool g_stream_tracing_active = false;
bool g_in_op_core_context = false;

void prepareStreamTracer() {
  auto cbapi = StreamTracerCallbackApi::singleton();
  cbapi->initCallbackApi();
}

void enableStreamTracer() {
  auto cbapi = StreamTracerCallbackApi::singleton();
  cbapi->start();
}

void disableStreamTracer() {
  auto cbapi = StreamTracerCallbackApi::singleton();
  cbapi->stop();
}

void activateStreamTracer() {
  g_stream_tracing_active = true;
}

void inactivateStreamTracer() {
  g_stream_tracing_active = false;
}

void enterOpCore() {
  g_in_op_core_context = true;
}

void leaveOpCore() {
  g_in_op_core_context = false;
}

void setCurOpName(const char* cur_op_name) {
  assert(cur_op_name);
  g_cur_op_name = std::string(cur_op_name);
}

StreamTraceData
getStreamTraceData() {
  auto cbapi = StreamTracerCallbackApi::singleton();
  return cbapi->getTraceData();
}

