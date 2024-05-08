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

#include <map>
#include <memory>
#include <vector>
#include <cassert>
#include <cupti.h>
#include <string>
#pragma once

//#define TRACER_VERBOSE

extern std::string g_cur_op_name;
extern bool g_stream_tracing_active;
extern bool g_in_op_core_context;

typedef struct StreamTraceData_st {
  void addOpStream(const std::string& op_name, uint32_t stream_id,
                   bool is_core) {
    if (is_core) {
      op_streams_[op_name].emplace_back(stream_id);
    } else {
      op_extra_streams_[op_name].emplace_back(stream_id);
    }
  }

#ifdef TRACER_VERBOSE
  void addOpKernel(const std::string& op_name, const std::string& kernel_name,
                   bool is_core) {
    if (is_core) {
      op_kernels_[op_name].emplace_back(kernel_name);
    } else {
      op_extra_kernels_[op_name].emplace_back(kernel_name);
    }
  }

  std::string toString() const {
    std::string ret;
    for (auto& item : op_streams_) {
      ret += "op: " + item.first + ", kernel num: " + \
              std::to_string(item.second.size()) + "\n";
      auto& kernels = op_kernels_.at(item.first);
      assert(kernels.size() == item.second.size());
      for (int i=0; i<kernels.size(); ++i) {
        ret += "  streamid: " + std::to_string(item.second[i]) + "\n";
        ret += "  kernel:   " + kernels[i] + "\n";
      }
    }

    for (auto& item : op_extra_streams_) {
      ret += "op: " + item.first + ", extra kernel num: " + \
              std::to_string(item.second.size()) + "\n";
      auto& kernels = op_extra_kernels_.at(item.first);
      assert(kernels.size() == item.second.size());
      for (int i=0; i<kernels.size(); ++i) {
        ret += "  extra streamid: " + std::to_string(item.second[i]) + "\n";
        ret += "  extra kernel:   " + kernels[i] + "\n";
      }
    }

    return ret;
  }
#endif

  void clear() {
    op_streams_.clear();
    op_extra_streams_.clear();
#ifdef TRACER_VERBOSE
    op_kernels_.clear();
    op_extra_kernels_.clear();
#endif
  }

  std::map<std::string, std::vector<uint32_t>> op_streams_;
  std::map<std::string, std::vector<uint32_t>> op_extra_streams_;

#ifdef TRACER_VERBOSE
  std::map<std::string, std::vector<std::string>> op_kernels_;
  std::map<std::string, std::vector<std::string>> op_extra_kernels_;
#endif
} StreamTraceData;

class StreamTracerCallbackApi {
 public:
  StreamTracerCallbackApi() = default;
  StreamTracerCallbackApi(const StreamTracerCallbackApi&) = delete;
  StreamTracerCallbackApi& operator=(const StreamTracerCallbackApi&) = delete;
  ~StreamTracerCallbackApi();

  static std::shared_ptr<StreamTracerCallbackApi> singleton();

  void initCallbackApi();
  void start();
  void stop();
  StreamTraceData getTraceData() {
    return trace_data_;
  }
 private:
  CUpti_SubscriberHandle subscriber_ {0};
  StreamTraceData trace_data_;
};

