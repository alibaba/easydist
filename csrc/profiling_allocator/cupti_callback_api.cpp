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

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <Python.h>
#include <cassert>

//#define DEMANGLE_FUNC_NAME

#ifdef DEMANGLE_FUNC_NAME
#include <cxxabi.h>
#endif

#include "cupti_callback_api.h"

#define CUPTI_CALL(call)                                                        \
do {                                                                            \
    CUptiResult _status = call;                                                 \
    if (_status != CUPTI_SUCCESS) {                                             \
      const char* errstr;                                                       \
      cuptiGetResultString(_status, &errstr);                                   \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",      \
              __FILE__, __LINE__, #call, errstr);                               \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
} while (0)


void CUPTIAPI
getStreamIdCallback(void *userdata, CUpti_CallbackDomain domain,
                    CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  if (!g_stream_tracing_active) {
    return;
  }
  //std::cout << "getStreamIdCallback with cbid: " << cbid << std::endl;
  StreamTraceData* traceData = (StreamTraceData*)userdata;

  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
      cbInfo->callbackSite == CUPTI_API_ENTER) {
    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
      fprintf(stderr, "%s:%d: error: function getStreamIdCallback failed with v3020 device.\n",
              __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
      cudaLaunchKernel_v7000_params *funcParams =
                    (cudaLaunchKernel_v7000_params *)(cbInfo->functionParams);
      cudaStream_t stream = funcParams->stream;
      uint32_t stream_id = 0;
      cuptiGetStreamIdEx(cbInfo->context, stream, 1, &(stream_id));

      traceData->addOpStream(g_cur_op_name, stream_id, g_in_op_core_context);
#ifdef TRACER_VERBOSE
      assert(cbInfo->symbolName);

#ifdef DEMANGLE_FUNC_NAME
      char* sym_name = abi::__cxa_demangle(cbInfo->symbolName,
                                           nullptr,
                                           nullptr,
                                           nullptr);
      std::string func_name(sym_name);
      free(sym_name);
#else
      std::string func_name(cbInfo->symbolName);
#endif
      std::string dump_str = "launch kernel: " + func_name + \
                             ", stream id: " + std::to_string(stream_id);
      if (stream == nullptr) {
        dump_str += ", null stream";
      }
      std::cout << dump_str << std::endl;

      traceData->addOpKernel(g_cur_op_name, func_name, g_in_op_core_context);
#endif
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
      cudaMemcpy_v3020_params * memcpy_params =
                        (cudaMemcpy_v3020_params *)(cbInfo->functionParams);
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060) {
      cudaLaunchKernelExC_v11060_params *funcParams =
                    (cudaLaunchKernelExC_v11060_params *)(cbInfo->functionParams);
      const cudaLaunchConfig_t *config = funcParams->config;
      cudaStream_t stream = config->stream;
      uint32_t stream_id = 0;
      cuptiGetStreamIdEx(cbInfo->context, stream, 1, &(stream_id));

      traceData->addOpStream(g_cur_op_name, stream_id, g_in_op_core_context);
#ifdef TRACER_VERBOSE
      assert(cbInfo->symbolName);
#ifdef DEMANGLE_FUNC_NAME
      char* sym_name = abi::__cxa_demangle(cbInfo->symbolName,
                                           nullptr,
                                           nullptr,
                                           nullptr);
      std::string func_name(sym_name);
      free(sym_name);
#else
      std::string func_name(cbInfo->symbolName);
#endif
      std::string dump_str = "launch kernel exc: " + func_name + \
                             ", stream id: " + std::to_string(stream_id);
      if (stream == nullptr) {
        dump_str += ", null stream";
      }
      std::cout << dump_str << std::endl;

      traceData->addOpKernel(g_cur_op_name, func_name, g_in_op_core_context);
#endif
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060) {
    }
  }
}

StreamTracerCallbackApi::~StreamTracerCallbackApi() {
}

std::shared_ptr<StreamTracerCallbackApi> StreamTracerCallbackApi::singleton() {
	static const std::shared_ptr<StreamTracerCallbackApi>
		instance = [] {
			std::shared_ptr<StreamTracerCallbackApi> inst =
				std::shared_ptr<StreamTracerCallbackApi>(new StreamTracerCallbackApi());
			return inst;
	}();
  return instance;
}

void StreamTracerCallbackApi::initCallbackApi() {
}

void StreamTracerCallbackApi::start() {
  CUPTI_CALL(cuptiSubscribe(&subscriber_,
                            (CUpti_CallbackFunc)getStreamIdCallback,
                            &trace_data_));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API));
  /*
  CUPTI_CALL(cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
  CUPTI_CALL(cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
  */
  trace_data_.clear();
}

void StreamTracerCallbackApi::stop() {
  CUPTI_CALL(cuptiEnableDomain(0, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API));
  /*
  CUPTI_CALL(cuptiEnableCallback(0, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
  CUPTI_CALL(cuptiEnableCallback(0, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
  */
  CUPTI_CALL(cuptiUnsubscribe(subscriber_));

#ifdef TRACER_VERBOSE
  std::cout << "trace data:" << std::endl << trace_data_.toString() << std::endl;
#endif
}



