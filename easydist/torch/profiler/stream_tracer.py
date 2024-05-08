# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from profiling_allocator import _prepare_stream_tracer, _enable_stream_tracer, \
                                _disable_stream_tracer, _get_stream_trace_data

__all__ = [
    "StreamTracer",
]


class StreamTracer():
    def __init__(
        self,
        enabled=True,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return

        self.entered = False

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Stream tracer's context manager is not reentrant")
        self.prepare_trace()
        self.start_trace()
        return self

    def prepare_trace(self):
        self.entered = True
        _prepare_stream_tracer()

    def start_trace(self):
        self.entered = True
        _enable_stream_tracer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        _disable_stream_tracer()
        return False

    def get_stream_trace_data(self):
        return _get_stream_trace_data()


