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

import os
import time

from metadist.platform import get_backend
import metadist.config as mdconfig


class MDTimer:

    def __init__(self, func, trials=3, warmup_trials=3, in_ms=True, device=None) -> None:
        self.func = func
        self.warmup_trials = warmup_trials
        self.trials = trials
        self.in_ms = in_ms

        self.device = device
        if self.device == None:
            self.device = mdconfig.metadist_device

        self.backend = get_backend()

    def time(self):
        if self.backend == "jax":
            return self.time_jax()
        elif self.backend == "torch":
            if self.device == "cuda":
                return self.time_torch_cuda()
            elif self.device == "cpu":
                return self.time_cpu()
        return None

    def time_cpu(self):
        for _ in range(self.warmup_trials):
            self.func()

        start_t = time.perf_counter()
        for _ in range(self.trials):
            self.func()

        elapsed_time = time.perf_counter() - start_t
        elapsed_time = elapsed_time / self.trials

        # time elapsed in **milliseconds**
        if self.in_ms:
            return elapsed_time * 1000
        return elapsed_time

    def time_torch_cuda(self):
        import torch

        start_evt = []
        end_evt = []
        for _ in range(0, self.trials):
            start_evt.append(torch.cuda.Event(enable_timing=True))
            end_evt.append(torch.cuda.Event(enable_timing=True))

        for trial_idx in range(0, self.trials + self.warmup_trials):
            evt_idx = trial_idx - self.warmup_trials

            if evt_idx >= 0:
                start_evt[evt_idx].record()

            self.func()

            if evt_idx >= 0:
                end_evt[evt_idx].record()

        torch.cuda.synchronize()
        ops_elapsed_time = 0
        for evt_idx in range(0, self.trials):
            # time elapsed in **milliseconds**
            ops_elapsed_time += start_evt[evt_idx].elapsed_time(end_evt[evt_idx])
        ops_elapsed_time = ops_elapsed_time / self.trials

        if self.in_ms:
            return ops_elapsed_time
        return ops_elapsed_time / 1000

    def time_jax(self):
        import jax
        for _ in range(self.warmup_trials):
            self.func()
            (jax.device_put(0.) + 0).block_until_ready()

        start_t = time.perf_counter()
        for _ in range(self.trials):
            self.func()
            (jax.device_put(0.) + 0).block_until_ready()

        elapsed_time = time.perf_counter() - start_t
        elapsed_time = elapsed_time / self.trials

        # time elapsed in **milliseconds**
        if self.in_ms:
            return elapsed_time * 1000
        return elapsed_time
