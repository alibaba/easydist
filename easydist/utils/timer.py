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

import time

import numpy as np

from easydist.platform import get_backend
import easydist.config as mdconfig


class EDTimer:

    def __init__(self,
                 func,
                 trials=3,
                 warmup_trials=3,
                 times_per_trials=1,
                 in_ms=True,
                 device=None) -> None:
        self.func = func
        self.warmup_trials = warmup_trials
        self.trials = trials
        self.times_per_trials = times_per_trials
        self.in_ms = in_ms

        self.device = device
        if self.device == None:
            self.device = mdconfig.easydist_device

        self.backend = get_backend()

    def time(self, return_all=False):
        all_elapsed_time = None
        if self.backend == "jax":
            all_elapsed_time = self.time_jax()
        elif self.backend == "torch":
            if self.device == "cuda":
                all_elapsed_time = self.time_torch_cuda()
            elif self.device == "cpu":
                all_elapsed_time = self.time_cpu()
        if all_elapsed_time is not None:
            if return_all is True:
                return all_elapsed_time
            return np.mean(all_elapsed_time)
        return None

    def time_cpu(self):
        for _ in range(self.warmup_trials):
            self.func()

        elapsed_time = []
        for _ in range(self.trials):
            start_t = time.perf_counter()
            for _ in range(self.times_per_trials):
                self.func()
            elapsed_time.append(time.perf_counter() - start_t)

        elapsed_time = np.array(elapsed_time) / self.times_per_trials

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

            for _ in range(self.times_per_trials):
                self.func()

            if evt_idx >= 0:
                end_evt[evt_idx].record()

        torch.cuda.synchronize()
        elapsed_time = []
        for evt_idx in range(0, self.trials):
            # time elapsed in **milliseconds**
            elapsed_time.append(start_evt[evt_idx].elapsed_time(end_evt[evt_idx]))
        elapsed_time = np.array(elapsed_time) / self.times_per_trials

        if self.in_ms:
            return elapsed_time
        return elapsed_time / 1000

    def time_jax(self):
        import jax
        for _ in range(self.warmup_trials):
            self.func()
            (jax.device_put(0.) + 0).block_until_ready()

        elapsed_time = []
        for _ in range(self.trials):

            start_t = time.perf_counter()

            for _ in range(self.times_per_trials):
                self.func()

            (jax.device_put(0.) + 0).block_until_ready()

            elapsed_time.append(time.perf_counter() - start_t)
        elapsed_time = np.array(elapsed_time) / self.times_per_trials

        # time elapsed in **milliseconds**
        if self.in_ms:
            return elapsed_time * 1000
        return elapsed_time
