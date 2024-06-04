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

# torchrun --nproc_per_node 4 tests/test_torch/test_pp/test_reslink.py
import argparse
from contextlib import nullcontext
import os
import random
import time

import numpy as np

import torch
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.distributed._tensor import DeviceMesh
from tqdm import tqdm

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points,
                                                             split_into_equal_size)
from torch.profiler import profile, ProfilerActivity


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(1024, 1024)
        self.layer1 = torch.nn.Linear(1024, 1024)
        self.layer2 = torch.nn.Linear(1024, 1024)
        self.layer3 = torch.nn.Linear(1024, 1)

    def forward(self, x):
        res = self.layer0(x)
        x = self.layer1(res)
        x = self.layer2(x)
        x = self.layer3(x + res)
        return x


def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for built-in Python
    random.seed(seed)
    # Set(seed) for each of the random number generators in python:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_main(args):
    per_chunk_sz = args.micro_batch_size
    num_chunks = args.num_chunks
    batch_size = per_chunk_sz * num_chunks
    schedule_cls = ScheduleGPipe if args.schedule == 'gpipe' else ScheduleDAPPLE
    do_profile = args.do_profile
    seed(42)
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    pp_size = int(os.environ["WORLD_SIZE"])
    device = torch.device('cuda')
    torch.cuda.set_device(rank)

    set_device_mesh(DeviceMesh("cuda", torch.arange(pp_size), mesh_dim_names=['pp']))

    module = Foo().train().to(device)
    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)

    annotate_split_points(module, {'layer0', 'layer1', 'layer2'})

    @easydist_compile(parallel_mode="pp",
                      tracing_mode="fake",
                      cuda_graph=False,
                      schedule_cls=schedule_cls,
                      num_chunks=num_chunks,
                      return_to_all_stages=False)
    def train_step(input, label, model, opt):
        out = model(input)
        loss = out.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        return out, loss

    dataset_size = 10000
    train_dataloader = [(torch.randn(
        batch_size, 1024, device=device), torch.randint(0, 10, (batch_size, ), device=device))
                        ] * (dataset_size // batch_size)

    x_batch, y_batch = next(iter(train_dataloader))
    train_step(x_batch, y_batch, module, opt)  # compile
    epochs = 1
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            #  experimental_config=torch._C._profiler._ExperimentalConfig(
            #      verbose=True),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f'./log/res-{schedule_cls.__name__}-{rank}')) if do_profile else nullcontext(
                ) as prof:
        time_start = time.time()
        torch.cuda.synchronize()
        for _ in range(epochs):
            for x_batch, y_batch in tqdm(train_dataloader,
                                         dynamic_ncols=True) if rank == 0 else train_dataloader:
                if x_batch.size(0) != batch_size:  # TODO need to solve this
                    continue
                _ = train_step(x_batch, y_batch, module, opt)
        torch.cuda.synchronize()
        time_sum = time.time() - time_start
        if rank == 0:
            print(f"Finish in {time_sum:.3f} seconds")
    with open(f'{schedule_cls.__name__}-{rank}-test.txt', 'a') as f:
        f.write(
            f'num_chunks: {num_chunks}, chunk_sz {per_chunk_sz}, time: {time_sum / epochs:2.1f}, memory: {torch.cuda.max_memory_allocated(rank) / 1024 / 1024:5.0f}MB\n'
        )
    if do_profile:
        config = {
            'row_limit': 100,
            'max_src_column_width': 75,
            'max_name_column_width': 55,
            'max_shapes_column_width': 80,
        }
        with open(f'./log/res-{schedule_cls.__name__}-profile-{rank}.txt', 'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total",
                                              top_level_events_only=True,
                                              header='sort by cuda_time_total',
                                              **config))
            f.write('\n\n\n')
            f.write(prof.key_averages().table(sort_by="cpu_time_total",
                                              top_level_events_only=True,
                                              header='sort by cpu_time_total',
                                              **config))
        # prof.export_stacks(f"{schedule_cls.__name__}-profile-fg.txt",
        #                    "self_cuda_time_total")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-batch-size', type=int, default=128)
    parser.add_argument('--num-chunks', type=int, default=4)
    parser.add_argument('--schedule', type=str, default='gpipe', choices=['gpipe', 'dapple'])
    parser.add_argument('--do-profile', action='store_true', default=False)
    args = parser.parse_args()
    test_main(args)
    print("no deadlock")
