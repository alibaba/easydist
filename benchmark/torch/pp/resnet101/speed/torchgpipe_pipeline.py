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

# python benchmark/torch/pp/resnet101/speed/torchgpipe_pipeline.py
import argparse
from contextlib import nullcontext
import random
from re import M
import time
from typing import cast

import numpy as np

import torch

from torchgpipe import GPipe
from resnet import resnet101
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity


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


criterion = torch.nn.CrossEntropyLoss()


def test_main(args):
    per_chunk_sz = args.micro_batch_size
    num_chunks = args.num_chunks
    do_profile = args.do_profile
    batch_size = per_chunk_sz * num_chunks
    devices = [0, 1, 2, 3]
    in_device = devices[0]
    out_device = devices[-1]
    torch.cuda.set_device(in_device)

    seed(42)

    module = resnet101().train()
    module.fc = torch.nn.Linear(module.fc.in_features, 1000)

    module = cast(torch.nn.Sequential, module)
    module = GPipe(module, [25, 45, 135, 165],
                   devices=devices,
                   chunks=num_chunks,
                   checkpoint='never')
    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)

    def train_step(input, label, model, opt):
        out = model(input)
        loss = criterion(out, label)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return out, loss

    dataset_size = 10000
    train_dataloader = [
        (torch.randn(batch_size, 3, 224, 224, device=in_device),
         torch.randint(0, 10, (batch_size, ), device=out_device))
    ] * (dataset_size // batch_size)

    x_batch, y_batch = next(iter(train_dataloader))
    train_step(x_batch, y_batch, module, opt)  # compile
    epochs = 1
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=True,
                 experimental_config=torch._C._profiler._ExperimentalConfig(
                     verbose=True),
                     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/torchgpipe')
                     ) if do_profile else nullcontext() as prof:
        time_start = time.time()
        torch.cuda.synchronize(in_device)
        for epoch in range(epochs):
            all_cnt, correct_cnt, loss_sum = 0, 0, 0
            for x_batch, y_batch in tqdm(train_dataloader, dynamic_ncols=True):
                if x_batch.size(0) != batch_size:  # TODO need to solve this
                    continue
                _ = train_step(x_batch, y_batch, module, opt)
        torch.cuda.synchronize(in_device)
        print(f"Finish in {time.time() - time_start:.3f} seconds")
    if do_profile:
        config = {
            'row_limit': 100,
            'max_src_column_width': 75,
            'max_name_column_width': 55,
            'max_shapes_column_width': 80,
        }
        with open(f'./log/torchgpipe-profile.txt', 'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total",
                                              top_level_events_only=True,
                                              header='sort by cuda_time_total',
                                              **config))
            f.write('\n\n\n')
            f.write(prof.key_averages().table(sort_by="cpu_time_total",
                                              top_level_events_only=True,
                                              header='sort by cpu_time_total',
                                              **config))
        prof.export_stacks(f"torchgpipe-profile-fg.txt",
                           "self_cuda_time_total")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-batch-size', type=int, default=128)
    parser.add_argument('--num-chunks', type=int, default=4)
    parser.add_argument('--do-profile', action='store_true', default=False)
    args = parser.parse_args()
    test_main(args)
