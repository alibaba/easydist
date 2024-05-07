
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

# torchrun --nproc_per_node 4 benchmark/torch/pp/resnet101/speed/pippy_pipeline.py
import argparse
import os
import random
import time

import numpy as np

from pippy import annotate_split_points, split_into_equal_size
from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy import pipeline
from pippy.PipelineStage import PipelineStage
from pippy.PipelineSchedule import ScheduleGPipe

import torch
import torch.distributed as dist

from torchvision.models import resnet101
from tqdm import tqdm



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
    batch_size = per_chunk_sz * num_chunks

    seed(42)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device('cuda')
    torch.cuda.set_device(rank)

    module = resnet101().train().to(device)
    module.fc = torch.nn.Linear(module.fc.in_features, 1000).to(device)

    annotate_split_points(
        module,
        {
            # "layer1.2": PipeSplitWrapper.SplitPoint.END,
            # "layer2.2": PipeSplitWrapper.SplitPoint.END,
            "layer3": PipeSplitWrapper.SplitPoint.END,
        },
    )
    dataset_size = 10000
    train_dataloader = [
        (torch.randn(batch_size, 3, 224, 224, device=device), torch.randint(0, 10, (batch_size,), device=device))
    ] * (dataset_size // batch_size)

    module = pipeline(module, num_chunks, example_args=(train_dataloader[0][0],)) #, split_policy=split_into_equal_size(2))
    module = PipelineStage(module, rank, device)
    opt = torch.optim.Adam(module.submod.parameters(), foreach=True, capturable=True)
    module = ScheduleGPipe(module, num_chunks)

    def train_step(input, label, model, opt):
        if rank == 0:
            model.step(input)
        elif rank == world_size - 1:
            losses = []
            output = model.step(target=label, losses=losses)
        else:
            model.step()
        opt.step()
        return output, losses

    x_batch, y_batch = next(iter(train_dataloader))
    train_step(x_batch.to(device), y_batch.to(device), module, opt) # compile
    epochs = 1
    time_sum = 0
    for epoch in range(epochs):
        all_cnt, correct_cnt, loss_sum = 0, 0, 0
        time_start = time.time()
        for x_batch, y_batch in tqdm(train_dataloader, dynamic_ncols=True) if rank == 0 else train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            _ = train_step(x_batch, y_batch, module, opt)
        time_sum += time.time() - time_start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--num-chunks', type=int, default=2)
    args = parser.parse_args()
    test_main(args)
