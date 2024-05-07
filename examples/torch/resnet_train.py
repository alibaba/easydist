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

# ENABLE_COMPILE_CACHE=1 torchrun --nproc_per_node 4 examples/torch/resnet_train.py
import os
import random

import numpy as np

import torch
import torch.distributed as dist

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.distributed._tensor import DeviceMesh
from tqdm import tqdm

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import get_pp_size, set_device_mesh
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE
from easydist.torch.experimental.pp.compile_pipeline import (
    split_into_equal_size)


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


@easydist_compile(tracing_mode="fake",
                  cuda_graph=False,
                  schedule_cls=ScheduleDAPPLE,
                  num_chunks=4)
def train_step(input, label, model, opt):
    opt.zero_grad()
    out = model(input)
    loss = criterion(out, label)
    loss.backward()
    opt.step()
    return out, loss


def test_main():
    seed(42)
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    set_device_mesh(
        DeviceMesh("cuda", [[[0, 2], [1, 3]]],
                   mesh_dim_names=["spmd0", "spmd1", "pp"]))

    device = torch.device('cuda')

    module = resnet18().train().to(device)
    module.fc = torch.nn.Linear(512, 10).to(device)
    pp_size = get_pp_size()
    _, module = split_into_equal_size(pp_size)(module)

    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)
    # opt = torch.optim.SGD(module.parameters(), lr=0.001, foreach=True)

    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_data = datasets.CIFAR10('./data',
                                  train=True,
                                  download=True,
                                  transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=batch_size)

    def validation(module, valid_dataloader, epoch, state_dict):
        module.load_state_dict(state_dict)
        module.to(rank)
        module.eval()
        correct_cnt = 0
        all_cnt = 0
        for x_batch, y_batch in tqdm(valid_dataloader, dynamic_ncols=True):
            x_batch = x_batch.to(f'cuda:{rank}')
            y_batch = y_batch.to(f'cuda:{rank}')
            out = module(x_batch)
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch).sum()
            all_cnt += len(y_batch)
        print(f'epoch {epoch} valid accuracy: {correct_cnt / all_cnt}')

    epochs = 5
    for epoch in range(epochs):
        all_cnt, correct_cnt, loss_sum = 0, 0, 0
        for x_batch, y_batch in (tqdm(train_dataloader, dynamic_ncols=True) if rank == 0
                                    else train_dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            out, loss = train_step(x_batch, y_batch, module, opt)
            all_cnt += len(out)
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch.to(f'cuda:{rank}')).sum()
            loss_sum += loss.mean().item()
        state_dict_list = train_step.compiled_func.state_dict(world_rank=0)
        if rank == 0:
            print(
                f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}, loss sum {loss_sum}, avg loss: {loss_sum / all_cnt}'
            )
            state_dict = {}
            for st in state_dict_list:
                state_dict.update(st)
            validation(module, valid_dataloader, epoch, state_dict)

    print(f"rank {rank} peek memory: {torch.cuda.max_memory_allocated()}")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(cur_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    state_dict_list = train_step.compiled_func.state_dict(0)
    if rank == 0:
        state_dict = {}
        for st in state_dict_list:
            state_dict.update(st)
        torch.save(state_dict, os.path.join(ckpt_dir, 'resnet.pth'))


if __name__ == '__main__':
    test_main()
