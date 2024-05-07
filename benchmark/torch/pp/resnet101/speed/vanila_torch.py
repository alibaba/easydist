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
import random
import time

import numpy as np

import torch

from torchvision import datasets, transforms
from torchvision.models import resnet101
from torch.profiler import profile, record_function, ProfilerActivity

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


def train_step(input, label, model, opt):
    out = model(input)
    loss = criterion(out, label)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return out, loss


def test_main():
    seed(42)

    device = torch.device('cuda')

    module = resnet101().train().to(device)
    module.fc = torch.nn.Linear(2048, 10).to(device)

    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)

    dataset_size = 10000
    batch_size = 128
    train_dataloader = [(torch.randn(batch_size, 3, 224, 224), torch.randint(
        0, 10, (batch_size, )))] * (dataset_size // batch_size)

    x_batch, y_batch = next(iter(train_dataloader))
    train_step(x_batch.to(device), y_batch.to(device), module, opt)
    epochs = 1
    for epoch in range(epochs):
        all_cnt, correct_cnt, loss_sum = 0, 0, 0
        time_start = time.time()
        for x_batch, y_batch in tqdm(train_dataloader, dynamic_ncols=True):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            out, loss = train_step(x_batch, y_batch, module, opt)
            all_cnt += len(out)
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch).sum()
            loss_sum += loss.mean().item()
    print(
        f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}, loss sum {loss_sum}, avg loss: {loss_sum / all_cnt} '
        f'time: {time.time() - time_start}'
        f'max memory: {torch.cuda.max_memory_allocated() / 1024 / 1024}mb')


if __name__ == '__main__':
    test_main()
