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

import random

import numpy as np

import torch
import torch.distributed as dist

from torch.distributed._tensor import DeviceMesh
from tqdm import tqdm

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.utils import seed
from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import annotate_split_points
from easydist.utils.testing import spawn

import pytest

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


def main(schedule_cls):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    pp_size = world_size
    per_chunk_sz = 1
    num_chunks = 16
    batch_size = per_chunk_sz * num_chunks
    seed(42)
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

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

    dataset_size = 100
    train_dataloader = [(torch.randn(
        batch_size, 1024, device=device), torch.randint(0, 10, (batch_size, ), device=device))
                        ] * (dataset_size // batch_size)

    x_batch, y_batch = next(iter(train_dataloader))
    epochs = 1

    for _ in range(epochs):
        for x_batch, y_batch in tqdm(train_dataloader,
                                        dynamic_ncols=True) if rank == 0 else train_dataloader:
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            _ = train_step(x_batch, y_batch, module, opt)

@pytest.mark.torch
@pytest.mark.parametrize("schedule_cls", [ScheduleGPipe, ScheduleDAPPLE])
@pytest.mark.timeout(50)
def test_reslink(schedule_cls):
    spawn(main, (schedule_cls,), nprocs=4)
