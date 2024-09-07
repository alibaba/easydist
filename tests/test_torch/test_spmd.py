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

from copy import deepcopy

import pytest
import torch.distributed as dist

import easydist.config as mdconfig
from easydist import easydist_setup
from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.api import easydist_compile
from easydist.torch.experimental.pp.runtime import ScheduleGPipe


import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.distributed._tensor import DeviceMesh

from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points)
from easydist.utils.testing import spawn


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(1024)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


class Foo1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(1024)
        self.linear0_0 = torch.nn.Linear(1024, 512)
        self.linear0_1 = torch.nn.Linear(512, 256)
        self.linear1 = torch.nn.Linear(256, 1024)

    def forward(self, x):
        x = self.norm(x)
        x0 = self.linear0_0(x)
        x0 = self.linear0_1(x0)
        x1 = self.linear1(x0)
        y = x + x1
        return y.relu()


def train_step(input, model, opt):
    out = model(input)
    loss = out.mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())

    return model



def inner(module_cls, split_ann, schedule_cls):
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    mdconfig.comm_optimization = False  # reduce compile time
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    set_device_mesh(DeviceMesh("cuda", torch.arange(2), mesh_dim_names=['spmd']))

    device = torch.device("cuda")
    module_torch = module_cls().to(device)
    module_torch = broadcast_module(module_torch)
    module_pipe = deepcopy(module_torch)
    annotate_split_points(module_pipe, split_ann)
    opt_torch = torch.optim.Adam(module_torch.parameters(), lr=0.12345, foreach=True, capturable=True)
    opt_pipe = torch.optim.Adam(module_pipe.parameters(), lr=0.12345, foreach=True, capturable=True)

    compiled_pipe = easydist_compile(train_step, 'auto', 'fake', cuda_graph=False, schedule_cls=None, num_chunks=1)

    steps = 2
    dataset = [
        torch.randn(1024, 1024, device=device)
        for _ in range(steps)
    ]
    for data in dataset:
        dist.broadcast(data, src=0)

    for data in dataset:
        out_pipe = compiled_pipe(data, module_pipe, opt_pipe).mean()
        out_torch = train_step(data, module_torch, opt_torch)

    assert torch.allclose(out_torch, out_pipe)


# @pytest.mark.skip  # this test sometimes fails
@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls", [
    (Foo, {'norm'}, ScheduleGPipe),
    (Foo1, {'linear0_1'}, ScheduleGPipe),
])
@pytest.mark.timeout(100)
def test_auto(module_cls, split_ann, schedule_cls):
    spawn(inner, (module_cls, split_ann, schedule_cls), nprocs=2)


if __name__ == '__main__':
    test_auto()