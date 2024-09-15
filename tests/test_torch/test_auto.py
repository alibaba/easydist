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
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE


import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.distributed._tensor import DeviceMesh

from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points)
from easydist.utils.testing import spawn

from tests.test_torch.test_utils import train_step, train_step_chunked, broadcast_module

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
        self.linear0 = torch.nn.Linear(1024, 512)
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x.relu()


def inner(module_cls, split_ann, schedule_cls, pp_size):
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    mdconfig.comm_optimization = False  # reduce compile time
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    set_device_mesh(DeviceMesh("cuda", torch.arange(4).reshape(pp_size, -1), mesh_dim_names=['pp', 'spmd']))

    device = torch.device("cuda")
    module_torch = module_cls().to(device)
    module_torch = broadcast_module(module_torch)
    module_pipe = deepcopy(module_torch)
    if split_ann is not None and schedule_cls is not None:
        annotate_split_points(module_pipe, split_ann)
    opt_torch = torch.optim.Adam(module_torch.parameters(), lr=0.12345, foreach=True, capturable=True)
    opt_pipe = torch.optim.Adam(module_pipe.parameters(), lr=0.12345, foreach=True, capturable=True)

    num_chunks = world_size * 8
    if pp_size == world_size:  # TODO @botbw: auto should behave like pp when pp_size == world_size
        compiled_pipe = easydist_compile(train_step, 'pp', 'fake', cuda_graph=False, schedule_cls=schedule_cls, num_chunks=num_chunks)
    else:
        compiled_pipe = easydist_compile(train_step, 'auto', 'fake', cuda_graph=False, schedule_cls=schedule_cls, num_chunks=num_chunks)

    steps = 2
    dataset = [
        torch.randn(1024, 1024, device=device)
        for _ in range(steps)
    ]
    for data in dataset:
        dist.broadcast(data, src=0)

    for data in dataset:
        out_pipe = compiled_pipe(data, module_pipe, opt_pipe).mean()
        if split_ann is not None and schedule_cls is not None:
            out_torch = train_step_chunked(data, module_torch, opt_torch, num_chunks).mean()
        else:
            out_torch = train_step(data, module_torch, opt_torch).mean()

        assert torch.allclose(out_torch.to(device), out_pipe.to(device))


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls, pp_size", [
    (Foo, None, None, 1),
    (Foo, {'norm'}, ScheduleDAPPLE, 2),
    (Foo1, {'linear1'}, ScheduleDAPPLE, 2),
    (Foo1, {'norm', 'linear0', 'linear1'}, ScheduleDAPPLE, 4),
])
@pytest.mark.timeout(100)
def test_auto(module_cls, split_ann, schedule_cls, pp_size):
    spawn(inner, (module_cls, split_ann, schedule_cls, pp_size), nprocs=4)


if __name__ == '__main__':
    test_auto()