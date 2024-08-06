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

from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.api import easydist_compile
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE, ScheduleGPipe


import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.distributed._tensor import DeviceMesh

from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points)
from easydist.utils.testing import spawn


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(1024)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


class Foo1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(1024)
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
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    set_device_mesh(DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=['pp']))

    device = torch.device("cuda")
    module_torch = module_cls().to(device)
    module_torch = broadcast_module(module_torch)
    module_pipe = deepcopy(module_torch)
    annotate_split_points(module_pipe, split_ann)
    opt_torch = torch.optim.Adam(module_torch.parameters(), lr=0.12345, foreach=True, capturable=True)
    opt_pipe = torch.optim.Adam(module_pipe.parameters(), lr=0.12345, foreach=True, capturable=True)

    compiled_pipe = easydist_compile(train_step, 'pp', 'fake', cuda_graph=False, schedule_cls=schedule_cls, num_chunks=1)

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

    state_torch = module_torch.state_dict()
    state_pipe = compiled_pipe.compiled_func.state_dict()
    assert set(state_torch.keys()) == set(state_pipe.keys())
    for k in state_torch.keys():
        torch.testing.assert_allclose(state_torch[k].to(device), state_pipe[k].to(device).detach())
        # TODO @botbw: torch.allclose failed for some reason
        # assert torch.allclose(state_orig[k].to(device), state_pipe[k].to(device).detach()), f"{k}: {state_orig[k]} != {state_pipe[k]}"

    # TODO @botbw: pp optimizer state_dict interface not sync with torch
    # optim_state_torch = opt_orig.state_dict()['state']
    # optim_state_pipe = compiled_pipe.compiled_func.state_dict()
    # assert set(optim_state_torch.keys()) == set(optim_state_pipe.keys()), f"{optim_state_torch.keys()} != {optim_state_pipe.keys()}"
    # for k in optim_state_torch.keys():
    #     assert set(optim_state_torch[k].keys()) == set(optim_state_pipe[k].keys())
    #     for kk in optim_state_torch[k].keys():
    #         assert torch.allclose(optim_state_torch[k][kk].to(device), optim_state_pipe[k][kk].to(device))


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls", [
    (Foo, {'norm'}, ScheduleGPipe),
    (Foo1, {'linear0_1'}, ScheduleDAPPLE),
    (Foo1, {'linear0_1'}, ScheduleGPipe),
    (Foo1, {'linear0_0'}, ScheduleDAPPLE),
])
@pytest.mark.timeout(50)
def test_runtime(module_cls, split_ann, schedule_cls):
    spawn(inner, (module_cls, split_ann, schedule_cls), nprocs=2)


if __name__ == '__main__':
    test_runtime()