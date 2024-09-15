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
from torch.distributed._tensor import DeviceMesh

from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points)
from easydist.torch.utils import seed
from easydist.utils.testing import spawn

from tests.test_torch.test_utils import train_step, train_step_chunked, broadcast_module, Foo

def inner(module_cls, split_ann, schedule_cls, optim, use_native_optimizer):
    seed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    set_device_mesh(DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=['pp']))

    device = torch.device("cuda")
    module_torch = module_cls().to(device)
    module_torch = broadcast_module(module_torch)
    module_pipe = deepcopy(module_torch)
    annotate_split_points(module_pipe, split_ann)
    if optim == 'adam':
        opt_torch = torch.optim.Adam(module_torch.parameters(), lr=1e-5, foreach=True, capturable=True)
        opt_pipe = torch.optim.Adam(module_pipe.parameters(), lr=1e-5, foreach=True, capturable=True)
    elif optim =='sgd':
        opt_torch = torch.optim.SGD(module_torch.parameters(), lr=1e-5, foreach=True, momentum=0.9)
        opt_pipe = torch.optim.SGD(module_pipe.parameters(), lr=1e-5, foreach=True, momentum=0.9)
    else:
        raise RuntimeError("Unknown optimizer")

    num_chunks = world_size * 8
    compiled_pipe = easydist_compile(train_step, 'pp', 'fake', cuda_graph=False, schedule_cls=schedule_cls, num_chunks=num_chunks, strict=False)

    steps = 2
    dataset = [
        torch.randn(32, 512, 128, device=device)
        for _ in range(steps)
    ]

    for data in dataset:
        dist.broadcast(data, src=0)

    for _, data in enumerate(dataset):
        out_torch = train_step_chunked(data, module_torch, opt_torch, num_chunks).mean()

        if use_native_optimizer:
            out_pipe = compiled_pipe(data, module_pipe, None).mean()
            opt_pipe.step()
            opt_pipe.zero_grad()
        else:
            out_pipe = compiled_pipe(data, module_pipe, opt_pipe).mean()

        assert torch.allclose(out_torch.to(device), out_pipe.to(device), rtol=1e-5, atol=1e-8)

    state_torch = module_torch.state_dict()
    state_pipe = compiled_pipe.compiled_func.state_dict()
    assert set(state_torch.keys()) == set(state_pipe.keys())
    for k in state_torch.keys():
        assert torch.allclose(state_pipe[k].to(device), state_torch[k].to(device), rtol=1e-4, atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls, optim, use_native_optimizer", [
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'adam', True),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'adam', False),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'sgd', True),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'sgd', False),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'adam', True),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'adam', False),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'sgd', True),
    (Foo, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'sgd', False),

])
@pytest.mark.timeout(50)
def test_runtime(module_cls, split_ann, schedule_cls, optim, use_native_optimizer):
    spawn(inner, (module_cls, split_ann, schedule_cls, optim, use_native_optimizer), nprocs=4, port=12344)
