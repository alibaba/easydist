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

from easydist.torch.experimental.pp.compile_pipeline import annotate_split_points
from easydist.torch.utils import seed
from easydist.utils.testing import spawn

from tests.test_torch.test_utils import get_module_opt_states, train_step, train_step_chunked, broadcast_module, TEST_GPT


class Foo1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(1024)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


class Foo2(torch.nn.Module):

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


BATCH_SIZE = 48


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
    rtol, atol = 1e-5, 1e-8
    if optim == 'adam':
        opt_torch = torch.optim.Adam(module_torch.parameters(), lr=1e-5, foreach=True, capturable=True)
        opt_pipe = torch.optim.Adam(module_pipe.parameters(), lr=1e-5, foreach=True, capturable=True)
        if not use_native_optimizer:  # TODO @botbw: traced Adam suffers precision loss?
            rtol, atol = 1e-4, 1e-4
    elif optim =='sgd':
        opt_torch = torch.optim.SGD(module_torch.parameters(), lr=1e-5, foreach=True, momentum=0.9)
        opt_pipe = torch.optim.SGD(module_pipe.parameters(), lr=1e-5, foreach=True, momentum=0.9)
    else:
        raise RuntimeError("Unknown optimizer")

    num_chunks = world_size * 4
    assert BATCH_SIZE % num_chunks == 0, f"only support fix micro batch size {BATCH_SIZE=} {num_chunks=}"
    compiled_pipe = easydist_compile(train_step, 'pp', 'fake', cuda_graph=False, schedule_cls=schedule_cls, num_chunks=num_chunks, strict=False)

    steps = 50
    if module_cls is TEST_GPT:
        dataset = [
            torch.randn(BATCH_SIZE, 512, 128, device=device)
            for _ in range(steps)
        ]
    else:
        dataset = [
            torch.randn(BATCH_SIZE, 1024, device=device)
            for _ in range(steps)
        ]

    for data in dataset:
        dist.broadcast(data, src=0)

    for _, data in enumerate(dataset):
        out_torch, _, _ = train_step_chunked(data, module_torch, opt_torch, num_chunks)

        if use_native_optimizer:
            out_pipe = compiled_pipe(data, module_pipe, None)
            opt_pipe.step()
            opt_pipe.zero_grad()
        else:
            out_pipe = compiled_pipe(data, module_pipe, opt_pipe)

        assert torch.allclose(out_torch, out_pipe.to(device), rtol=rtol, atol=atol)

        params_torch, buffers_torch, optimstates_torch = get_module_opt_states(module_torch, opt_torch, False)
        params_compiled = compiled_pipe.compiled_func.named_parameters()
        buffers_compiled = compiled_pipe.compiled_func.named_buffers()
        if use_native_optimizer:
            _, _, optimstates_compiled = get_module_opt_states(module_pipe, opt_pipe, False)
        else:
            optimstates_compiled = compiled_pipe.compiled_func._optimstate_state_dict()
        for k in buffers_torch:
            assert torch.allclose(buffers_torch[k], buffers_compiled[k].to(device), rtol=rtol, atol=atol)
        for k in params_torch:
            assert torch.allclose(params_torch[k], params_compiled[k].to(device), rtol=rtol, atol=atol)
        for k in optimstates_compiled:  # states are not gathered when using native optimizer
            for kk in optimstates_torch[k]:
                assert torch.allclose(optimstates_torch[k][kk], optimstates_compiled[k][kk].to(device), rtol=rtol, atol=atol)


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls, optim, use_native_optimizer", [
    (Foo1, {'norm'}, ScheduleGPipe, 'adam', True),
    (Foo1, {'norm'}, ScheduleGPipe, 'adam', False),
    (Foo1, {'norm'}, ScheduleGPipe, 'sgd', True),
    (Foo1, {'norm'}, ScheduleGPipe, 'sgd', False),
    (Foo1, {'norm'}, ScheduleDAPPLE, 'adam', True),
    (Foo1, {'norm'}, ScheduleDAPPLE, 'adam', False),
    (Foo1, {'norm'}, ScheduleDAPPLE, 'sgd', True),
    (Foo1, {'norm'}, ScheduleDAPPLE, 'sgd', False),
])
@pytest.mark.timeout(100)
def test_runtime_world_2(module_cls, split_ann, schedule_cls, optim, use_native_optimizer):
    spawn(inner, (module_cls, split_ann, schedule_cls, optim, use_native_optimizer), nprocs=2, port=12345)


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls, optim, use_native_optimizer", [
    (Foo2, {'norm', 'linear0_1',}, ScheduleGPipe, 'adam', True),
    (Foo2, {'norm', 'linear0_1',}, ScheduleGPipe, 'adam', False),
    (Foo2, {'norm', 'linear0_1',}, ScheduleGPipe, 'sgd', True),
    (Foo2, {'norm', 'linear0_1',}, ScheduleGPipe, 'sgd', False),
    (Foo2, {'norm', 'linear0_1',}, ScheduleDAPPLE, 'adam', True),
    (Foo2, {'norm', 'linear0_1',}, ScheduleDAPPLE, 'adam', False),
    (Foo2, {'norm', 'linear0_1',}, ScheduleDAPPLE, 'sgd', True),
    (Foo2, {'norm', 'linear0_1',}, ScheduleDAPPLE, 'sgd', False),
])
@pytest.mark.timeout(100)
def test_runtime_world_3(module_cls, split_ann, schedule_cls, optim, use_native_optimizer):
    spawn(inner, (module_cls, split_ann, schedule_cls, optim, use_native_optimizer), nprocs=3, port=12345)


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, split_ann, schedule_cls, optim, use_native_optimizer", [
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'adam', True),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'adam', False),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'sgd', True),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleGPipe, 'sgd', False),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'adam', True),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'adam', False),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'sgd', True),
    (TEST_GPT, {'blocks.0', 'blocks.1', 'blocks.2'}, ScheduleDAPPLE, 'sgd', False),
])
@pytest.mark.timeout(100)
def test_runtime_world_4(module_cls, split_ann, schedule_cls, optim, use_native_optimizer):
    spawn(inner, (module_cls, split_ann, schedule_cls, optim, use_native_optimizer), nprocs=4, port=12345)
