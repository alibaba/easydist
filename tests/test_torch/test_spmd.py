# Copyright (c) 2024, Alibaba Group;
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
from functools import reduce

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.utils import seed
from easydist.utils.testing import spawn
from tests.test_torch.test_utils import (
    TEST_GPT,
    TEST_GPT_CASE,
    broadcast_module,
    get_module_opt_states,
    train_step,
    train_step_chunked,
)


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(1024)  # TODO @botbw: SPMD somehow doesn't support BatchNorm
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


BATCH_SIZE = 64


def inner(module_cls, optim, spmd_size):
    seed()
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    set_device_mesh(DeviceMesh("cuda", torch.arange(world_size).reshape(*spmd_size), mesh_dim_names=[f'spmd{i}' for i in range(len(spmd_size))]))

    device = torch.device("cuda")
    module_torch = module_cls().to(device)
    module_torch = broadcast_module(module_torch)
    module_auto = deepcopy(module_torch)

    rtol, atol = 1e-4, 1e-5  # TODO @botbw: spmd precision loss?
    if optim == 'adam':
        opt_torch = torch.optim.Adam(module_torch.parameters(), lr=1e-5, foreach=True, capturable=True)
        opt_auto = torch.optim.Adam(module_auto.parameters(), lr=1e-5, foreach=True, capturable=True)
    elif optim =='sgd':
        opt_torch = torch.optim.SGD(module_torch.parameters(), lr=1e-5, foreach=True, momentum=0.9)
        opt_auto = torch.optim.SGD(module_auto.parameters(), lr=1e-5, foreach=True, momentum=0.9)
    else:
        raise RuntimeError("Unknown optimizer")

    num_chunks = world_size * 4 if reduce(lambda a, b: a * b, spmd_size) != world_size else 1  # if pure spmd, disable microbatches
    assert BATCH_SIZE % num_chunks == 0, f"only support fix micro batch size {BATCH_SIZE=} {num_chunks=}"
    assert (BATCH_SIZE // num_chunks) % 2 == 0, f"microbatch size must be divided by 2 (spmd)"
    compiled_auto = easydist_compile(train_step, 'auto', 'fake', cuda_graph=False, schedule_cls=None, num_chunks=num_chunks, strict=True)

    steps = 5
    if module_cls is TEST_GPT:
        dataset = [
            torch.randn(BATCH_SIZE, TEST_GPT_CASE.seq_size, TEST_GPT_CASE.hidden_dim, device=device)
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
        out_auto = compiled_auto(data, module_auto, opt_auto)

        assert torch.allclose(out_torch, out_auto.to(device), rtol=rtol, atol=atol)

        params_torch, buffers_torch, optimstates_torch = get_module_opt_states(module_torch, opt_torch, False)
        params_compiled = compiled_auto.compiled_func.named_parameters()
        buffers_compiled = compiled_auto.compiled_func.named_buffers()
        optimstates_compiled = compiled_auto.compiled_func._optimizer_state_dict()
        for k in buffers_torch:
            assert torch.allclose(buffers_torch[k], buffers_compiled[k].to(device), rtol=rtol, atol=atol)
        for k in params_torch:
            assert torch.allclose(params_torch[k], params_compiled[k].to(device), rtol=rtol, atol=atol)
        for k in optimstates_compiled:  # states are not gathered when using native optimizer
            for kk in optimstates_torch[k]:
                assert torch.allclose(optimstates_torch[k][kk], optimstates_compiled[k][kk].to(device), rtol=rtol, atol=atol)


@pytest.mark.torch
@pytest.mark.world_2
@pytest.mark.long_duration
@pytest.mark.parametrize("module_cls, optim", [
    (Foo, 'adam'),
    (Foo, 'sgd'),
    (TEST_GPT, 'adam'),
    (TEST_GPT, 'sgd'),
])
@pytest.mark.timeout(100)
def test_runtime_world_2(module_cls, optim):
    spawn(inner, (module_cls, optim, (2, )), nprocs=2, port=12345)


@pytest.mark.torch
@pytest.mark.world_4
@pytest.mark.long_duration
@pytest.mark.parametrize("module_cls, optim, spmd_size", [
    (Foo, 'sgd', (4,)),
    (Foo, 'sgd', (2, 2)),
    (TEST_GPT, 'sgd', (4,)),
    (TEST_GPT, 'sgd', (2, 2)),
])
@pytest.mark.timeout(200)
def test_runtime_world_4(module_cls, optim, spmd_size):
    spawn(inner, (module_cls, optim, spmd_size), nprocs=4, port=12345)


@pytest.mark.torch
@pytest.mark.world_8
@pytest.mark.long_duration
@pytest.mark.parametrize("module_cls, optim, spmd_size", [
    (TEST_GPT, 'sgd', (8, )),
    (TEST_GPT, 'sgd', (4, 2)),
    (TEST_GPT, 'sgd', (2, 2, 2)),
])
@pytest.mark.timeout(200)
def test_runtime_world_8(module_cls, optim, spmd_size):
    spawn(inner, (module_cls, optim, spmd_size), nprocs=8, port=12345)
