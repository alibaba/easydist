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

# torchrun --nproc_per_node 2 tests/test_torch/test_pp/test_runtime.py
import copy
import os
import random
from contextlib import nullcontext
from functools import partial
from typing import cast

import torch.distributed

from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.api import easydist_compile
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE

import numpy as np

import torch
import torch.utils._pytree as pytree
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.distributed._tensor import DeviceMesh

from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import FakeTensor
from torchvision.models import (alexnet, densenet121, efficientnet_b0, resnet18, swin_t, vgg19,
                                vit_b_16)
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, annotate_split_points,
                                                             compile_pipeline,
                                                             graph_outputs_to_func_outputs,
                                                             split_into_equal_size,
                                                             set_backward_flag)
from easydist.utils import rgetattr, rsetattr
from torch.fx.experimental.proxy_tensor import make_fx
# from easydist.torch.experimental.pp.ed_make_fx import ed_make_fx
from easydist.torch.experimental.pp.utils import _to_tuple, save_graphviz_dot
from easydist.torch.experimental.pp.split_utils import set_updated_params_states, get_updated_params_states
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])

def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
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
    out = out.view(out.shape[0], -1).mean(dim=1)
    loss = out.mean()
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss

def gen_rand_input_foo():
    return torch.rand(1024, 1024)


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())

    return model


def test_main(module, split_ann, rand_input_gen_method, train_step_func):
    device = torch.device("cuda")
    module = module.train().to(device)
    module = broadcast_module(module)
    annotate_split_points(module, split_ann)
    opt_config = {
        'lr': 0.123456789,
        'foreach': True,
        'capturable': True
    }
    opt = None  # inference only
    opt = torch.optim.Adam(module.parameters(), **opt_config)
    rand_input = rand_input_gen_method().to(device)
    args = (rand_input, module, opt)
    kwargs = {}

    # Copied from _compile
    ##################################################################################################
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    named_states = {}
    if opt is not None:
        # assign grad and warm up optimizer
        mode = nullcontext()
        for name in dict(module.named_parameters()):
            with torch.no_grad():
                rsetattr(module, name + ".grad", torch.zeros_like(rgetattr(module, name).data))
                if isinstance(rgetattr(module, name).data, FakeTensor):
                    mode = rgetattr(module, name).data.fake_mode

        with _enable_compile(), mode:
            opt.step()
            opt.zero_grad(True)

        for n, p in params.items():
            if p in opt.state:
                named_states[n] = opt.state[p]  # type: ignore[index]
                # if step in state, reduce one for warmup step.
                if 'step' in named_states[n]:
                    named_states[n]['step'] -= 1

    flat_named_states, named_states_spec = pytree.tree_flatten(named_states)

    # fix for sgd withtout momentum
    if all(state is None for state in flat_named_states):
        named_states = {}
        flat_named_states, named_states_spec = pytree.tree_flatten(named_states)

    def stateless_func(func, params, buffers, named_states, args, kwargs):
        set_updated_params_states(params, named_states)
        with stateless._reparametrize_module(
                cast(torch.nn.Module, module), {
                    **params,
                    **buffers
                }, tie_weights=True) if module else nullcontext(), _rematerialize_optimizer(
                    opt, named_states, params) if opt else nullcontext():
            ret = func(*args, **kwargs)
        params, named_states = get_updated_params_states()
        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    with _enable_compile(), SplitPatcher(module, opt):
        set_backward_flag(False)
        traced_stateless_func = make_fx(partial(stateless_func, train_step_func),
                                        tracing_mode='fake',
                                        decomposition_table=EASYDIST_DECOMP_TABLE,
                                        _allow_non_fake_inputs=False)(params, buffers,
                                                                      named_states, args, kwargs)

    traced_stateless_func.graph.eliminate_dead_code()
    traced_stateless_func = preprocess_traced_graph(traced_stateless_func)
    traced_stateless_func.recompile()
    ##################################################################################################

    save_graphviz_dot(traced_stateless_func, 'traced_graph')

    stateless_func_args = [params, buffers, named_states, args, kwargs]

    def arg_copy_func(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        else:
            return x

    stateless_func_args_copy = pytree.tree_map(arg_copy_func, stateless_func_args)

    pipe = easydist_compile(train_step_func, 'pp', 'fake', cuda_graph=False, schedule_cls=ScheduleDAPPLE, num_chunks=1)

    epochs = 10
    dataset = []
    for _ in range(epochs):
        rand_input = rand_input_gen_method().to(device)
        torch.distributed.broadcast(rand_input, src=0)
        dataset.append(rand_input)

    seed()
    for rand_input in dataset:
        args = (rand_input, module, opt)
        returns = pipe(*args)

    seed()
    with torch.no_grad():
        for rand_input in dataset:
            stateless_func_args_copy[3] = list(stateless_func_args_copy[3])
            stateless_func_args_copy[3][0] = rand_input
            pararms_, buffers_, optimstates_, grads_, returns_ = traced_stateless_func(
                *stateless_func_args_copy)
            stateless_func_args_copy[:3] = [pararms_, buffers_, optimstates_]

    assert torch.allclose(returns, returns_)
    print(f"Test passed for {module.__class__.__name__}")


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    set_device_mesh(DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=['pp']))
    
    # models need to be determined here (no dropout)
    test_main(Foo(), {
        'norm'
    }, gen_rand_input_foo, train_step)
    test_main(Foo1(), {
        'linear0_1'
    }, gen_rand_input_foo, train_step)
    print("All tests passed!")