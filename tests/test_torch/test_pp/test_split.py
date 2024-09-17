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

from functools import reduce
import random
from copy import deepcopy

import pytest
import torch
import torch.utils._pytree as pytree
from torchvision.models import (alexnet, densenet121, resnet18, swin_t, vgg19, efficientnet_b0,
                                vit_b_16)
from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points,
                                                             compile_pipeline)
from easydist.torch.utils import seed
from easydist.torch.compile import ed_compile_func
from easydist.torch.experimental.pp.runtime import ScheduleGPipe
from benchmark.torch.model import GPT

from transformers import OpenAIGPTModel, OpenAIGPTConfig

from tests.test_torch.test_utils import get_module_opt_states


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


def train_step(input, label, model, opt):
    out = model(input)
    out = out.view(out.shape[0], -1).mean(dim=1)
    loss = (out - label).pow(2).mean()
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return out


def train_step_hf(input, label, model, opt):
    if opt is not None:
        opt.zero_grad()
    out = model(input).last_hidden_state
    loss = out.mean()
    if opt is not None:
        loss.backward()
        opt.step()
    return out


def gen_rand_input_foo():
    return torch.rand(16, 1024)


def gen_rand_input_imagenet():
    return torch.rand(16, 3, 224, 224)

def gen_fix_len_embed(hidden_dim):
    def inner():
        return torch.ones(16, 512, hidden_dim, device='cuda')
    return inner

def factory_gen_rand_input_ids(vocab_size):

    def gen_rand_input_ids():
        return torch.randint(0, vocab_size, (3, 256))

    return gen_rand_input_ids

def gen_rand_input_vit():
    return torch.rand(16, 3, 224, 224)

def inner(module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func):
    device = torch.device("cuda")

    module_torch = module_cls(*module_init_args).to(device)
    module_compiled = deepcopy(module_torch)

    opt_torch = torch.optim.SGD(module_torch.parameters(), lr=1e-3, momentum=0.9, foreach=True)
    opt_compiled = torch.optim.SGD(module_compiled.parameters(), lr=1e-3, momentum=0.9, foreach=True)

    if isinstance(split_ann_or_policy, set):
        annotate_split_points(module_compiled, split_ann_or_policy)
        nstages = len(split_ann_or_policy) + 1
    else:
        nstages, module_compiled = split_ann_or_policy(module_compiled)

    rand_input = rand_input_gen_method().to(device)
    label = torch.tensor([random.random() for _ in range(rand_input.shape[0])]).to(device)

    train_step_func_args = (rand_input, label, module_compiled, opt_compiled)
    params, buffers, named_states, _, traced_stateless_func = ed_compile_func(train_step_func, 'fake', None, (rand_input, label, module_compiled, opt_compiled), {}, ScheduleGPipe, module_compiled, opt_compiled)
    stateless_func_args = (params, buffers, named_states, train_step_func_args, {})
    compiled_meta, compiled_stages, local_gm, _ = compile_pipeline(traced_stateless_func, nstages, stateless_func_args, strict=True)  # some models have unsed param

    epochs = 50
    dataset = []
    for _ in range(epochs):
        rand_input = rand_input_gen_method().to(device)
        label = torch.tensor([random.random() for _ in range(rand_input.shape[0])]).to(device)
        dataset.append((rand_input, label))

    seed()
    params_torch, buffers_torch, optimstates_torch = get_module_opt_states(module_torch, opt_torch)
    with torch.no_grad():
        for rand_input, label in dataset:
            args, kwargs = (rand_input, label, module_torch, opt_torch), {}
            params_torch, buffers_torch, optimstates_torch, _, return_torch = traced_stateless_func(params_torch, buffers_torch, optimstates_torch, args, kwargs)

    seed()
    with torch.no_grad():
        for rand_input, label in dataset:
            args, kwargs = (rand_input, label, None, None), {}
            args_kwargs_vals_flatten, _ = pytree.tree_flatten((args, kwargs))
            args_kwargs_nodes_flatten, _ = pytree.tree_flatten(
                (compiled_meta.args_nodes_unflatten, compiled_meta.kwargs_nodes_unflatten))
            input_node_vals = {node: val for node, val in zip(args_kwargs_nodes_flatten, args_kwargs_vals_flatten)}
            return_compiled = local_gm(**input_node_vals)
    params_compiled = reduce(lambda x, y: {**x, **y.named_parameters()}, compiled_stages, {})
    buffers_compiled = reduce(lambda x, y: {**x, **y.named_buffers()}, compiled_stages, {})
    optimstates_compiled = reduce(lambda x, y: {**x, **y._optimizer_state_dict()}, compiled_stages, {})

    assert torch.allclose(return_torch, return_compiled)

    for k in buffers_torch:
        assert torch.allclose(buffers_torch[k], buffers_compiled[k])

    for k in params_torch:
        assert torch.allclose(params_torch[k], params_compiled[k])

    for k in optimstates_torch:
        for kk in optimstates_torch[k]:
            assert torch.allclose(optimstates_torch[k][kk], optimstates_compiled[k][kk])


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func", [
    (Foo, {}, {'norm'}, gen_rand_input_foo, train_step),
    (Foo1, {}, {
        'norm',
        'linear0_1',
    }, gen_rand_input_foo, train_step),
    (alexnet, {}, {
        'features.10',
        'classifier.3',
    }, gen_rand_input_imagenet, train_step),
    (densenet121, {}, {
        'features.denseblock1.denselayer4.norm2',
        'features.transition2.conv',
        'features.denseblock4.denselayer1.relu1',
        'features',
    }, gen_rand_input_imagenet, train_step),
    (efficientnet_b0, {}, {
        'features.2.0.block.1',
        'features.4.1.block.3',
        'features.6.1.block.3',
        'features.8',
    }, gen_rand_input_imagenet, train_step),
    (resnet18, {}, {
        'layer1',
        'layer2',
        'layer3',
        'layer4',
    }, gen_rand_input_imagenet, train_step),
    (swin_t, {}, {
        'features.2.reduction',
        'features.3.0.mlp.1',
        'features.5.1.attn.qkv',
        'features.7.0.stochastic_depth',
    }, gen_rand_input_imagenet, train_step),
    (vgg19, {}, {
        'features.10',
        'features.20',
        'classifier.3',
    }, gen_rand_input_imagenet, train_step),
    (vit_b_16, {}, {
        'encoder.layers.encoder_layer_1.self_attention',
        'encoder.layers.encoder_layer_5.mlp.3',
        'encoder.layers.encoder_layer_9.ln_2',
    }, gen_rand_input_vit, train_step),
])
def test_vision(module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func):
    inner(module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func)


@pytest.mark.torch
@pytest.mark.parametrize("module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func", [
    (OpenAIGPTModel, (OpenAIGPTConfig(n_layer=4),), {
        'h.0',
        'h.1',
        'h.2',
    }, factory_gen_rand_input_ids(40478), train_step_hf),
    (GPT, (4, 32, 4), {
        'blocks.0',
        'blocks.1',
        'blocks.2'
    }, gen_fix_len_embed(32), train_step)
])
def test_split_language(module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func):
    inner(module_cls, module_init_args, split_ann_or_policy, rand_input_gen_method, train_step_func)
