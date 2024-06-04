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

# python tests/test_torch/test_pp/test_split.py
import copy
import os
import random
from contextlib import nullcontext
from functools import partial
from typing import cast

# make easydist happy without torchrun
os.environ['MASTER_PORT'] = '-1'

import numpy as np

import torch
import torch.utils._pytree as pytree
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


def train_step(input, label, model, opt):

    out = model(input)
    out = out.view(out.shape[0], -1).mean(dim=1)
    loss = (out - label).pow(2).mean()
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss


def train_step_gpt(input, label, model, opt):
    if opt is not None:
        opt.zero_grad()
    out = model(input)
    loss = 0
    for key in [
            'attentions', 'cross_attentions', 'hidden_states', 'pooler_output', 'pask_key_values',
            'last_hidden_state'
    ]:
        if hasattr(out, key) and (attr := getattr(out, key)) is not None:
            if isinstance(attr, torch.Tensor):
                attr_broadcast = attr.permute([i for i in range(attr.dim() - 1, -1, -1)])
                loss += (attr_broadcast - label).pow(2).mean()
            elif isinstance(attr, (tuple, list)):
                for a in attr:
                    attr_broadcast = a.permute([i for i in range(a.dim() - 1, -1, -1)])
                    loss += (attr_broadcast - label).pow(2).mean()
    if opt is not None:
        loss.backward()
        opt.step()
    return loss


def train_step_t5(input, label, model, opt):
    label = torch.ones_like(input)
    if opt is not None:
        opt.zero_grad()
    loss = model(input_ids=input, labels=label).loss
    if opt is not None:
        loss.backward()
        opt.step()
    return loss


def gen_rand_input_foo():
    return torch.rand(16, 1024)


def gen_rand_input_imagenet():
    return torch.rand(16, 3, 224, 224)


def factory_gen_rand_input_ids(vocab_size):

    def gen_rand_input_ids():
        return torch.randint(0, vocab_size, (3, 256))

    return gen_rand_input_ids


def gen_rand_input_vit():
    return torch.rand(16, 3, 224, 224).half()


def test_main(module, split_ann_or_policy, rand_input_gen_method, train_step_func):
    device = torch.device("cuda")
    module = module.train().to(device)
    opt_config = {
        'lr': 0.123456789,
        'momentum': 0.9,
        'foreach': True,
    }
    opt = None  # inference only
    # opt = torch.optim.Adam(module.parameters(), **opt_config)
    opt = torch.optim.SGD(module.parameters(), **opt_config)

    if isinstance(split_ann_or_policy, set):
        annotate_split_points(module, split_ann_or_policy)
        nstages = len(split_ann_or_policy) + 1
    else:
        nstages, module = split_ann_or_policy(module)

    rand_input = rand_input_gen_method().to(device)
    label = torch.tensor([random.random() for _ in range(rand_input.shape[0])]).to(device)
    args = (rand_input, label, module, opt)
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

    compiled_meta, compiled_stages, local_gm, erased_tensor_keys = compile_pipeline(
        traced_stateless_func, nstages, stateless_func_args, strict=True)

    epochs = 5
    dataset = []
    for _ in range(epochs):
        rand_input = rand_input_gen_method().to(device)
        label = torch.tensor([random.random() for _ in range(rand_input.shape[0])]).to(device)
        dataset.append((rand_input, label))

    seed()
    with torch.no_grad():
        for rand_input, label in dataset:
            args = (rand_input, label, module, opt)
            kwargs = {}
            args_kwargs_vals_flatten, _ = pytree.tree_flatten((args, kwargs))
            args_kwargs_nodes_flatten, _ = pytree.tree_flatten(
                (compiled_meta.args_nodes_unflatten, compiled_meta.kwargs_nodes_unflatten))
            input_node_vals = {}
            for node, val in zip(args_kwargs_nodes_flatten, args_kwargs_vals_flatten):
                input_node_vals[node] = val
            local_gm(**input_node_vals)

    outputs = {}
    for stage in compiled_stages:
        outputs.update(stage.outputs)

    params, buffers, optimstates, grads, returns = graph_outputs_to_func_outputs(compiled_meta,
                                                                                 outputs,
                                                                                 strict=False)
    returns = _to_tuple(returns)

    seed()
    with torch.no_grad():
        for rand_input, label in dataset:
            stateless_func_args_copy[3] = list(stateless_func_args_copy[3])
            stateless_func_args_copy[3][0] = rand_input
            stateless_func_args_copy[3][1] = label
            pararms_, buffers_, optimstates_, grads_, returns_ = traced_stateless_func(
                *stateless_func_args_copy)
            stateless_func_args_copy[:3] = [pararms_, buffers_, optimstates_]

    returns_ = _to_tuple(returns_)

    # key check
    assert set(params.keys()) == set(pararms_.keys())
    assert set(buffers.keys()) == set(buffers_.keys())
    assert set(optimstates.keys()) == set(optimstates_.keys())
    assert set(grads.keys()) == set(grads_.keys())
    assert len(returns) == len(returns_)

    # value check
    for k in params.keys():
        assert torch.allclose(params[k], pararms_[k])
    for k in buffers.keys():
        assert torch.allclose(buffers[k], buffers_[k])
    for k in optimstates.keys():
        for kk in optimstates[k].keys():
            assert torch.allclose(optimstates[k][kk], optimstates_[k][kk])
    for k in grads.keys():
        assert torch.allclose(grads[k], grads_[k])
    for i in range(len(returns)):
        assert torch.allclose(returns[i], returns_[i])

    print(f"Test passed for {module.__class__.__name__}")


if __name__ == '__main__':
    # human annotated split points
    test_main(Foo(), {'norm'}, gen_rand_input_foo, train_step)
    test_main(Foo1(), {
        'norm',
        'linear0_1',
    }, gen_rand_input_foo, train_step)
    test_main(alexnet(), {
        'features.10',
        'classifier.3',
    }, gen_rand_input_imagenet, train_step)
    test_main(
        densenet121(), {
            'features.denseblock1.denselayer4.norm2',
            'features.transition2.conv',
            'features.denseblock4.denselayer1.relu1',
            'features',
        }, gen_rand_input_imagenet, train_step)
    test_main(efficientnet_b0(), {
        'features.2.0.block.1',
        'features.4.1.block.3',
        'features.6.1.block.3',
        'features.8',
    }, gen_rand_input_imagenet, train_step)
    test_main(resnet18(), {
        'layer1',
        'layer2',
        'layer3',
        'layer4',
    }, gen_rand_input_imagenet, train_step)
    test_main(
        swin_t(), {
            'features.2.reduction',
            'features.3.0.mlp.1',
            'features.5.1.attn.qkv',
            'features.7.0.stochastic_depth',
        }, gen_rand_input_imagenet, train_step)
    test_main(vgg19(), {
        'features.10',
        'features.20',
        'classifier.3',
    }, gen_rand_input_imagenet, train_step)
    test_main(
        vit_b_16().half(), {
            'encoder.layers.encoder_layer_1.self_attention',
            'encoder.layers.encoder_layer_5.mlp.3',
            'encoder.layers.encoder_layer_9.ln_2',
        }, gen_rand_input_vit, train_step)

    # test split_into_equal_size
    test_main(Foo(), split_into_equal_size(2), gen_rand_input_foo, train_step)
    test_main(Foo1(), split_into_equal_size(2), gen_rand_input_foo, train_step)
    test_main(alexnet(), split_into_equal_size(3), gen_rand_input_imagenet, train_step)
    test_main(densenet121(), split_into_equal_size(5), gen_rand_input_imagenet, train_step)
    test_main(efficientnet_b0(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)
    test_main(resnet18(), split_into_equal_size(4), gen_rand_input_imagenet, train_step)
    test_main(swin_t(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)
    test_main(vgg19(), split_into_equal_size(3), gen_rand_input_imagenet, train_step)
    test_main(vit_b_16().half(), split_into_equal_size(10), gen_rand_input_vit, train_step)

    # ======== transformers ========
    from transformers import OpenAIGPTModel, OpenAIGPTConfig
    test_main(OpenAIGPTModel(OpenAIGPTConfig()), {
        'h.3',
        'h.6',
        'h.9',
    }, factory_gen_rand_input_ids(OpenAIGPTConfig().vocab_size), train_step_gpt)

    from transformers import AutoModel
    test_main(AutoModel.from_pretrained("bert-base-uncased"), {
        'encoder.layer.3',
        'encoder.layer.6',
        'encoder.layer.9',
    }, factory_gen_rand_input_ids(30522), train_step_gpt)

    from transformers import GPT2Model, GPT2Config
    test_main(GPT2Model(GPT2Config()), {
        'h.3',
        'h.6',
        'h.9',
    }, factory_gen_rand_input_ids(50257), train_step_gpt)

    from transformers import LlamaModel, LlamaConfig
    config = LlamaConfig()
    config.num_attention_heads = config.num_key_value_heads = 16
    config.num_hidden_layers = 16
    config.hidden_size = 768
    config.use_cache = False
    test_main(LlamaModel(config), {
        'layers.3',
        'layers.7',
        'layers.11',
    }, factory_gen_rand_input_ids(config.vocab_size), train_step_gpt)

    print("All tests passed!")
