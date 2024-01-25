import os
import random
from contextlib import nullcontext
from functools import partial
from typing import cast
# make easydist happy without torchrun
os.environ['MASTER_PORT'] = '-1'

import numpy as np

from transformers import OpenAIGPTConfig, OpenAIGPTModel, AutoModel, LlamaModel, LlamaConfig, T5Tokenizer, T5ForConditionalGeneration, T5ForConditionalGeneration

import torch
import torch.utils._pytree as pytree
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import FakeTensor
from torchvision.models import (alexnet, densenet121, efficientnet_b0, resnet18, swin_t, vgg19,
                                vit_b_16)
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, annotate_split_points,
                                                             PipeSplitWrapper,
                                                             compile_stateful_stages,
                                                             split_into_equal_size)
from easydist.utils import rgetattr, rsetattr
from easydist.torch.experimental.pp.ed_make_fx import ed_make_fx
from easydist.torch.experimental.pp.utils import save_graphviz_dot
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
        self.norm = torch.nn.LayerNorm(1024)
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
    if opt is not None:
        opt.zero_grad()
    out = model(input)
    loss = (out - torch.ones_like(out) * label).pow(2).mean()
    loss.backward()
    if opt is not None:
        opt.step()
    return loss


def train_step_gpt(input, label, model, opt):
    if opt is not None:
        opt.zero_grad()
    out = model(input).last_hidden_state
    loss = (out - torch.ones_like(out) * label).pow(2).mean()
    loss.backward()
    if opt is not None:
        opt.step()
    return loss


def train_step_t5(input, label, model, opt):
    label = torch.ones_like(input)
    if opt is not None:
        opt.zero_grad()
    loss = model(input_ids=input, labels=label).loss
    loss.backward()
    if opt is not None:
        opt.step()
    return loss


def test_main(module, split_ann_or_policy, rand_input_gen_method, train_step_func):
    module = module.train().double().cuda()
    # opt = None
    # opt = torch.optim.Adam(module.parameters(), lr=0.123456789, foreach=True, capturable=True)
    opt = torch.optim.SGD(module.parameters(), lr=0.123456789, foreach=True, momentum=0.9)
    if isinstance(split_ann_or_policy, set):
        annotate_split_points(module, split_ann_or_policy)
    else:
        module = split_ann_or_policy(module)

    rand_input = rand_input_gen_method()
    args = [rand_input, 0.0012345, module, opt]
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
        with mode:
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
        with stateless._reparametrize_module(
                cast(torch.nn.Module, module), {
                    **params,
                    **buffers
                }, tie_weights=True) if module else nullcontext(), _rematerialize_optimizer(
                    opt, named_states, params) if opt else nullcontext():
            ret = func(*args, **kwargs)

        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    with _enable_compile(), SplitPatcher(module, opt):
        traced_graph = ed_make_fx(partial(stateless_func, train_step_func),
                                  tracing_mode='fake',
                                  decomposition_table=EASYDIST_DECOMP_TABLE,
                                  _allow_non_fake_inputs=False)(params, buffers, named_states,
                                                                args, kwargs)

    traced_graph.graph.eliminate_dead_code()
    traced_graph = preprocess_traced_graph(traced_graph)
    traced_graph.recompile()
    ##################################################################################################

    # print("traced_graph:\n", traced_graph.code)
    save_graphviz_dot(traced_graph, 'traced_graph')

    args_unflatten = [params, buffers, named_states, args, kwargs]

    def arg_copy_func(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        else:
            return x

    args_copy = pytree.tree_map(arg_copy_func, args_unflatten)
    args_flatten, args_spec = pytree.tree_flatten(args_unflatten)

    idx2phname, outname2idx, compiled_stages, gm = compile_stateful_stages(
        module, traced_graph, args_flatten, args_spec)

    id_rand_input = -1
    for i, arg in enumerate(args_flatten):
        if arg is rand_input:
            id_rand_input = i
            break

    rand_input1 = rand_input_gen_method()

    seed()
    with torch.no_grad():
        gm(**{idx2phname[id_rand_input]: rand_input})
        gm(**{idx2phname[id_rand_input]: rand_input1})

    outputs = {}
    for stage in compiled_stages:
        outputs.update(stage.outputs)

    out_flatten = [None] * len(outname2idx)
    for name in outputs:
        out_flatten[outname2idx[name]] = outputs[name]

    seed()
    with torch.no_grad():
        out_copy = traced_graph(*args_copy)
        args_copy[:3] = out_copy[:3]
        args_copy[3][0] = rand_input1
        out_copy = traced_graph(*args_copy)

    out_flatten_copy, _ = pytree.tree_flatten(out_copy)

    for i, (val, val_copy) in enumerate(zip(out_flatten, out_flatten_copy)):
        assert val is not val_copy
        if isinstance(val, torch.Tensor):
            assert torch.allclose(val, val_copy)
        else:
            assert val == val_copy


def gen_rand_input_foo():
    return torch.rand(16, 1024).cuda().double()


def gen_rand_input_imagenet():
    return torch.rand(16, 3, 224, 224).cuda().double()


def factory_gen_rand_input_ids(vocab_size):

    def gen_rand_input_ids():
        return torch.randint(0, vocab_size, (3, 200)).long().cuda()

    return gen_rand_input_ids


if __name__ == '__main__':
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
        vit_b_16(), {
            'encoder.layers.encoder_layer_1.self_attention',
            'encoder.layers.encoder_layer_5.mlp.3',
            'encoder.layers.encoder_layer_9.ln_2',
        }, gen_rand_input_imagenet, train_step)
    test_main(OpenAIGPTModel(OpenAIGPTConfig()), {
        'h.3',
        'h.6',
        'h.9',
    }, factory_gen_rand_input_ids(OpenAIGPTConfig().vocab_size), train_step_gpt)

    test_main(Foo(), split_into_equal_size(2), gen_rand_input_foo, train_step)
    test_main(Foo1(), split_into_equal_size(2), gen_rand_input_foo, train_step)
    test_main(alexnet(), split_into_equal_size(3), gen_rand_input_imagenet, train_step)
    test_main(densenet121(), split_into_equal_size(5), gen_rand_input_imagenet, train_step)
    test_main(efficientnet_b0(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)
    test_main(resnet18(), split_into_equal_size(4), gen_rand_input_imagenet, train_step)
    test_main(swin_t(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)
    test_main(vgg19(), split_into_equal_size(3), gen_rand_input_imagenet, train_step)
    test_main(vit_b_16(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)

    # # ======== not working ========
    # test_main(AutoModel.from_pretrained("bert-base-uncased"), {
    #     'encoder.layer.3',
    #     'encoder.layer.6',
    #     'encoder.layer.9',
    # }, factory_gen_rand_input_ids(30522), train_step_gpt)  # None in output?

    # test_main(LlamaModel(LlamaConfig()), {
    #     'layers.8',
    #     'layers.16',
    #     'layers.24',
    # }, factory_gen_rand_input_ids(LlamaConfig().vocab_size),
    #           train_step_gpt)  # overflow? OOM on my machine

    # test_main(
    #     T5ForConditionalGeneration.from_pretrained("t5-small"), {
    #         'encoder.block.2',
    #         'encoder',
    #         'decoder.block.2',
    #     }, factory_gen_rand_input_ids(32128),
    #     train_step_t5)  # module not outputting tensors or list/tuple of tensors but python class