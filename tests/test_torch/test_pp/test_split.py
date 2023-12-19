import os
import copy
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
from torchvision.models import (alexnet, densenet121, efficientnet_b0, resnet18,
                                swin_t, vgg19, vit_b_16)
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.split_model import (split_by, fw_bw_split_point, SplitPatcher,
                                                        annotate_split_points, PipeSplitWrapper,
                                                        compile_stateful_stages,
                                                        split_into_equal_size)
from easydist.torch.experimental.pp.my_make_fx import my_make_fx
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
    out = model(input)
    loss = (out - torch.ones_like(out) * label).pow(2).sum()
    loss.backward()
    return loss


def test_main(model_cls, input_size, split_ann_or_policy):
    model = model_cls().cuda().train().double()
    optim = None
    if isinstance(split_ann_or_policy, dict):
        annotate_split_points(model, split_ann_or_policy)
    else:
        model = split_ann_or_policy(model)
    rand_input = torch.rand(input_size).cuda().double()
    args = (rand_input, 0.0012345, model, optim)
    kwargs = {}
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    named_states = {}

    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(
                cast(torch.nn.Module, model), {
                    **params,
                    **buffers
                }, tie_weights=True) if model else nullcontext(), _rematerialize_optimizer(
                    optim, named_states, params) if optim else nullcontext():
            ret = func(*args, **kwargs)

        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    with _enable_compile(), SplitPatcher(model, None):
        traced_graph = my_make_fx(partial(stateless_func, train_step),
                                  tracing_mode='fake',
                                  decomposition_table=EASYDIST_DECOMP_TABLE,
                                  _allow_non_fake_inputs=False)(params, buffers, named_states,
                                                                args, kwargs)
    traced_graph = preprocess_traced_graph(traced_graph)
    split = split_by(model, traced_graph, fw_bw_split_point)

    print("traced_graph:\n", traced_graph.code)

    for name, submod in split.named_children():
        print(name, ':\n', submod.code)
        save_graphviz_dot(submod, name)

    print("split:\n", split.code)
    save_graphviz_dot(split, 'split')

    args_flatten, _ = pytree.tree_flatten((params, buffers, named_states, args, kwargs))

    ph2name, out2idx, compiled_stages, gm = compile_stateful_stages(
        split, args_flatten[:len(params) + len(buffers)])

    id_rand_input = -1
    for i, arg in enumerate(args_flatten):
        if arg is rand_input:
            id_rand_input = i
            break

    seed()
    with torch.no_grad():
        gm(**{ph2name[id_rand_input]: rand_input})

    outputs = {}
    for stage in compiled_stages:
        outputs.update(stage.outputs)

    out_flatten = [None] * len(out2idx)
    for name in outputs:
        out_flatten[out2idx[name]] = outputs[name]

    seed()
    with torch.no_grad():
        out_copy = traced_graph(params, buffers, named_states, args, kwargs)
    out_flatten_copy, _ = pytree.tree_flatten(out_copy)

    for val, val_copy in zip(out_flatten, out_flatten_copy):
        if isinstance(val, torch.Tensor):
            assert torch.allclose(val, val_copy)
        else:
            assert val == val_copy

    print(f"passed {model_cls.__name__}")


if __name__ == '__main__':
    test_main(Foo, (16, 1024), {'norm': PipeSplitWrapper.SplitPoint.END})
    test_main(Foo1, (16, 1024), {
        'norm': PipeSplitWrapper.SplitPoint.END,
        'linear0_1': PipeSplitWrapper.SplitPoint.END
    })
    test_main(alexnet, (16, 3, 224, 224), {
        'features.10': PipeSplitWrapper.SplitPoint.END,
        'classifier.3': PipeSplitWrapper.SplitPoint.END
    })
    test_main(densenet121, (16, 3, 224, 224), {
        'features.denseblock1.denselayer4.norm2': PipeSplitWrapper.SplitPoint.END,
        'features.transition2.conv': PipeSplitWrapper.SplitPoint.END,
        'features.denseblock4.denselayer1.relu1': PipeSplitWrapper.SplitPoint.END,
        'classifier': PipeSplitWrapper.SplitPoint.BEGINNING
    })
    test_main(efficientnet_b0, (16, 3, 224, 224), {
        'features.2.0.block.1': PipeSplitWrapper.SplitPoint.END,
        'features.4.1.block.3': PipeSplitWrapper.SplitPoint.BEGINNING,
        'features.6.1.block.3': PipeSplitWrapper.SplitPoint.BEGINNING,
        'features.8': PipeSplitWrapper.SplitPoint.BEGINNING
    })
    test_main(
        resnet18, (16, 3, 224, 224), {
            'layer1': PipeSplitWrapper.SplitPoint.END,
            'layer2': PipeSplitWrapper.SplitPoint.END,
            'layer3': PipeSplitWrapper.SplitPoint.END,
            'layer4': PipeSplitWrapper.SplitPoint.END,
        })
    test_main(swin_t, (16, 3, 224, 224), {
        'features.2.reduction': PipeSplitWrapper.SplitPoint.END,
        'features.3.0.mlp.1': PipeSplitWrapper.SplitPoint.END,
        'features.5.1.attn.qkv': PipeSplitWrapper.SplitPoint.END,
        'features.7.0.stochastic_depth': PipeSplitWrapper.SplitPoint.END
    })
    test_main(
        vgg19, (16, 3, 224, 224), {
            'features.10': PipeSplitWrapper.SplitPoint.END,
            'features.20': PipeSplitWrapper.SplitPoint.END,
            'classifier.3': PipeSplitWrapper.SplitPoint.END
        })
    test_main(vit_b_16, (16, 3, 224, 224), {
        'encoder.layers.encoder_layer_1.self_attention': PipeSplitWrapper.SplitPoint.END,
        'encoder.layers.encoder_layer_5.mlp.3': PipeSplitWrapper.SplitPoint.END,
        'encoder.layers.encoder_layer_9.ln_2': PipeSplitWrapper.SplitPoint.END
    })

    test_main(Foo, (16, 1024), split_into_equal_size(2))
    test_main(Foo1, (16, 1024), split_into_equal_size(2))
    test_main(alexnet, (16, 3, 224, 224), split_into_equal_size(2))
    test_main(densenet121, (16, 3, 224, 224), split_into_equal_size(5))
    test_main(efficientnet_b0, (16, 3, 224, 224), split_into_equal_size(7))
    test_main(resnet18, (16, 3, 224, 224), split_into_equal_size(3))
    test_main(swin_t, (16, 3, 224, 224), split_into_equal_size(3))
    test_main(vgg19, (16, 3, 224, 224), split_into_equal_size(3))
    test_main(vit_b_16, (16, 3, 224, 224), split_into_equal_size(4))
