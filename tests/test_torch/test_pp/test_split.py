import copy
import gc
import os
import random

# make easydist happy without torchrun
os.environ['MASTER_PORT'] = '-1'

import numpy as np
import torch
from torchvision.models import (alexnet, densenet121, densenet201, efficientnet_b0, resnet18,
                                swin_t, vgg19, vit_b_16)

from easydist.torch.experimental.pp.compile_splited import compile_splited
from easydist.torch.experimental.pp.split_model import symbolic_split, split_into_equal_size

split_sz = 3


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


def loss_fn(x):
    return x.mean()


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm([1024])
        self.linear1_0 = torch.nn.Linear(1024, 256)
        self.linear1_1 = torch.nn.Linear(256, 128)
        self.linear1_2 = torch.nn.Linear(128, 10)

        self.norm0 = torch.nn.BatchNorm1d(1024)
        self.linear0_0 = torch.nn.Linear(1024, 512)
        self.linear0_1 = torch.nn.Linear(512, 256)
        self.linear0_2 = torch.nn.Linear(256, 128)
        self.linear0_3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x0 = self.norm0(x)
        x0 = self.linear0_0(x0)
        x0 = self.linear0_1(x0)
        x0 = self.linear0_2(x0)
        x0 = self.linear0_3(x0)

        x1 = self.norm1(x)
        x1 = self.linear1_0(x1)
        x1 = self.linear1_1(x1)
        x1 = self.linear1_2(x1)

        return (x0 + x1).relu()


# TODO @botbw: need to fix this
def replace_dot(x: dict):
    return {k.replace('moved_', '').replace('.', '_'): v for k, v in x.items()}


def test_main(model_gen, input_size):
    model = model_gen().cuda().train().double()
    model_copy = copy.deepcopy(model)

    rand_input1 = torch.rand(input_size, dtype=torch.double).cuda()
    rand_input2 = torch.rand(input_size, dtype=torch.double).cuda()

    out_torch, loss_torch, buffers_torch, params_torch, grads_torch = run_torch(
        model, rand_input1, rand_input2)

    out_split, loss_split, buffers_split, params_split, grads_split = run_split(
        model_copy, rand_input1, rand_input2)

    assert torch.allclose(out_split, out_torch)
    assert torch.allclose(loss_split, loss_torch)
    assert len(buffers_split) == len(buffers_torch) and all(
        torch.allclose(buffers_split[name], buffers_torch[name]) and name in buffers_split
        for name in buffers_torch)
    assert len(grads_split) == len(grads_torch) and all(
        torch.allclose(grads_split[name], grads_torch[name]) and name in grads_split
        for name in grads_torch)
    assert len(params_split) == len(params_torch) and all(
        torch.allclose(params_split[name], params_torch[name]) and name in params_split
        for name in params_torch)
    gc.collect()


def run_split(model_copy, rand_input1, rand_input2):
    opt_splited = torch.optim.SGD(model_copy.parameters(), lr=0.01)
    split_policy = split_into_equal_size(split_sz)
    model_splited, _ = symbolic_split(model_copy, split_policy=split_policy)
    compiled_splited = compile_splited(model_splited, rand_input1)

    seed()
    # step1
    out_split = compiled_splited.forward(rand_input1)
    loss_split = loss_fn(out_split)
    out_grads = torch.autograd.grad(loss_split,
                                    [t for t in compiled_splited.raw_returns() if t.requires_grad])
    compiled_splited.backward(*out_grads)
    opt_splited.step()
    # step2
    opt_splited.zero_grad()
    out_split = compiled_splited.forward(rand_input2)
    loss_split = loss_fn(out_split)
    out_grads = torch.autograd.grad(loss_split,
                                    [t for t in compiled_splited.raw_returns() if t.requires_grad])
    compiled_splited.backward(*out_grads)
    opt_splited.step()

    buffer_split = replace_dot(compiled_splited.named_buffers())
    param_split = replace_dot(compiled_splited.named_parameters())
    grad_split = replace_dot({k: v.grad for k, v in compiled_splited.named_parameters().items()})
    return out_split, loss_split, buffer_split, param_split, grad_split


def run_torch(model, rand_input1, rand_input2):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    seed()
    # step1
    out_torch = model(rand_input1)
    loss_torch = loss_fn(out_torch)
    loss_torch.backward()
    opt.step()
    # step2
    opt.zero_grad()
    out_torch = model(rand_input2)
    loss_torch = loss_fn(out_torch)
    loss_torch.backward()
    opt.step()

    buffers_torch = replace_dot(dict(model.named_buffers()))
    params_torch = replace_dot(dict(model.named_parameters()))
    grads_torch = replace_dot({k: v.grad for k, v in model.named_parameters()})
    return out_torch, loss_torch, buffers_torch, params_torch, grads_torch


if __name__ == "__main__":
    imgnet = (16, 3, 224, 224)
    print(f'Split size: {split_sz}')
    test_main(Foo, (16, 1024))
    test_main(resnet18, imgnet)
    test_main(alexnet, imgnet)
    test_main(densenet121, imgnet)
    test_main(densenet201, imgnet)
    test_main(efficientnet_b0, imgnet)
    test_main(vgg19, imgnet)
    test_main(vit_b_16, imgnet)
    test_main(swin_t, imgnet)
    print("passed")
