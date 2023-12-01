import copy
import random

import numpy as np
import torch
from torchvision.models import alexnet, resnet18, vgg19

from easydist.torch.experimental.pp.IR import symbolic_split
from easydist.torch.experimental.pp.compile_splited import compile_splited
from easydist.torch.experimental.pp.split_policies import split_into_equal_size


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
        self.norm = torch.nn.BatchNorm1d(1024)
        self.linear = torch.nn.Linear(1024, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        norm = self.norm(x)
        linear = self.linear(norm)
        return self.relu(linear)


def test_main(model_class, input_size):
    model = model_class().cuda().train()
    model_copy = copy.deepcopy(model)

    rand_input1 = torch.rand(input_size).cuda()
    rand_input2 = torch.rand(input_size).cuda()

    out_torch, loss_torch, buffers_torch, params_torch, grads_torch = run_torch(
        model, rand_input1, rand_input2)

    out_split, loss_split, buffers_split, params_split, grads_split = run_split(
        model_copy, rand_input1, rand_input2)

    # TODO @botbw: precision loss?
    assert torch.allclose(out_split, out_torch, atol=1e-4)
    assert torch.allclose(loss_split, loss_torch, atol=1e-4)
    assert len(buffers_split) == len(buffers_torch) and all(
        torch.allclose(b1, b2, atol=1e-4) for b1, b2 in zip(buffers_split, buffers_torch))
    assert len(grads_split) == len(grads_torch) and all(
        torch.allclose(g1, g2, atol=1e-4) for g1, g2 in zip(grads_split, grads_torch))
    assert len(params_split) == len(params_torch) and all(
        torch.allclose(p1, p2, atol=1e-4) for p1, p2 in zip(params_split, params_torch))


def run_split(model_copy, rand_input1, rand_input2):
    opt_splited = torch.optim.SGD(model_copy.parameters(), lr=0.01)
    split_policy = split_into_equal_size(2)
    model_splited, _ = symbolic_split(model_copy, split_policy=split_policy)
    compiled_splited = compile_splited(model_splited, rand_input1)

    seed()
    # step1
    out_split = compiled_splited.forward(rand_input1)
    loss_split = loss_fn(out_split)
    out_grads = torch.autograd.grad(loss_split,
                                    [t for t in compiled_splited.raw_returns() if t.requires_grad])
    compiled_splited.backward(out_grads)
    opt_splited.step()
    # step2
    opt_splited.zero_grad()
    out_split = compiled_splited.forward(rand_input2)
    loss_split = loss_fn(out_split)
    out_grads = torch.autograd.grad(loss_split,
                                    [t for t in compiled_splited.raw_returns() if t.requires_grad])
    compiled_splited.backward(out_grads)
    opt_splited.step()

    buffer_split = [t for t in compiled_splited.named_buffers().values()]
    param_split = [t for t in compiled_splited.named_parameters().values()]
    grad_split = [t.grad for t in compiled_splited.named_parameters().values()]
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

    buffers_torch = [t for t in model.buffers()]
    params_torch = [t for t in model.parameters()]
    grads_torch = [t.grad for t in model.parameters()]
    return out_torch, loss_torch, buffers_torch, params_torch, grads_torch


if __name__ == "__main__":
    imgnet = (16, 3, 224, 224)
    test_main(Foo, (16, 1024))
    test_main(resnet18, imgnet)
    test_main(vgg19, imgnet)
    test_main(alexnet, imgnet)
    print("passed")
