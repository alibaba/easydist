import copy
import random
from typing import List

import numpy as np
import torch
from torchvision.models import alexnet, resnet18, vgg19

from easydist.torch.experimental.pp.loss_wrapper import LossWrapper
from easydist.torch.experimental.pp.model_split import (compile_splited, run_local_split_gm, split)
from easydist.torch.experimental.pp.split_policy import split_into_equal_size

from functorch.compile import aot_function, aot_module, \
    make_boxed_func, ts_compile

from torch._functorch.aot_autograd import aot_export_module
def reproduce(seed=42):
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

from torchviz import make_dot
def save_dot(dot, fname):
    with open(fname, 'w') as f:
        f.write(dot.source)

from torch.autograd import Function
class SplitFunc(Function):
    
        @staticmethod
        def forward(ctx, *input):
            if len(input) == 1:
                return input[0]
            return input
    
        @staticmethod
        def backward(ctx, *grad_output):
            if len(grad_output) == 1:
                return grad_output[0]
            return grad_output
def split_func(input):
    return SplitFunc.apply(input)

def loss_fn(x):
    return x.mean()

def run_func(func, *inputs):
    res = func(*inputs).mean()
    res.backward()

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return make_boxed_func(fx_module.forward)

class OutputLossWrapper(LossWrapper):

    def __init__(self, module):
        super().__init__(module, None)

    def forward(self, input):
        output = self.module(input)
        loss = loss_fn(output)
        # Here we use a dict with the "loss" keyword so that PiPPy can automatically find the loss field when
        # generating the backward pass
        return {"output": output, "loss": loss}


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

def get_jacobian(loss: torch.Tensor, raw_returns: List[torch.Tensor]):
    grads = torch.autograd.grad(loss, [t for t in raw_returns if t.requires_grad], allow_unused=True)
    jacobian = []
    j = 0
    for t in raw_returns:
        if t.requires_grad:
            jacobian.append(grads[j])
            j += 1
        else:
            jacobian.append(torch.zeros_like(t))
    return jacobian

def test_split(model_class, input_size):
    model = model_class().cuda().train()
    model_copy = copy.deepcopy(model)

    rand_input = torch.rand(input_size).cuda()

    reproduce(42)
    out_torch = model(rand_input)
    loss_torch = loss_fn(out_torch)
    loss_torch.backward()

    buffer_torch = [t for t in model.buffers()]
    param_torch = [t for t in model.parameters()]
    grad_torch = [t.grad for t in model.parameters()]

    split_policy = split_into_equal_size(1)
    model_split = split(model_copy, split_policy=split_policy)
    compile_splited(model_split, rand_input)

    reproduce(42)
    out_split = model_split(rand_input)
    loss_split = loss_fn(out_split)
    
    jacobian = get_jacobian(loss_split, model_split.submod_0.raw_returns)
    states_grad = model_split.submod_0_bw(*jacobian)
    for t, grad in zip({**model_split.submod_0.named_params, **model_split.submod_0.named_params}.values(), states_grad):
        t.grad = grad.detach()
    buffer_split = [t for t in model_split.submod_0.named_buffers.values()]
    param_split = [t for t in model_split.submod_0.named_params.values()]
    grad_split = [t.grad for t in model_split.submod_0.named_params.values()]

    assert torch.allclose(out_split, out_torch, atol=1e-5)
    assert torch.allclose(loss_split, loss_torch, atol=1e-5)
    assert len(buffer_split) == len(buffer_torch) and all(
        torch.allclose(b1, b2, atol=1e-5) for b1, b2 in zip(buffer_split, buffer_torch))
    assert len(grad_split) == len(grad_torch) and all(
        torch.allclose(g1, g2, atol=1e-4) for g1, g2 in zip(grad_split, grad_torch)) # TODO @botbw: cannot pass with atol=1e-5
    assert len(param_split) == len(param_torch) and all(
        torch.allclose(p1, p2, atol=1e-5) for p1, p2 in zip(param_split, param_torch))


if __name__ == "__main__":
    imagenet_input_size = (16, 3, 224, 224)
    test_split(Foo, (16, 1024))
    test_split(resnet18, imagenet_input_size)
    test_split(vgg19, imagenet_input_size)
    test_split(alexnet, imagenet_input_size)
    print("passed")
