import copy
import random

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
    res = func(*inputs)
    # res.backward()

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return make_boxed_func(fx_module.forward)


# aot_print_fn = aot_module(model, fw_compiler=compiler_fn,
#     bw_compiler=compiler_fn)

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

    split_policy = split_into_equal_size()
    model_split = split(model_copy, split_policy=split_policy)
    aot_print_fn = aot_module(model_split.submod_0, fw_compiler=compiler_fn,
    bw_compiler=compiler_fn)
    # run_func(aot_print_fn, rand_input)
    compile_splited(model_split, rand_input, tracing_mode="real")

    reproduce(42)
    out_dict = run_local_split_gm(model_split, rand_input)[0]
    out_split = out_dict['output']
    loss_split = out_dict['loss']

    buffer_split = [t for t in model_split.buffers()]
    param_split = [t for t in model_split.parameters()]
    grad_split = [t.grad for t in model_split.parameters()]

    assert torch.allclose(out_split, out_torch, atol=1e-5)
    assert torch.allclose(loss_split, loss_torch, atol=1e-5)
    assert len(buffer_split) == len(buffer_torch) and all(
        torch.allclose(b1, b2, atol=1e-5) for b1, b2 in zip(buffer_split, buffer_torch))
    assert len(grad_split) == len(grad_torch) and all(
        torch.allclose(g1, g2, atol=1e-5) for g1, g2 in zip(grad_split, grad_torch))
    assert len(param_split) == len(param_torch) and all(
        torch.allclose(p1, p2, atol=1e-5) for p1, p2 in zip(param_split, param_torch))


if __name__ == "__main__":
    imagenet_input_size = (16, 3, 224, 224)
    test_split(Foo, (16, 1024))
    # test_split(resnet18, imagenet_input_size)
    # test_split(vgg19, imagenet_input_size)
    # test_split(alexnet, imagenet_input_size)
    print("passed")
