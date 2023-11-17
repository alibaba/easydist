import copy
import random

import numpy as np
import torch
from torchvision.models import alexnet, resnet18, vgg19

from easydist.torch.experimental.pp.loss_wrapper import LossWrapper
from easydist.torch.experimental.pp.model_split import (_from_tracing, run_local_split_gm)
from easydist.torch.experimental.pp.split_policy import split_into_equal_size


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


class OutputLossWrapper(LossWrapper):

    def __init__(self, module):
        super().__init__(module, None)

    def forward(self, input):
        output = self.module(input)
        loss = output.mean()
        # Here we use a dict with the "loss" keyword so that PiPPy can automatically find the loss field when
        # generating the backward pass
        return {"output": output, "loss": loss}


def test_split(model_class):
    model = model_class().cuda().train()
    model_copy = copy.deepcopy(model)

    rand_input = torch.rand(16, 3, 224, 224).cuda()

    reproduce(42)
    out_torch = model(rand_input)
    out_torch.mean().backward()
    grad_torch = [t.grad for t in model.parameters()]

    split_policy = split_into_equal_size(2)
    model_split = OutputLossWrapper(model_copy)
    gm = _from_tracing(model_split, rand_input, split_policy=split_policy)

    reproduce(42)
    out_split = run_local_split_gm(gm, rand_input)[0]['output']
    grad_split = [t.grad for t in model_copy.parameters()]
    assert torch.allclose(out_split, out_torch, atol=1e-5)
    assert len(grad_split) == len(grad_torch) and all(
        torch.allclose(g1, g2, atol=1e-5) for g1, g2 in zip(grad_split, grad_torch))


if __name__ == "__main__":
    test_split(resnet18)
    test_split(vgg19)
    test_split(alexnet)
    print("passed")
