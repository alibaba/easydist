import torch
import torch.utils._pytree as pytree

import random

from torchvision.models import resnet18, vgg19, alexnet
from easydist.torch.experimental.pp.loss_wrapper import LossWrapper

from easydist.torch.experimental.pp.model_split import _from_tracing, run_local_split_gm
from easydist.torch.experimental.pp.split_policy import split_into_equal_size

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
    rand_input = torch.rand(16, 3, 224, 224).cuda()

    out_torch = model(rand_input)

    split_policy = split_into_equal_size(2)
    model = OutputLossWrapper(model)
    gm = _from_tracing(model, rand_input, split_policy=split_policy)
    
    run_local_split_gm(gm, rand_input)
    return 0
if __name__ == "__main__":
    test_split(resnet18)
    test_split(vgg19)
    test_split(alexnet)
    print("passed")
