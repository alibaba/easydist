import torch
import random
import copy

from torchvision.models import resnet18, vgg19, alexnet

from easydist.torch.experimental.pp.model_split import _from_tracing
from easydist.torch.experimental.pp.split_policy import split_into_equal_size
from easydist.torch.experimental.pp.loss import LossWrapper
from easydist.torch.experimental.pp.DetachExecutor import run_local_split_gm


class MeanLossWrapper(LossWrapper):

    def forward(self, input):
        x = self.module(input)
        return {"output": x, "loss": x.mean()}


def test_split(model_class, split_size=2):
    model_torch = model_class().cuda()
    rand_input = torch.rand(16, 3, 224, 224).cuda()

    out_torch = model_torch(rand_input)
    loss_torch = out_torch.mean()

    model = copy.deepcopy(model_torch)
    model = MeanLossWrapper(model)

    split_policy = split_into_equal_size(split_size)
    split_gm = _from_tracing(model, split_policy=split_policy)

    out_dict, _ = run_local_split_gm(split_gm, rand_input)
    out_split = out_dict["output"]
    loss_split = out_dict["loss"]

    assert torch.allclose(out_split, out_torch)
    assert torch.allclose(loss_split, loss_torch)


if __name__ == "__main__":
    test_split(resnet18)
    # test_split(vgg19)
    # test_split(alexnet)
    print("passed")
