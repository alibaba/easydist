import torch
import os
import logging
import random

import numpy as np

from torch.distributed._tensor import DeviceMesh
from torchvision.models import resnet18, vgg19, alexnet

from easydist import easydist_setup, mdconfig
from easydist.torch import set_device_mesh, get_device_mesh
from easydist.torch.experimental.pp.model_split import split_and_compile
from easydist.torch.experimental.pp.loss import LossWrapper
from easydist.torch.experimental.pp.split_policy import split_into_equal_size


def setup():
    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    mdconfig.forced_compile = True

    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    mesh_shape = np.array(range(world_size)).reshape(1, -1)
    mesh = DeviceMesh("cuda", mesh_shape.tolist())

    if get_device_mesh() == None:
        set_device_mesh(mesh)


class MeanLossWrapper(LossWrapper):

    def forward(self, input):
        x = self.module(input)
        return {"output": x, "loss": x.mean()}


def test_compile(model_class):
    model = model_class().cuda()
    rand_input = torch.rand(16, 3, 224, 224).cuda()
    model = MeanLossWrapper(model)
    split_gm = split_and_compile(model,
                                 rand_input,
                                 split_policy=split_into_equal_size(2))
    return split_gm


if __name__ == "__main__":
    setup()
    test_compile(resnet18)
    test_compile(vgg19)
    test_compile(alexnet)
    print("passed")
