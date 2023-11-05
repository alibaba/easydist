import torch
import os
import logging

import numpy as np

from torchvision.models import resnet18
from torch.distributed._tensor import DeviceMesh

from easydist import easydist_setup, mdconfig
from easydist.torch.device_mesh import get_device_mesh, set_device_mesh
from easydist.torch.experimental.api import easydist_compile
from easydist.torch.pp.split import from_tracing, LossWrapper, split_into_equal_size


@easydist_compile
def train_step(model, opt, *input):
    out = model(*input)
    out.backward()
    opt.step()
    opt.zero_grad(True)
    return out


def setup():
    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    mesh_shape = np.array(range(world_size)).reshape(1, -1)
    mesh = DeviceMesh("cuda", mesh_shape.tolist())

    if get_device_mesh() == None:
        set_device_mesh(mesh)


def main():
    setup()

    class MeanLossWrapper(LossWrapper):
        def forward(self, input):
            x = self.module(input)
            return {"output": x, "loss": x.mean()}

    model = MeanLossWrapper(resnet18().cuda())
    rand_input = [torch.rand(16, 3, 224, 224).cuda()]
    split_policy = split_into_equal_size(2)
    staged_module = from_tracing(model, split_policy=split_policy)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, foreach=True, capturable=True)
    train_step(staged_module, optimizer, *rand_input)

    print("Success!")


if __name__ == "__main__":
    main()
