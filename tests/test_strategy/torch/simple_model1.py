import os
import difflib

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from easydist import easydist_setup
from easydist.torch.experimental.api import easydist_compile
from easydist.torch import set_device_mesh
from easydist.utils.testing import TorchMockDeviceMesh


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(5)
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


@easydist_compile(tracing_mode="fake", cuda_graph=False, compile_only=True)
def train_step(input, model, opt):
    out = model(input).mean()
    out.backward()
    opt.step()
    opt.zero_grad(True)

    return out


def train_example():
    fake_mode = FakeTensorMode()

    torch.ones(1).cuda()
    with torch.device('cuda'), fake_mode:
        model = Foo()
        randn_input = torch.randn(16, 5)

        torch.distributed.broadcast(randn_input, src=0)

        opt = torch.optim.Adam(model.parameters(), lr=0.001, foreach=True, capturable=True)

    # trace train step func
    mock_mesh = TorchMockDeviceMesh(1, 2, debug_only=True)
    set_device_mesh(mock_mesh)

    train_step(randn_input, model, opt)

def main():
    # setting up easydist and torch.distributed
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    train_example()

if __name__ == "__main__":
    main()

