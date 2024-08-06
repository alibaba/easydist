import pytest
import torch
import torch.distributed

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import set_device_mesh, NDDeviceMesh
from easydist.utils.testing import spawn
from torch.distributed._tensor import DeviceMesh


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.linear = torch.nn.Linear(8, 8)

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

    torch.ones(1).cuda()
    with torch.device('cuda'):
        model = Foo()
        randn_input = torch.randn(16, 8)

        torch.distributed.broadcast(randn_input, src=0)

        opt = torch.optim.Adam(model.parameters(), lr=0.001, foreach=True, capturable=True)

    # trace train step func
    mesh = NDDeviceMesh(DeviceMesh(
        "cuda", [0, 1], mesh_dim_names=["spmd"]
    ))
    set_device_mesh(mesh)

    train_step(randn_input, model, opt)

def main():
    # setting up easydist and torch.distributed
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    torch.cuda.set_device(torch.distributed.get_rank())
    train_example()

@pytest.mark.torch
def test_simple_model():
    spawn(main, nprocs=2)

if __name__ == "__main__":
    test_simple_model()
