import argparse
import copy
import logging
import os
from contextlib import nullcontext

import torch
from torch.distributed._tensor import DeviceMesh
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.utils.checkpoint import checkpoint
from easydist.torch.experimental.pp.IR import (compile_symbolic_splited, run_local_split_gm, symbolic_split)
from easydist.torch.experimental.pp.split_policies import split_into_equal_size
from easydist.torch.device_mesh import (device_mesh_world_size, get_device_mesh, set_device_mesh)

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile
from torchvision.models import alexnet, resnet18, vgg19


def save_dot(dot, fname):
    with open(fname, 'w') as f:
        f.write(dot.source)

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


def test_compile(model_class, input_size):
    model = model_class().cuda().train()
    rand_input = torch.rand(input_size).cuda()

    split_policy = split_into_equal_size(2)
    model_split = symbolic_split(model, split_policy=split_policy)

    compile_symbolic_splited(model_split, rand_input)

    from easydist.torch.compiler import easydist_shard
    traced_graph = model_split.submod_0.fw_gm
    state_tensor_num = len(model_split.submod_0.named_buffers) + len(model_split.submod_0.named_params)
    input_signature = 102313124124
    params = model_split.submod_0.named_params
    buffers = model_split.submod_0.named_buffers
    named_states = {
        **params,
        **buffers
    }
    args = (rand_input, )
    kwargs = {}
    ret = easydist_shard(traced_graph, state_tensor_num, input_signature, params, buffers, args, kwargs)
    print(ret)

if __name__ == "__main__":
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    torch.distributed.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    import numpy
    mesh_shape = numpy.array(range(world_size)).reshape(1, -1)
    mesh = DeviceMesh("cuda", mesh_shape.tolist())

    if get_device_mesh() == None:
        set_device_mesh(mesh)
    imagenet_input_size = (16, 3, 224, 224)
    test_compile(Foo, (16, 1024))
    test_compile(resnet18, imagenet_input_size)
    test_compile(vgg19, imagenet_input_size)
    test_compile(alexnet, imagenet_input_size)
    print("passed")
