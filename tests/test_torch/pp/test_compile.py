import logging
import os
import numpy

import torch
from torch.distributed._tensor import DeviceMesh
from easydist.torch.experimental.pp.IR import symbolic_split
from easydist.torch.experimental.pp.compile_splited import compile_splited
from easydist.torch.experimental.pp.split_policies import split_into_equal_size
from easydist.torch.device_mesh import (get_device_mesh, set_device_mesh)

from easydist import easydist_setup, mdconfig
from easydist.torch.compiler import easydist_shard

from torchvision.models import alexnet, resnet18, vgg19


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


signature = 0x3f3f3f3f


def test_compile(model_class, input_size):
    model = model_class().cuda().train()
    rand_input = torch.rand(input_size).cuda()
    split_policy = split_into_equal_size(2)
    model_split, _ = symbolic_split(model, split_policy=split_policy)
    compiled_splited = compile_splited(model_split, rand_input)

    out = compile_forward(compiled_splited, rand_input)
    if not isinstance(out, tuple):
        out = (out, )
    compile_backward(compiled_splited, [torch.ones_like(t) for t in out])


def compile_forward(compiled, rand_input):
    global signature
    args = (rand_input, )
    kwargs = {}
    for name, (compiled_fw, _) in compiled.compiled_submods().items():
        gm = compiled_fw.fw_gm
        params = compiled_fw.named_params
        buffers = compiled_fw.named_buffers
        states = {**params, **buffers}
        signature += 1
        easydist_shard(gm, len(states), signature, params, buffers, args, kwargs)
        args = compiled_fw(*args)
        if not isinstance(args, tuple):
            args = (args, )
    return args


def compile_backward(compiled, out_grads):
    global signature
    kwargs = {}
    for name, (_, compiled_bw) in reversed(compiled.compiled_submods().items()):
        args = compiled._get_args(out_grads, compiled_bw)
        args = list(compiled_bw.compiled_fw.saved_tensors) + args
        gm = compiled_bw.bw_gm
        params = {}
        buffers = {}
        states = {**params, **buffers}
        signature += 1
        easydist_shard(gm, len(states), signature, params, buffers, args, kwargs)
        out_all = gm(*args)
        out_grads = out_all[len(compiled_bw.compiled_fw.named_params) +
                            len(compiled_bw.compiled_fw.named_buffers):]
    return out_grads


if __name__ == "__main__":
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    torch.distributed.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
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
