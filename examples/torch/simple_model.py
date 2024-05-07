import argparse
import copy
import logging
import os
from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.utils.checkpoint import checkpoint
from torch.distributed._tensor import DeviceMesh

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.experimental.pp.compile_pipeline import annotate_split_points


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())

    return model


class Foo(torch.nn.Module):

    def __init__(self, enable_checkpoint=False):
        super().__init__()
        self.enable_checkpoint = enable_checkpoint
        self.norm = torch.nn.LayerNorm(1024)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        if self.enable_checkpoint:
            x = checkpoint(self.linear, x, preserve_rng_state=False)
        else:
            x = self.linear(x)
        return x.relu()


def inference_example(fake_init=True, cpu_init_helper=False):

    @easydist_compile(tracing_mode="fake")
    @torch.inference_mode()
    def inference_step(model, input):
        out = model(input)
        return out

    fake_mode = FakeTensorMode()

    # (NOTE) initialize cuda context first see https://github.com/pytorch/pytorch/issues/92627
    torch.ones(1).cuda()
    with torch.device('cuda'), fake_mode if fake_init else nullcontext():
        model = Foo()
        randn_input = torch.randn(1024, 1024)

        if not fake_init:
            # broadcast the parameter and input
            model = broadcast_module(model)
            torch.distributed.broadcast(randn_input, src=0)

    torch_out = inference_step.original_func(model, randn_input)

    if fake_init:
        randn_input = torch.randn(1024, 1024).cuda()
        torch.distributed.broadcast(randn_input, src=0)

    if cpu_init_helper:
        module = Foo().cuda()
        module = broadcast_module(module)

        inference_step.register_cpu_module(copy.deepcopy(module).to(device="cpu"))

        torch_out = inference_step.original_func(module, randn_input)

    md_out = inference_step(model, randn_input)

    if fake_init and not cpu_init_helper:
        assert md_out.shape == torch_out.shape, "shape mismatch"
    else:
        assert torch.allclose(torch_out, md_out), "simple model inference test failed."

    print("simple model inference example pass.")


def train_example(fake_init=True, enable_checkpoint=False, cpu_init_helper=False):

    # when using cuda_graph, because of the warm-up and cuda graph capture,
    # the result of the first step is equivalent to the original result of the third step
    @easydist_compile(tracing_mode="fake", cuda_graph=False)
    def train_step(input, model, opt):
        out = model(input).mean()
        out.backward()
        opt.step()
        opt.zero_grad(True)

        return out

    fake_mode = FakeTensorMode()

    # (NOTE) initialize cuda context first see https://github.com/pytorch/pytorch/issues/92627
    torch.ones(1).cuda()
    with torch.device('cuda'), fake_mode if fake_init else nullcontext():
        model = Foo(enable_checkpoint)
        annotate_split_points(model, {
            "norm"
        })

        randn_input = torch.randn(1024, 1024)

        if not fake_init:
            # broadcast the parameter and input
            model = broadcast_module(model)
            torch.distributed.broadcast(randn_input, src=0)

        opt = torch.optim.SGD(model.parameters(), lr=0.001, foreach=True, momentum=0.9, capturable=True)

        model_2 = copy.deepcopy(model)
        opt_2 = torch.optim.SGD(model_2.parameters(), lr=0.001, foreach=True, momentum=0.9, capturable=True)

        torch_step_1_result = train_step.original_func(randn_input, model, opt)
        torch_step_2_result = train_step.original_func(randn_input, model, opt)

    # need real input for compiled func
    if fake_init:
        randn_input = torch.randn(1024, 1024).cuda()
        torch.distributed.broadcast(randn_input, src=0)

    if cpu_init_helper:
        module = Foo(enable_checkpoint).cuda()
        module = broadcast_module(module)
        train_step.register_cpu_module(copy.deepcopy(module).to("cuda"))

        opt = torch.optim.Adam(module.parameters(), lr=0.001, foreach=True, capturable=True)

        torch_step_1_result = train_step.original_func(randn_input, module, opt)
        torch_step_2_result = train_step.original_func(randn_input, module, opt)

    md_step_1_result = train_step(randn_input, model_2, opt_2)
    md_step_2_result = train_step(randn_input, model_2, opt_2)

    if fake_init and not cpu_init_helper:
        assert torch_step_1_result.shape == md_step_1_result.shape, "shape mismatch"
        assert torch_step_2_result.shape == md_step_2_result.shape, "shape mismatch"
    else:
        assert torch.allclose(torch_step_1_result,
                              md_step_1_result), "simple model training test failed."
        assert torch.allclose(torch_step_2_result,
                              md_step_2_result), "simple model training test failed."

    print("simple model training example pass.")


def main():
    parser = argparse.ArgumentParser(description="Simple example of parallelize model.")

    parser.add_argument("--mode",
                        type=str,
                        default=None,
                        choices=["train", "inference"],
                        required=True)
    parser.add_argument("--fake-init", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--cpu-init-helper", action="store_true")

    args = parser.parse_args()

    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    
    set_device_mesh(DeviceMesh("cuda", [
        [
            [0, 2],
            [1, 3]
        ]
    ], mesh_dim_names=["spmd0", "spmd1", "pp"]))

    if args.mode == "train":
        train_example(fake_init=args.fake_init,
                      enable_checkpoint=args.checkpoint,
                      cpu_init_helper=args.cpu_init_helper)
    if args.mode == "inference":
        inference_example(fake_init=args.fake_init, cpu_init_helper=args.cpu_init_helper)


if __name__ == "__main__":
    main()
