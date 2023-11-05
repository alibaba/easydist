import argparse
import os
import copy
import logging

import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states

from torchvision.models import resnet18

from pippy.IR import Pipe
from pippy import split_into_equal_size

from easydist import easydist_setup, mdconfig
from easydist.torch.experimental.api import easydist_compile
from easydist.torch import set_device_mesh
from easydist.utils.testing import TorchMockDeviceMesh


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())
    return model


def train_submodule(train_step, name, module, *prev_mod_input):
    with torch.device('cuda'):
        opt = torch.optim.Adam(module.parameters(),
                               lr=0.001, foreach=True, capturable=True)
        module_2 = copy.deepcopy(module)
        opt_2 = torch.optim.Adam(
            module_2.parameters(), lr=0.001, foreach=True, capturable=True)

        torch_step_1_result = train_step.original_func(
            module, opt, *prev_mod_input)
        torch_step_2_result = train_step.original_func(
            module, opt, *prev_mod_input)

    md_step_1_result = train_step(module_2, opt_2, *prev_mod_input)
    md_step_2_result = train_step(module_2, opt_2, *prev_mod_input)

    def warp_fn(item): return [item] if isinstance(
        item, torch.Tensor) else item
    local_rank = int(os.environ["LOCAL_RANK"])
    for i, (torch_res, md_res) in enumerate(zip(warp_fn(torch_step_1_result), warp_fn(md_step_1_result))):
        # assert torch.allclose(torch_res, md_res), f"{name} training test step 1 failed."
        if not torch.allclose(torch_res, md_res):
            print(
                '\033[31m'
                f'rank {local_rank} {name} step 1 {i}-th output failed.'
                f'\n\tabs sum: {torch.sum(torch.abs(torch_res - md_res))}'
                '\033[0m'
            )

    for i, (torch_res, md_res) in enumerate(zip(warp_fn(torch_step_2_result), warp_fn(md_step_2_result))):
        # assert torch.allclose(torch_res, md_res), f"{name} training test step 2 failed."
        if not torch.allclose(torch_res, md_res):
            print(
                '\033[31m'
                f'rank {local_rank} {name} step 2 {i}-th output failed.'
                f'\n\tabs sum: {torch.sum(torch.abs(torch_res - md_res))}'
                '\033[0m'
            )

    return torch_step_2_result


@easydist_compile(tracing_mode="fake", cuda_graph=False)
def train_step_submod0(model, opt, *input):
    out = model(*input)
    _out = out[0].mean() + out[1].mean()
    _out.backward()
    opt.step()
    opt.zero_grad(True)
    return out


@easydist_compile(tracing_mode="fake", cuda_graph=False)
def train_step_submod1(model, opt, *input):
    out = model(*input)
    _out = out.mean()
    _out.backward()
    opt.step()
    opt.zero_grad(True)
    return out


def train_example():
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    model = resnet18().cuda()
    model.train()
    # broadcast the parameter and input
    model = broadcast_module(model)

    split_policy = split_into_equal_size(world_size)
    pipe = Pipe.from_tracing(model, split_policy=split_policy).cuda()

    input = [torch.randn(16, 3, 224, 224).cuda()]
    torch.distributed.broadcast(input[0], src=0)
    input = train_submodule(train_step_submod0, "submod_0",
                            pipe.split_gm.submod_0, *input)

    input = [t.detach() for t in input]
    train_submodule(train_step_submod1, "submod_1",
                    pipe.split_gm.submod_1, *input)

    print(f"rank {local_rank} training test finished.")


def inference_example():
    raise NotImplementedError()


def main():
    parser = argparse.ArgumentParser(
        description="Simple example of parallelize model.")
    parser.add_argument("--mode",
                        type=str,
                        default=None,
                        choices=["train", "inference"],
                        required=True)
    args = parser.parse_args()

    mdconfig.log_level = logging.ERROR
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This example only support 2 GPUs."
    mock_mesh = TorchMockDeviceMesh(1, 2, debug_only=True)
    set_device_mesh(mock_mesh)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if args.mode == "train":
        train_example()
    else:
        inference_example()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
