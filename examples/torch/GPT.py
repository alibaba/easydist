# EASYDIST_LOGLEVEL=INFO torchrun --nproc_per_node 8 examples/torch/GPT.py --mode train
import argparse
import copy
import os
from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.utils.checkpoint import checkpoint
from torch.distributed._tensor import DeviceMesh

from benchmark.bench_case import GPTCase
from benchmark.torch.model.gpt import GPT
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

GPT_CASE = GPTCase(
    num_layers=1,
    hidden_dim=1024,
    num_heads=32,
    seq_size=128
)

def train_example():

    # when using cuda_graph, because of the warm-up and cuda graph capture,
    # the result of the first step is equivalent to the original result of the third step
    @easydist_compile(tracing_mode="fake", cuda_graph=False)
    def train_step(input, model, opt):
        out = model(input).mean()
        out.backward()
        opt.step()
        opt.zero_grad(True)
        return out

    # (NOTE) initialize cuda context first see https://github.com/pytorch/pytorch/issues/92627
    torch.ones(1).cuda()
    with torch.device('cuda'):
        model = GPT(
            depth=GPT_CASE.num_layers,
            dim=GPT_CASE.hidden_dim,
            num_heads=GPT_CASE.num_heads,
        )

        randn_input = torch.randn(GPT_CASE.batch_size, GPT_CASE.seq_size, GPT_CASE.hidden_dim)

        # broadcast the parameter and input
        model = broadcast_module(model)
        torch.distributed.broadcast(randn_input, src=0)

        opt = torch.optim.SGD(model.parameters(), lr=0.001, foreach=True)

        model_2 = copy.deepcopy(model)
        opt_2 = torch.optim.SGD(model_2.parameters(), lr=0.001, foreach=True)

        torch_step_1_result = train_step.original_func(randn_input, model, opt)
        torch_step_2_result = train_step.original_func(randn_input, model, opt)

    md_step_1_result = train_step(randn_input, model_2, opt_2)
    md_step_2_result = train_step(randn_input, model_2, opt_2)

    assert torch.allclose(torch_step_1_result,
                            md_step_1_result), f"GPT model training test failed. {torch_step_1_result} {md_step_1_result}"
    assert torch.allclose(torch_step_2_result,
                            md_step_2_result), f"GPT model training test failed."

    print("GPT model training example pass.")


def main():
    # setting up easydist and torch.distributed
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    mesh = torch.arange(world_size).reshape(2, 2, 2)
    set_device_mesh(DeviceMesh("cuda", mesh, mesh_dim_names=["spmd0", "spmd1", "spmd2"]))

    train_example()


if __name__ == "__main__":
    main()