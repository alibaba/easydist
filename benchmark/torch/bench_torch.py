# torchrun --nproc_per_node 2 --master_port 26543 ./benchmark/bench_torch.py

import logging
import os
import sys
from functools import partial

import torch
import torch.optim as optim
import torch.utils._pytree as pytree
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile
from easydist.utils.timer import EDTimer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark.torch.model import GPT, GATLayer, wresnet50
from benchmark.bench_case import GPTCase, ResNetCase, GATCase


def get_gpt_case(device="cuda"):

    case = GPTCase()
    model = GPT(depth=case.num_layers, dim=case.hidden_dim, num_heads=case.num_heads)
    data_in = torch.ones(case.batch_size, case.seq_size, case.hidden_dim)

    model.device = torch.device(device)
    return model.to(device=device), data_in.to(device=device)


def get_resnet_case(device="cuda"):
    case = ResNetCase()
    model = wresnet50()
    data_in = torch.ones(case.batch_size, 3, 224, 224)

    model.device = torch.device(device)
    return model.to(device=device), data_in.to(device=device)


def get_gat_case(device="cuda"):
    case = GATCase()
    model = GATLayer(case.in_feature, case.out_feature)
    data_in = torch.ones(case.num_node, case.in_feature)
    adj = torch.ones(case.num_node, case.num_node)

    model.device = torch.device(device)
    return model.to(device=device), [data_in.to(device=device), adj.to(device=device)]


def bench_ddp(model, data_in):

    if not isinstance(data_in, list):
        data_in = [data_in]

    world_size = torch.distributed.get_world_size()
    for i in range(len(data_in)):
        data_in[i] = torch.chunk(data_in[i], world_size)[0]

    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    def train_step():
        optimizer.zero_grad()
        out = ddp_model(*data_in)
        out_grad = torch.ones_like(out)
        out.backward(out_grad)
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    timer = EDTimer(train_step, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"Time: {elaps_time}")


def bench_fsdp(model, data_in):

    if not isinstance(data_in, list):
        data_in = [data_in]

    if not isinstance(model, GATLayer):
        world_size = torch.distributed.get_world_size()
        for i in range(len(data_in)):
            data_in[i] = torch.chunk(data_in[i], world_size)[0]

    ddp_model = FSDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    def train_step():
        optimizer.zero_grad()
        out = ddp_model(*data_in)
        out_grad = torch.ones_like(out)
        out.backward(out_grad)
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    timer = EDTimer(train_step, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"Time: {elaps_time}")


def to_meta(node_output):
    if type(node_output) is torch.Tensor:
        return node_output.to(device="meta")
    elif type(node_output) is torch.nn.parameter.Parameter:
        return node_output.to(device="meta")
    else:
        return node_output


def bench_easydist(model, data_in):

    if not isinstance(data_in, list):
        data_in = [data_in]

    if model.device == torch.device("meta"):
        data_in = pytree.tree_map(to_meta, data_in)

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    @easydist_compile(cuda_graph=False)
    def train_step(model, optimizer, data_in):
        output_ = model(*data_in)
        output_grad = torch.ones_like(output_)
        output_.backward(output_grad)
        optimizer.step()
        optimizer.zero_grad()
        return output_

    train_step_partial = partial(train_step, model, optimizer, data_in)
    train_step_partial()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    timer = EDTimer(train_step_partial, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"[{local_rank}] Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"[{local_rank}] Time: {elaps_time}")


def main():
    # setup easydist
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda")

    # setup distributed
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model, data_in = get_gpt_case(device="cuda")

    bench_easydist(model, data_in)


if __name__ == '__main__':
    main()
