import os
import sys

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel import get_data_parallel_group

from metadist.utils.timer import MDTimer
from metadist import metadist_setup

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark.bench_case import GPTCase
from benchmark.torch.model.gpt_tp import GPT


def get_gpt_case(cuda=True):

    case = GPTCase()
    model = GPT(depth=case.num_layers, dim=case.hidden_dim, num_heads=case.num_heads)
    data_in = torch.ones(case.batch_size, case.seq_size, case.hidden_dim)

    if cuda:
        return model.cuda(), data_in.cuda()

    return model, data_in


def bench_tp(model, data_in):

    ddp_model = DDP(model, process_group=get_data_parallel_group())
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    def train_step():
        optimizer.zero_grad()
        out = ddp_model(data_in)
        out_grad = torch.ones_like(out)
        out.backward(out_grad)
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    timer = MDTimer(train_step, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"Time: {elaps_time}")


def main():
    metadist_setup(backend="torch", device="cuda")
    # setup distributed
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.distributed.get_world_size()
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    model, data_in = get_gpt_case(cuda=True)

    bench_tp(model, data_in)


if __name__ == '__main__':
    main()
