# torchrun --nproc_per_node 4 examples/torch/pipeline_parallelism/resnet_train.py
import os
import sys
import random

import numpy as np

import torch
import torch.distributed as dist

from torchvision import datasets, transforms
from torchvision.models import resnet18

from easydist.torch.experimental.pp.api import _compile_pp
from easydist.torch.experimental.pp.compile_pipeline import (split_into_equal_size)
from easydist.torch.init_helper import SetParaInitHelper
from easydist.torch.experimental.pp.PipelineStage import ScheduleGPipe, ScheduleDAPPLE
from easydist.torch.experimental.pp.microbatch import CustomReducer, TensorChunkSpec, Replicate



def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for built-in Python
    random.seed(seed)
    # Set(seed) for each of the random number generators in python:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


criterion = torch.nn.CrossEntropyLoss()


def train_step(input, label, model, opt):
    opt.zero_grad()
    out = model(input)
    loss = criterion(out, label)
    loss.backward()
    opt.step()
    return out, loss


def test_main():
    seed(42)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f


    dist.init_process_group(rank=rank, world_size=world_size)

    compile_device = torch.device('cpu')

    module = resnet18().train().to(compile_device)
    module.fc = torch.nn.Linear(512, 10).to(compile_device)
    _, module = split_into_equal_size(world_size)(module)

    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)
    # opt = torch.optim.SGD(module.parameters(), lr=0.001, foreach=True)

    batch_size = 64
    num_chunks = world_size * 4 # high comm overhead
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    x, y = next(iter(train_dataloader))

    args = [x.to(compile_device), y.to(compile_device), module.to(compile_device), opt]
    kwargs = {}

    args_chunk_spec = [TensorChunkSpec(0), TensorChunkSpec(0), Replicate(), Replicate()]
    kwargs_chunk_spec = {}
    output_chunk_spec = [TensorChunkSpec(0), CustomReducer(lambda x, y: x + y)]

    compiled_fn = _compile_pp(train_step, 'fake', SetParaInitHelper(), None, args, kwargs,
                       ScheduleDAPPLE, args_chunk_spec, kwargs_chunk_spec,
                       output_chunk_spec, num_chunks)


    def validation(module, valid_dataloader, epoch, params, buffers):
        module.load_state_dict({**params, **buffers})
        module.to(rank)
        module.eval()
        correct_cnt = 0
        all_cnt = 0
        for x_batch, y_batch in valid_dataloader:
            x_batch = x_batch.to(f'cuda:{rank}')
            y_batch = y_batch.to(f'cuda:{rank}')
            out = module(x_batch)
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch).sum()
            all_cnt += len(y_batch)
        print(f'epoch {epoch} valid accuracy: {correct_cnt / all_cnt}')

    epochs = 5
    for epoch in range(epochs):
        all_cnt, correct_cnt, loss_sum = 0, 0, 0
        for x_batch, y_batch in train_dataloader:
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            out, loss = compiled_fn(x_batch, y_batch, module, opt)
            all_cnt += len(out)
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch.to(f'cuda:{rank}')).sum()
            loss_sum += loss.item()

        print(f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}, loss sum {loss_sum}, avg loss: {loss_sum / all_cnt}')

        params, buffers, _, _, _ = compiled_fn.all_gather_outputs()
        validation(module, valid_dataloader, epoch, params, buffers)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(cur_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = compiled_fn.state_dict()
    opt_state_dict = compiled_fn.optimizer_state_dict()

    torch.save(state_dict, os.path.join(ckpt_dir, f'state_dict_{rank}.pt'))
    torch.save(opt_state_dict, os.path.join(ckpt_dir, f'opt_state_dict_{rank}.pt'))

if __name__ == '__main__':
    test_main()
