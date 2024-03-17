import os
import random
from contextlib import nullcontext
from functools import partial, reduce
from typing import cast

import numpy as np

import torch
import torch.utils._pytree as pytree
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import FakeTensor
import torch.distributed as dist

from torchvision import datasets, transforms
from torchvision.models import resnet18

from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.api import _compile_pp
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, compile_pipeline,
                                                             split_into_equal_size,
                                                             set_backward_flag)
from easydist.torch.init_helper import SetParaInitHelper
from easydist.utils import rgetattr, rsetattr
from easydist.torch.experimental.pp.ed_make_fx import ed_make_fx
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.torch.experimental.pp.PipelineStage import PipelineStageBase, Schedule1F1B, ScheduleGPipe
from easydist.torch.experimental.pp.microbatch import CustomReducer, split_args_kwargs_into_chunks, TensorChunkSpec, Replicate

from tqdm import tqdm


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

    num_chunks = world_size * 4 # 1
    dist.init_process_group(rank=rank, world_size=world_size)

    compile_device = torch.device('cpu')

    module = resnet18().train().to(compile_device)
    module.fc = torch.nn.Linear(512, 10).to(compile_device)
    _, module = split_into_equal_size(world_size)(module)

    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)
    # opt = torch.optim.SGD(module.parameters(), lr=0.001, foreach=True)

    batch_size = 64
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

    compiled_fn = _compile_pp(train_step, None, SetParaInitHelper, None, args, kwargs,
                       ScheduleGPipe, args_chunk_spec, kwargs_chunk_spec,
                       output_chunk_spec, num_chunks)

    epochs = 5
    for epoch in range(epochs):
        all_cnt = 0
        correct_cnt = 0
        loss_sum = 0
        for x_batch, y_batch in (tqdm(train_dataloader, dynamic_ncols=True)
                                 if rank == 0 else train_dataloader):
            if x_batch.size(0) != batch_size:  # need to solve this?
                continue
            args = (x_batch, y_batch, module, opt)
            kwargs = {}
            ret = compiled_fn(*args, **kwargs)
            if rank == world_size - 1:
                out = ret[0]
                loss = ret[-1]
                all_cnt += len(out)
                preds = out.argmax(-1)
                correct_cnt += \
                    (preds == y_batch.to(f'cuda:{rank}')).sum()
                loss_sum += loss.sum().item()

        if rank == world_size - 1:
            print(
                f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}, loss sum {loss_sum}, avg loss: {loss_sum / all_cnt}'
            )

        outputs = compiled_fn.all_gather_outputs(0)
        if rank == 0:
            device = torch.device('cuda:0')
            def reduce_outputs(a, b):
                ret = []
                for aa, bb in zip(a, b):
                    if isinstance(aa, dict):
                        aa.update(bb)
                        ret.append(aa)
                    else:
                        tup = []
                        for aaa, bbb in zip(aa, bb):
                            if aaa is None:
                                tup.append(bbb)
                            elif bbb is None:
                                tup.append(aaa)
                            else:
                                raise ValueError('both are not None')
                        ret.append(tup)
                return ret

            params, buffers, _, _, _ = reduce(reduce_outputs, outputs)

            module.load_state_dict({**params, **buffers})
            module.eval()
            module.to(device)
            correct_cnt = 0
            all_cnt = 0
            for x_batch, y_batch in valid_dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out = module(x_batch)
                preds = out.argmax(-1)
                correct_cnt += (preds == y_batch).sum()
                all_cnt += len(y_batch)
            print(f'epoch {epoch} valid accuracy: {correct_cnt / all_cnt}')

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
