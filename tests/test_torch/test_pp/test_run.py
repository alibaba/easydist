from email import generator
from multiprocessing import process
import os
import random
from contextlib import nullcontext
from functools import partial, reduce
import threading
from typing import cast

import numpy as np
from sympy import N, comp

import torch
import torch.utils._pytree as pytree
from torch.nn.functional import cross_entropy
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import FakeTensor
import torch.distributed as dist
from torch.distributed import rpc

from torchvision import datasets, transforms
from torchvision.models import resnet18, vgg19

from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, annotate_split_points,
                                                             PipeSplitWrapper,
                                                             compile_pipeline,
                                                             split_into_equal_size,
                                                             set_backward_flag, process_inputs, process_outputs)
from easydist.utils import rgetattr, rsetattr
from easydist.torch.experimental.pp.ed_make_fx import ed_make_fx
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.torch.experimental.pp.PipelineStage import PipelineStage
from easydist.torch.experimental.pp.microbatch import merge_chunks, split_args_kwargs_into_chunks

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

sharding_sol = None
sol_rdy_cond = threading.Condition()

def fetch_strategy():
    with sol_rdy_cond:
        sol_rdy_cond.wait()
    return sharding_sol


def test_main(module, split_ann_or_policy, rand_input_gen_method, train_step_func, num_chunks):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed(42)

    # Figure out device to use
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    rpc.init_rpc(f"ed_worker{rank}", rank=rank, world_size=world_size)

    module = module.train().to(device)
    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)
    # opt = torch.optim.SGD(module.parameters(), lr=0.01, foreach=True)
    if isinstance(split_ann_or_policy, set):
        annotate_split_points(module, split_ann_or_policy)
        nstages = len(split_ann_or_policy) + 1
    else:
        nstages, module = split_ann_or_policy(module)

    batch_size = 64
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, generator=g)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, generator=g)

    x, y = next(iter(train_dataloader))
    
    args = [x.to(device), y.to(device), module, opt]
    kwargs = {}

    ##################################################################################################
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    named_states = {}
    if opt is not None:
        # assign grad and warm up optimizer
        mode = nullcontext()
        for name in dict(module.named_parameters()):
            with torch.no_grad():
                rsetattr(module, name + ".grad", torch.zeros_like(rgetattr(module, name).data))
                if isinstance(rgetattr(module, name).data, FakeTensor):
                    mode = rgetattr(module, name).data.fake_mode
        with mode:
            opt.step()
            opt.zero_grad(True)

        for n, p in params.items():
            if p in opt.state:
                named_states[n] = opt.state[p]  # type: ignore[index]
                # if step in state, reduce one for warmup step.
                if 'step' in named_states[n]:
                    named_states[n]['step'] -= 1

    flat_named_states, named_states_spec = pytree.tree_flatten(named_states)

    # fix for sgd withtout momentum
    if all(state is None for state in flat_named_states):
        named_states = {}
        flat_named_states, named_states_spec = pytree.tree_flatten(named_states)

    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(
                cast(torch.nn.Module, module), {
                    **params,
                    **buffers
                }, tie_weights=True) if module else nullcontext(), _rematerialize_optimizer(
                    opt, named_states, params) if opt else nullcontext():
            ret = func(*args, **kwargs)

        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    args_split, kwargs_split = split_args_kwargs_into_chunks(
        args,
        kwargs,
        num_chunks,
        None, #self.pipe.kwargs_chunk_spec,
    )

    with _enable_compile(), SplitPatcher(module, opt):
        set_backward_flag(False)
        traced_stateless_func = ed_make_fx(partial(stateless_func, train_step_func),
                                tracing_mode='fake',
                                decomposition_table=EASYDIST_DECOMP_TABLE,
                                _allow_non_fake_inputs=False)(params, buffers, named_states,
                                                                args_split[0], kwargs_split[0])

    traced_stateless_func.graph.eliminate_dead_code()
    traced_stateless_func = preprocess_traced_graph(traced_stateless_func)
    traced_stateless_func.recompile()
    ##################################################################################################
    save_graphviz_dot(traced_stateless_func, 'traced_graph')

    traced_stateless_func_node_metas = {node.name: node.meta for node in traced_stateless_func.graph.nodes}
    stateless_func_args = [params, buffers, named_states, args, kwargs]


    compiled_meta, compiled_stages, local_gm = compile_pipeline(
        traced_stateless_func, nstages, stateless_func_args, strict=True)

    pipe = PipelineStage(
        local_gm=local_gm,
        compiled_meta=compiled_meta,
        stage_idx=rank,
        compiled_stage=compiled_stages[rank],
        node_metas=traced_stateless_func_node_metas,
        num_chunks=num_chunks,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        outputs_chunk_spec=None,
        device=device
    )


    epochs = 5
    num_batches_tracked = 0
    num_step = 0

    for epoch in range(epochs):
        stage_kwarg = [None]
        if rank == 0:
            print(f'epoch {epoch}:')
            for i, (x_batch, y_batch) in enumerate(tqdm(train_dataloader, dynamic_ncols=True)):
                if x_batch.size(0) != batch_size: # need to solve this?
                    continue
                # print(f'rank {rank} epoch {epoch} batch {i} batch_size {x_batch.size(0)}')
                args = (x_batch, y_batch, module, opt)
                kwargs = {}
                stage_kwargs = process_inputs(compiled_meta, *args, **kwargs, move_to_device=True)
                dist.scatter_object_list(stage_kwarg, stage_kwargs, src=0)
                pipe(**stage_kwarg[0])
                if pipe.step_node is not None:
                    pipe.step()
            stage_kwargs = [-1] * compiled_meta.nstages
            dist.scatter_object_list(stage_kwarg, stage_kwargs, src=0)
        else:
            stage_kwargs = [None, None, None]
            dist.scatter_object_list(stage_kwarg, stage_kwargs, src=0)
            all_cnt = 0
            correct_cnt = 0
            while stage_kwarg[0] != -1:
                pipe(**stage_kwarg[0])
                if rank == world_size - 1:
                    out = pipe.outputs_batch[pipe.compiled_meta.returns_names_unflatten[0]]
                    all_cnt += len(out)
                    preds = out.argmax(-1)
                    correct_cnt += (preds == stage_kwarg[0][pipe.compiled_meta.args_names_unflatten[1]]).sum()
                if pipe.step_node is not None:
                    pipe.step()
                dist.scatter_object_list(stage_kwarg, stage_kwargs, src=0)

        if rank == world_size - 1:
            print(f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}')

        num_batches_tracked += 1
        num_step += 1

        outputs = pipe.all_gather_outputs(0)
        if rank == 0:
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
            params, buffers, optimstates, grads, returns = reduce(reduce_outputs, outputs)
            out, loss = returns
            print(f'epoch {epoch} loss: {loss.sum().item()}')

            module.load_state_dict({**params, **buffers})
            module.eval()
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

    print("finished")

def gen_rand_input_imagenet():
    return None


if __name__ == '__main__':
    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(rank=rank, world_size=world_size)
    test_main(resnet18(), split_into_equal_size(world_size), gen_rand_input_imagenet, train_step, num_chunks=world_size)