import os
import random
import inspect
from contextlib import nullcontext
from functools import partial
from typing import cast

from requests import get
from tqdm import tqdm
# make easydist happy without torchrun
os.environ['MASTER_PORT'] = '-1'

import numpy as np

from torchvision import datasets, transforms

import torch
import torch.utils._pytree as pytree
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import FakeTensor
from torchvision.models import (alexnet, densenet121, efficientnet_b0, resnet18, swin_t, vgg19,
                                vit_b_16)
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, annotate_split_points,
                                                             PipeSplitWrapper,
                                                             compile_pipeline, process_outputs,
                                                             split_into_equal_size,
                                                             before_split_register,
                                                             after_split_register,
                                                             tuple_after_split,
                                                             tuple_before_split,
                                                             set_backward_flag,
                                                             get_backward_flag,
                                                             process_inputs)
from easydist.utils import rgetattr, rsetattr
from easydist.torch.experimental.pp.ed_make_fx import ed_make_fx
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer


def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
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
    return out.detach(), loss.detach()

def test_main(split_ann_or_policy):
    module = resnet18().cuda()
    batch_size = 64
    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)

    if isinstance(split_ann_or_policy, set):
        annotate_split_points(module, split_ann_or_policy)
        nstages = len(split_ann_or_policy) + 1
    else:
        nstages, module = split_ann_or_policy(module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

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

    with _enable_compile(), SplitPatcher(module, opt):
        set_backward_flag(False)
        traced_stateless_func = ed_make_fx(partial(stateless_func, train_step),
                                  tracing_mode='fake',
                                  decomposition_table=EASYDIST_DECOMP_TABLE,
                                  _allow_non_fake_inputs=False)(params, buffers, named_states,
                                                                args, kwargs)

    traced_stateless_func.graph.eliminate_dead_code()
    traced_stateless_func = preprocess_traced_graph(traced_stateless_func)
    traced_stateless_func.recompile()
    ##################################################################################################

    # print("traced_graph:\n", traced_graph.code)
    save_graphviz_dot(traced_stateless_func, 'traced_graph')

    stateless_func_args = [params, buffers, named_states, args, kwargs]

    def arg_copy_func(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        else:
            return x

    stateless_func_args_copy = pytree.tree_map(arg_copy_func, stateless_func_args)

    compiled_meta, compiled_stages, local_gm = compile_pipeline(traced_stateless_func, nstages, stateless_func_args, strict=False)

    epochs = 5

    seed()
    with torch.no_grad():
        for epoch in range(epochs):
            print(f'epoch {epoch}')
            print('training')
            correct_cnt = 0
            all_cnt = 0
            for x_batch, y_batch in tqdm(train_dataloader, dynamic_ncols=True):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                if x_batch.size(0) != batch_size: # need to solve this?
                    continue
                args = (x_batch, y_batch, module, opt)
                kwargs = {}
                kwargs_stage = process_inputs(compiled_meta, *args, **kwargs)
                input_dict = {}
                for di in kwargs_stage:
                    input_dict.update(di)
                local_gm(**input_dict)
                outputs_dict = {}
                for stage in compiled_stages:
                    outputs_dict.update(stage.outputs)
                _, _, _, _, ret = process_outputs(compiled_meta, outputs_dict)
                out, loss = ret
                preds = out.argmax(-1)
                correct_cnt = (preds == y_batch).sum()
                all_cnt = len(y_batch)
                correct_cnt += correct_cnt.item()
                all_cnt += all_cnt

            print(f'{epoch} train accuracy: {correct_cnt / all_cnt}, loss: {loss.item()}')

            state_dict = {}
            for stage in compiled_stages:
                state_dict.update(stage.state_dict())
            module.load_state_dict(state_dict)
            module.eval()

            print('eval')
            for x_batch, y_batch in tqdm(valid_dataloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out = module(x_batch)
                preds = out.argmax(-1)
                correct_cnt = (preds == y_batch).sum()
                all_cnt = len(y_batch)
                correct_cnt += correct_cnt.item()
                all_cnt += all_cnt
            print(f'{epoch} eval accuracy: {correct_cnt / all_cnt}')


    state_dict = {}
    for stage in compiled_stages:
        state_dict.update(stage.state_dict())

    module.load_state_dict(state_dict)
    module.eval()
    correct_cnt = 0
    all_cnt = 0
    for x_batch, y_batch in valid_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out = module(x_batch)
        preds = out.argmax(-1)
        correct_cnt = (preds == y_batch).sum()
        all_cnt = len(y_batch)
        correct_cnt += correct_cnt.item()
        all_cnt += all_cnt
    print(f'eval accuracy: {correct_cnt / all_cnt}')

if __name__ == '__main__':
    test_main(split_into_equal_size(2))
