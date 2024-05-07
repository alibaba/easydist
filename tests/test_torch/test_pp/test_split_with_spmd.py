# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# torchrun --nproc_per_node=2 tests/test_torch/test_pp/test_split_with_spmd.py
import logging
import os
import random
import threading
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import cast

import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torchvision.models import (alexnet, densenet121, efficientnet_b0, resnet18, swin_t, vgg19,
                                vit_b_16)

from easydist import easydist_setup
import easydist.config as mdconfig
from easydist.torch.compile_auto import (easydist_shard, preprocess_traced_graph, sharded_tensor)
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.device_mesh import (get_device_mesh, get_pp_rank, get_pp_size, set_device_mesh,
                                        spmd_device_mesh)
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, annotate_split_points,
                                                             compile_pipeline,
                                                             split_into_equal_size)
from easydist.torch.experimental.pp.microbatch import \
    split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.runtime import ScheduleGPipe
from easydist.torch.experimental.pp.split_utils import (clear_pp_compile_states,
                                                        get_updated_params_states)
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.init_helper import (SetParaInitHelper, init_contiguous_buf, materialize_zero)
from easydist.torch.passes import (comm_optimize, fix_embedding, runtime_prof, tile_comm)
from easydist.torch.passes.fix_node_order import fix_node_order
from easydist.torch.passes.rule_override import rule_override_by_graph
from easydist.torch.passes.sharding import (sharding_transform, sharding_transform_dtensor)
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.utils import rgetattr, rsetattr


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


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(1024)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


class Foo1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(1024)
        self.linear0_0 = torch.nn.Linear(1024, 512)
        self.linear0_1 = torch.nn.Linear(512, 256)
        self.linear1 = torch.nn.Linear(256, 1024)

    def forward(self, x):
        x = self.norm(x)
        x0 = self.linear0_0(x)
        x0 = self.linear0_1(x0)
        x1 = self.linear1(x0)
        y = x + x1
        return y.relu()


def train_step(input, label, model, opt):
    if opt is not None:
        opt.zero_grad()
    out = model(input)
    out = out.view(out.shape[0], -1).mean(dim=1)
    loss = (out - label).pow(2).mean()
    if opt is not None:
        loss.backward()
        opt.step()
    return loss


def train_step_gpt(input, label, model, opt):
    if opt is not None:
        opt.zero_grad()
    out = model(input)
    loss = 0
    for key in [
            'attentions', 'cross_attentions', 'hidden_states', 'pooler_output', 'pask_key_values',
            'last_hidden_state'
    ]:
        if hasattr(out, key) and (attr := getattr(out, key)) is not None:
            if isinstance(attr, torch.Tensor):
                attr_broadcast = attr.permute([i for i in range(attr.dim() - 1, -1, -1)])
                loss += (attr_broadcast - label).pow(2).mean()
            elif isinstance(attr, (tuple, list)):
                for a in attr:
                    attr_broadcast = a.permute([i for i in range(a.dim() - 1, -1, -1)])
                    loss += (attr_broadcast - label).pow(2).mean()
    if opt is not None:
        loss.backward()
        opt.step()
    return loss


def train_step_t5(input, label, model, opt):
    label = torch.ones_like(input)
    if opt is not None:
        opt.zero_grad()
    loss = model(input_ids=input, labels=label).loss
    if opt is not None:
        loss.backward()
        opt.step()
    return loss


def gen_rand_input_foo():
    return torch.rand(16, 1024)


def gen_rand_input_imagenet():
    return torch.rand(16, 3, 224, 224)


def factory_gen_rand_input_ids(vocab_size):

    def gen_rand_input_ids():
        return torch.randint(0, vocab_size, (4, 256))

    return gen_rand_input_ids


def gen_rand_input_vit():
    return torch.rand(16, 3, 224, 224).half()


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())

    return model


def test_main(module, split_ann_or_policy, rand_input_gen_method, train_step_func):
    device = torch.device("cuda")
    module = module.train().to(device)
    module = broadcast_module(module)
    opt_config = {
        'lr': 0.123456789,
        'momentum': 0.9,
        'foreach': True,
    }
    opt = None  # inference only
    # opt = torch.optim.Adam(module.parameters(), **opt_config)
    opt = torch.optim.SGD(module.parameters(), **opt_config)

    if isinstance(split_ann_or_policy, set):
        annotate_split_points(module, split_ann_or_policy)
        nstages = len(split_ann_or_policy) + 1
    else:
        nstages, module = split_ann_or_policy(module)

    module_copy = deepcopy(module)
    opt_copy = type(opt)(module_copy.parameters(), **opt_config)

    if opt is None:
        module = module.eval()
        module_copy = module_copy.eval()

    rand_input = rand_input_gen_method().to(device)
    label = torch.tensor([random.random() for _ in range(rand_input.shape[0])]).to(device)
    train_func_args = (rand_input, label, module, opt)
    train_func_kwargs = {}

    sharded_graph, pp_compiled_meta, pp_compiled_stages, pp_local_gm, shard_args, shard_states = _compile_auto_local(
        train_step_func,
        "fake",
        SetParaInitHelper(),
        module.__class__.__name__,
        train_func_args,
        train_func_kwargs,
        nstages=nstages,
        num_chunks=1)

    epochs = 2
    dataset = []
    for _ in range(epochs):
        rand_input = rand_input_gen_method().to(device)
        torch.distributed.broadcast(rand_input, src=0)
        label = torch.tensor([random.random() for _ in range(rand_input.shape[0])]).to(device)
        torch.distributed.broadcast(label, src=0)
        dataset.append((rand_input, label))

    seed()
    with torch.no_grad():
        for rand_input, label in dataset:
            args = (rand_input, label, module, opt)
            kwargs = {}
            args, kwargs = shard_args(*args, **kwargs)
            sharded_flatten = pytree.tree_flatten([args, kwargs])[0]
            input_nodes = {
                node_name: sharded
                for node_name, sharded in zip(
                    pp_compiled_meta.input_nodes_flatten[-len(sharded_flatten):], sharded_flatten)
            }
            pp_local_gm(**input_nodes)

    state_dict = {}
    optimizer_state_dict = {}
    for stage in pp_compiled_stages:
        state_dict.update(stage.state_dict())
        optimizer_state_dict.update(stage.optimizer_state_dict())

    params = dict(module_copy.named_parameters())
    buffers = dict(module_copy.named_buffers())
    _, named_states = get_optstates(module_copy, opt_copy, params)
    named_states, state_tensor_num = shard_states(params, buffers, named_states)

    seed()
    with torch.no_grad():
        for rand_input, label in dataset:
            args = (rand_input, label, module_copy, opt_copy)
            kwargs = {}
            args, kwargs = shard_args(*args, **kwargs)
            params, buffers, named_states, grads, ret = sharded_graph(
                params, buffers, named_states, args, kwargs)
    state_dict_copy = {**params, **buffers}
    optimizer_state_dict_copy = named_states

    assert all(torch.allclose(state_dict[k], state_dict_copy[k]) for k in state_dict_copy)
    assert all(
        torch.allclose(optimizer_state_dict[k1][k2], optimizer_state_dict_copy[k1][k2])
        for k1 in optimizer_state_dict_copy for k2 in optimizer_state_dict_copy[k1])

    print(f"{module.__class__.__name__} pass")


sharding_sol = None
sol_rdy_cond = threading.Condition()


def fetch_strategy():
    with sol_rdy_cond:
        sol_rdy_cond.wait()

    return sharding_sol


def _compile_auto_local(func,
                        tracing_mode,
                        init_helper,
                        input_signature,
                        args,
                        kwargs,
                        nstages,
                        schedule_cls=ScheduleGPipe,
                        args_chunk_spec=None,
                        kwargs_chunk_spec=None,
                        outputs_chunk_spec=None,
                        num_chunks=1,
                        strict=True):
    module, opt = None, None

    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, torch.nn.Module):
            assert module is None, "Only support single nn.Module in args now"
            module = arg
        if isinstance(arg, torch.optim.Optimizer):
            assert opt is None, "Only support single Optimizer in args now"
            opt = arg

    params, buffers = {}, {}
    if module is not None:
        params = dict(module.named_parameters())
        buffers = dict(module.named_buffers())

        if isinstance(init_helper, SetParaInitHelper):
            init_helper.module = module

    flat_named_states, named_states = get_optstates(module, opt, params)

    # fix for sgd withtout momentum
    if all(state is None for state in flat_named_states):
        named_states = {}
        flat_named_states, _ = pytree.tree_flatten(named_states)

    state_tensor_num = len(params) + len(buffers) + len(flat_named_states)

    def stateless_func(func, params, buffers, named_states, args, kwargs):
        clear_pp_compile_states()
        with stateless._reparametrize_module(
                cast(torch.nn.Module, module), {
                    **params,
                    **buffers
                }, tie_weights=True) if module else nullcontext(), _rematerialize_optimizer(
                    opt, named_states, params) if opt else nullcontext():
            ret = func(*args, **kwargs)
        if (tup := get_updated_params_states()) != (None, None):
            params, named_states = tup
        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, num_chunks,
                                                             args_chunk_spec, kwargs_chunk_spec)

    with _enable_compile(), SplitPatcher(module, opt) if schedule_cls else nullcontext():
        traced_graph = make_fx(partial(stateless_func, func),
                               tracing_mode=tracing_mode,
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, named_states,
                                                             args_split[0], kwargs_split[0])

    traced_graph.graph.eliminate_dead_code()
    traced_graph = preprocess_traced_graph(traced_graph)
    traced_graph.recompile()

    save_graphviz_dot(traced_graph, 'traced_graph')

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Lansong(TODO) Currently send strategy by rpc. But broadcast way is more efficient.
    rpc.init_rpc(f"ed_worker{rank}", rank=rank, world_size=world_size)
    if rank == 0:
        shape_info, opt_strategy, sharding_strategy, args_strategy, state_io_map = easydist_shard(
            traced_graph, state_tensor_num, input_signature, params, buffers, named_states, args,
            kwargs)

        with sol_rdy_cond:
            global sharding_sol
            sharding_sol = [
                shape_info, opt_strategy, sharding_strategy, args_strategy, state_io_map
            ]
            sol_rdy_cond.notify_all()
    else:
        shape_info, opt_strategy, sharding_strategy, args_strategy, state_io_map = rpc.rpc_sync(
            "ed_worker0", fetch_strategy, args=(), timeout=0)

    rpc.shutdown()

    with spmd_device_mesh():
        if mdconfig.use_dtensor:
            sharded_graph = sharding_transform_dtensor(traced_graph, sharding_strategy)
        else:
            sharded_graph = sharding_transform(traced_graph, opt_strategy, state_io_map)
            if mdconfig.enable_tile_comm:
                sharded_graph = runtime_prof(sharded_graph, tiling_prof=True)
                sharded_graph = tile_comm(sharded_graph)

    save_graphviz_dot(sharded_graph, f'sharded_graph_raw_{rank}')
    sharded_graph = fix_embedding(sharded_graph, recover=True)

    if not mdconfig.use_dtensor:
        if schedule_cls is None and mdconfig.comm_optimization is True:
            sharded_graph = runtime_prof(sharded_graph)
            sharded_graph = comm_optimize(sharded_graph,
                                          'rcpsp',
                                          grouping=True,
                                          mem_restrain=False)

        # override pytorch dtensor propagate rules to optimize dispater behavior
        if mdconfig.override_dtensor_rule is True:
            sharded_graph = rule_override_by_graph(sharded_graph, opt_strategy, shape_info)

    if mdconfig.log_level <= logging.DEBUG:
        sharded_graph.print_readable()

    # keep fake params, buffers, named_states
    fake_tensor_mode = FakeTensorMode()

    def wrap_fake(x):
        if isinstance(x, torch.Tensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    fake_params = pytree.tree_map(wrap_fake, params)
    fake_buffers = pytree.tree_map(wrap_fake, buffers)
    fake_named_states = pytree.tree_map(wrap_fake, named_states)

    def shard_states(params, buffers, named_states):
        with spmd_device_mesh():
            device_mesh = get_device_mesh()
            device = mdconfig.easydist_device

            # pre-shard params, buffers, named_states
            params_strategy = args_strategy[:len(params)]
            buffers_strategy = args_strategy[len(params):len(params) + len(buffers)]

            if mdconfig.use_contiguous_buffer:
                contiguous_buf = init_contiguous_buf(params, params_strategy, device_mesh)

            index = 0
            for idx, param_name in enumerate(params):
                materialize_fn = init_helper.get_materialize_fn()
                materialize_fn = partial(materialize_fn,
                                         param_buf_key=param_name,
                                         materialization_device=device)
                params[param_name] = sharded_tensor(params[param_name],
                                                    params_strategy[idx],
                                                    get_device_mesh(),
                                                    materialize_fn=materialize_fn)

                size = params[param_name]._local_tensor.numel()

                if mdconfig.use_contiguous_buffer:
                    contiguous_buf[index:index + size] = params[param_name]._local_tensor.view(-1)
                    params[param_name]._local_tensor = contiguous_buf[index:index + size].view(
                        params[param_name]._local_tensor.shape)

                if not mdconfig.use_dtensor:
                    params[param_name] = params[param_name]._local_tensor

                index += size

            for idx, buffer_name in enumerate(buffers):
                materialize_fn = init_helper.get_materialize_fn()
                materialize_fn = partial(materialize_fn,
                                         param_buf_key=buffer_name,
                                         materialization_device=device)
                buffers[buffer_name] = sharded_tensor(buffers[buffer_name],
                                                      buffers_strategy[idx],
                                                      get_device_mesh(),
                                                      materialize_fn=materialize_fn)
                if not mdconfig.use_dtensor:
                    buffers[buffer_name] = buffers[buffer_name]._local_tensor

                # use zero init for optimizer states
            flat_named_states, named_states_spec = pytree.tree_flatten(named_states)
            state_tensor_num = len(params) + len(buffers)
            materialize_fn = partial(materialize_zero, materialization_device=device)
            for i in range(len(flat_named_states)):
                if isinstance(flat_named_states[i], torch.Tensor):
                    flat_named_states[i] = sharded_tensor(flat_named_states[i],
                                                          args_strategy[state_tensor_num],
                                                          get_device_mesh(),
                                                          materialize_fn=materialize_fn)
                    if not mdconfig.use_dtensor:
                        flat_named_states[i] = flat_named_states[i]._local_tensor

                    state_tensor_num += 1

            named_states = pytree.tree_unflatten(flat_named_states, named_states_spec)

            return named_states, state_tensor_num

    named_states, state_tensor_num = shard_states(params, buffers, named_states)

    if schedule_cls is not None:
        pp_rank, pp_size = get_pp_rank(), get_pp_size()
        traced_graph_node_metas = {node.name: node.meta for node in traced_graph.graph.nodes}
        sharded_graph = fix_node_order(sharded_graph)
        stateless_func_args = (params, buffers, named_states, args, kwargs)
        save_graphviz_dot(sharded_graph, 'sharded_graph')
        pp_compiled_meta, pp_compiled_stages, pp_local_gm, _ = compile_pipeline(
            sharded_graph, nstages, stateless_func_args, phs_stragegies=None, strict=strict)
        save_graphviz_dot(pp_local_gm, 'pp_local_gm')

    def shard_args(*args, **kwargs):
        with spmd_device_mesh():
            flatten_args, args_specs = pytree.tree_flatten([args, kwargs])

            device = mdconfig.easydist_device
            materialize_fn = partial(materialize_zero, materialization_device=device)

            args_strategy_idx = state_tensor_num
            for i in range(len(flatten_args)):
                if isinstance(flatten_args[i], torch.Tensor):
                    flatten_args[i] = sharded_tensor(flatten_args[i].detach(),
                                                     args_strategy[args_strategy_idx],
                                                     get_device_mesh(),
                                                     materialize_fn=materialize_fn)
                    if not mdconfig.use_dtensor:
                        flatten_args[i] = flatten_args[i]._local_tensor
                    args_strategy_idx += 1
            args, kwargs = pytree.tree_unflatten(flatten_args, args_specs)
        return args, kwargs

    return sharded_graph, pp_compiled_meta, pp_compiled_stages, pp_local_gm, shard_args, shard_states


def get_optstates(module, opt, params):
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

    flat_named_states, _ = pytree.tree_flatten(named_states)
    return flat_named_states, named_states


def init():
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    set_device_mesh(DeviceMesh("cuda", [[[0], [1]]], mesh_dim_names=["spmd0", "spmd1", "pp"]))


if __name__ == '__main__':
    init()
    # human annotated split points
    test_main(Foo(), {'norm'}, gen_rand_input_foo, train_step)
    # test_main(Foo1(), {  # meta graph failure (batchnorm 1d op)
    #     'norm',
    #     'linear0_1',
    # }, gen_rand_input_foo, train_step)
    test_main(alexnet(), {
        'features.10',
        'classifier.3',
    }, gen_rand_input_imagenet, train_step)
    test_main(
        densenet121(), {
            'features.denseblock1.denselayer4.norm2',
            'features.transition2.conv',
            'features.denseblock4.denselayer1.relu1',
            'features',
        }, gen_rand_input_imagenet, train_step)
    # test_main(  # memory failure?
    #     efficientnet_b0(), {
    #         'features.2.0.block.1',
    #         'features.4.1.block.3',
    #         'features.6.1.block.3',
    #         'features.8',
    #     }, gen_rand_input_imagenet, train_step)
    test_main(resnet18(), {
        'layer1',
        'layer2',
        'layer3',
        'layer4',
    }, gen_rand_input_imagenet, train_step)
    # test_main(  # meta graph failure (empty op)
    #     swin_t(), {
    #         'features.2.reduction',
    #         'features.3.0.mlp.1',
    #         'features.5.1.attn.qkv',
    #         'features.7.0.stochastic_depth',
    #     }, gen_rand_input_imagenet, train_step)
    test_main(vgg19(), {
        'features.10',
        'features.20',
        'classifier.3',
    }, gen_rand_input_imagenet, train_step)
    # test_main(
    #     vit_b_16().half(), {
    #         'encoder.layers.encoder_layer_1.self_attention',
    #         'encoder.layers.encoder_layer_5.mlp.3',
    #         'encoder.layers.encoder_layer_9.ln_2',
    #     }, gen_rand_input_vit, train_step)

    # test split_into_equal_size
    test_main(Foo(), split_into_equal_size(2), gen_rand_input_foo, train_step)
    test_main(Foo1(), split_into_equal_size(2), gen_rand_input_foo, train_step)
    test_main(alexnet(), split_into_equal_size(3), gen_rand_input_imagenet, train_step)
    test_main(densenet121(), split_into_equal_size(5), gen_rand_input_imagenet, train_step)
    test_main(efficientnet_b0(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)
    test_main(resnet18(), split_into_equal_size(4), gen_rand_input_imagenet, train_step)
    test_main(swin_t(), split_into_equal_size(10), gen_rand_input_imagenet, train_step)
    test_main(vgg19(), split_into_equal_size(3), gen_rand_input_imagenet, train_step)
    # test_main(vit_b_16().half(), split_into_equal_size(10), gen_rand_input_vit,
    #          train_step)

    # ======== transformers ========
    from transformers import OpenAIGPTConfig, OpenAIGPTModel
    test_main(OpenAIGPTModel(OpenAIGPTConfig()), {
        'h.3',
        'h.6',
        'h.9',
    }, factory_gen_rand_input_ids(OpenAIGPTConfig().vocab_size), train_step_gpt)

    from transformers import AutoModel
    test_main(AutoModel.from_pretrained("bert-base-uncased"), {
        'encoder.layer.3',
        'encoder.layer.6',
        'encoder.layer.9',
    }, factory_gen_rand_input_ids(30522), train_step_gpt)

    # from transformers import GPT2Config, GPT2Model
    # test_main(GPT2Model(GPT2Config()), {
    #     'h.3',
    #     'h.6',
    #     'h.9',
    # }, factory_gen_rand_input_ids(50257), train_step_gpt)

    # from transformers import LlamaConfig, LlamaModel
    # config = LlamaConfig()
    # config.num_attention_heads = config.num_key_value_heads = 16
    # config.num_hidden_layers = 16
    # config.hidden_size = 768
    # config.use_cache = False
    # test_main(LlamaModel(config), {
    # 'layers.3',
    # 'layers.7',
    # 'layers.11',
    # }, factory_gen_rand_input_ids(config.vocab_size), train_step_gpt)

    print("All tests passed!")
