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

import logging
import os
import pickle
import time
from functools import partial
from typing import Any, cast
from contextlib import nullcontext

import numpy
import rich
import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._tensor import (DeviceMesh, DTensor, Replicate, distribute_tensor)
from torch.fx._pytree import tree_flatten_spec
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless

import metadist.config as mdconfig
from metadist.autoflow.solver import AutoFlowSolver
from metadist.metashard import metair
from metadist.torch.bridge import (get_torch_sharding_strategy, to_torch_spmd, torch2meta_graph)
from metadist.torch.experimental.decomp_utils import METADIST_DECOMP_TABLE
from metadist.torch.experimental.init_helper import (SetParaInitHelper, init_contiguous_buf,
                                                     materialize_zero)
from metadist.torch.passes import (eliminate_detach, fix_addmm_bias, fix_convoluation_bias,
                                   fix_embedding, fix_meta_device, sharding_transform)
from metadist.torch.device_mesh import get_device_mesh, set_device_mesh
from metadist.torch.mem_anaylize import mem_anaylize
from metadist.torch.sharding_interpreter import MDTorchShardingAnn
from metadist.torch.utils import (_enable_compile, _rematerialize_optimizer, _sharding_ann_env)
from metadist.utils import rgetattr, rsetattr
from metadist.utils.testing import TorchMockDeviceMesh

# for pickle dump opt_strategy
import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


def preprocess_traced_graph(fx_module: torch.fx.GraphModule):
    fx_module = fix_meta_device(fx_module)
    fx_module = fix_embedding(fx_module)
    fx_module = fix_addmm_bias(fx_module)
    # fx_module = fix_convoluation_bias(fx_module)
    fx_module = eliminate_detach(fx_module)

    fx_module.recompile()

    return fx_module


def metadist_shard(fx_module: torch.fx.GraphModule, state_tensor_num, *args, **kwargs):

    # (1) preprocess pass
    fx_module = preprocess_traced_graph(fx_module)

    if mdconfig.log_level <= logging.DEBUG:
        fx_module.print_readable()

    # (2) sharding annotation
    with _sharding_ann_env():
        start_t = time.perf_counter()
        sharding_interpreter = MDTorchShardingAnn(fx_module)
        flatten_args = tree_flatten_spec(list(args) + list(kwargs.values()), fx_module._in_spec)
        sharding_info, shape_info = sharding_interpreter.run(*flatten_args)
        logger.info(f"[MDTorchShardingAnn.time]:\t {time.perf_counter() - start_t} s.")
        if mdconfig.log_level <= logging.DEBUG:
            rich.print("sharding_info:\n", sharding_info)
            rich.print("shape_info:\n", shape_info)

    # sync sharding info for all process
    torch.distributed.broadcast_object_list(sharding_info, src=0, device="cuda")

    # (3) translate fx.GraphModule into MetaGraph
    meta_graph = torch2meta_graph(fx_module, state_tensor_num, sharding_info, shape_info)
    meta_graph.dump()

    if mdconfig.log_level <= logging.DEBUG:
        rich.print(meta_graph)

    # (4) construct AutoFlowSolver and run ILP
    device_mesh = get_device_mesh()
    device_mesh_shape = (device_mesh.size(0), device_mesh.size(1))

    total_memery = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    solver = AutoFlowSolver(device_mesh_shape, total_memery=total_memery)

    if mdconfig.enable_graph_coarsen:
        logger.info(f"enable graph coarsen with level {mdconfig.coarsen_level}.")
        solver.add_coarsen_graph(meta_graph)
    else:
        solver.add_graph(meta_graph)

    start_t = time.perf_counter()
    if mdconfig.enable_graph_coarsen:
        opt_strategy = solver.ilp_solve()
    else:
        opt_strategy = solver.ilp_optimize()
    logger.info(f"[AutoFlowSolver.time]:\t {time.perf_counter() - start_t} s.")

    if mdconfig.log_level <= logging.DEBUG:
        rich.print(opt_strategy)

    sharding_strategies = get_torch_sharding_strategy(fx_module, opt_strategy)

    if mdconfig.log_level <= logging.DEBUG:
        rich.print(sharding_strategies)

    return shape_info, meta_graph, opt_strategy, sharding_strategies


@torch.no_grad()
def sharded_tensor(tensor, strategy, mesh, materialize_fn):
    # if tensor is DTensor, redistribute it
    if isinstance(tensor, DTensor):
        return tensor.redistribute(mesh, strategy)

    # materialize FakeTensor and distribute_tensor
    if isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):
        tensor = materialize_fn(tensor)
    if tensor.is_meta:
        tensor = materialize_fn(tensor)
    return distribute_tensor(tensor, mesh, strategy)


@torch.no_grad()
def dtensor_to_tensor(leaf):
    if isinstance(leaf, DTensor):
        replicate_leaf = leaf.redistribute(leaf.device_mesh, [Replicate()] * len(leaf.placements))
        return replicate_leaf.to_local()
    return leaf


def _compile(func, tracing_mode, init_helper, input_signature, args, kwargs):

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

    named_states = {}

    if opt is not None:
        # assign grad and warm up optimizer
        mode = nullcontext()
        for name in dict(module.named_parameters()):
            with torch.no_grad():
                rsetattr(module, name + ".grad", torch.zeros_like(rgetattr(module, name).data))
                if isinstance(rgetattr(module, name).data, FakeTensor):
                    mode = FakeTensorMode()
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

    state_tensor_num = len(params) + len(buffers) + len(flat_named_states)

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

    with _enable_compile():
        traced_graph = make_fx(partial(stateless_func, func),
                               tracing_mode=tracing_mode,
                               decomposition_table=METADIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, named_states, args,
                                                             kwargs)

    traced_graph.graph.eliminate_dead_code()
    traced_graph = preprocess_traced_graph(traced_graph)
    traced_graph.recompile()

    if mdconfig.enable_compile_cache:
        os.makedirs(mdconfig.compile_cache_dir, exist_ok=True)
        compiled_cache_file = os.path.join(mdconfig.compile_cache_dir, f"./{input_signature}.pkl")

    if mdconfig.enable_compile_cache and os.path.exists(compiled_cache_file):
        logger.info(f"load compiled result from {compiled_cache_file}.")
        shape_info, sharding_strategy, opt_strategy, args_strategy = pickle.load(
            open(compiled_cache_file, "rb"))
        # mem_anaylize(traced_graph, shape_info, opt_strategy)
        sharded_graph = sharding_transform(traced_graph, sharding_strategy)
        sharded_graph = fix_embedding(sharded_graph, recover=True)
    else:
        shape_info, meta_graph, opt_strategy, sharding_strategy = metadist_shard(
            traced_graph, state_tensor_num, params, buffers, named_states, args, kwargs)
        # mem_anaylize(traced_graph, shape_info, opt_strategy)
        sharded_graph = sharding_transform(traced_graph, sharding_strategy)
        sharded_graph = fix_embedding(sharded_graph, recover=True)
        args_strategy = meta_graph.get_input_strategy(opt_strategy)
        args_strategy = [[to_torch_spmd(i) for i in var_strategy]
                         for var_strategy in args_strategy]

        if mdconfig.enable_compile_cache and torch.distributed.get_rank() == 0:
            pickle.dump([shape_info, sharding_strategy, opt_strategy, args_strategy],
                        open(compiled_cache_file, "wb"))
            logger.info(f"compiled result saved in {compiled_cache_file}.")

    # do not use mock device after get sharded_graph
    device_mesh = get_device_mesh()
    if isinstance(device_mesh, TorchMockDeviceMesh):
        if device_mesh.debug_only:
            world_size = torch.distributed.get_world_size()
            mesh_shape = numpy.array(range(world_size)).reshape(1, -1).tolist()
        else:
            mesh_shape = device_mesh.shape
        mesh = DeviceMesh("cuda", mesh_shape)
        set_device_mesh(mesh)

    device = mdconfig.metadist_device

    # keep fake params, buffers, named_states
    fake_tensor_mode = FakeTensorMode()

    def wrap_fake(x):
        if isinstance(x, torch.Tensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    fake_params = pytree.tree_map(wrap_fake, params)
    fake_buffers = pytree.tree_map(wrap_fake, buffers)
    fake_named_states = pytree.tree_map(wrap_fake, named_states)

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
            state_tensor_num += 1

    named_states = pytree.tree_unflatten(flat_named_states, named_states_spec)

    @torch.no_grad()
    def compiled_func_return(graph, *args, **kwargs):
        nonlocal params, buffers, named_states

        # tensor to DTensor
        # (TODO) the redistributed of params, buffers, named_states should in graph
        flatten_args, args_specs = pytree.tree_flatten([args, kwargs])

        device = mdconfig.metadist_device
        materialize_fn = partial(materialize_zero, materialization_device=device)

        args_strategy_idx = state_tensor_num
        for i in range(len(flatten_args)):
            if isinstance(flatten_args[i], torch.Tensor):
                flatten_args[i] = sharded_tensor(flatten_args[i],
                                                 args_strategy[args_strategy_idx],
                                                 get_device_mesh(),
                                                 materialize_fn=materialize_fn)
                args_strategy_idx += 1

        args, kwargs = pytree.tree_unflatten(flatten_args, args_specs)

        # run the sharded_graph
        params, buffers, named_states, grads, sharded_out = graph(params, buffers, named_states,
                                                                  args, kwargs)

        for para_name in params:
            params[para_name].grad = grads[para_name]

        # out from DTensor to Tensor
        local_out = pytree.tree_map(dtensor_to_tensor, sharded_out)

        return local_out

    class MDCompiledFunc:

        def __init__(self, func, graph) -> None:
            self.func = func
            self.graph = graph

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.func(self.graph, *args, **kwargs)

        def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
            return self.func(graph, *args, **kwargs)

        def get_state(self):
            return params, buffers, named_states

        def compile_mono_graph(self, *args: Any, **kwargs: Any):
            with _enable_compile():
                fx_module = make_fx(partial(stateless_func, func),
                                    tracing_mode=tracing_mode,
                                    decomposition_table=METADIST_DECOMP_TABLE,
                                    _allow_non_fake_inputs=False)(fake_params, fake_buffers,
                                                                  fake_named_states, args, kwargs)

            fx_module.graph.eliminate_dead_code()
            fx_module.recompile()

            fx_module = preprocess_traced_graph(fx_module)

            sharded_fx_module = sharding_transform(fx_module, sharding_strategy)
            sharded_fx_module = fix_embedding(sharded_fx_module, recover=True)

            if mdconfig.log_level <= logging.DEBUG:
                sharded_fx_module.print_readable()

            return sharded_fx_module

        def parameters(self):
            return params.values()

        def named_parameters(self):
            return params

    # release all cuda memory from module here
    # the param maintain in the local of compiled function.
    if module is not None:
        module.to("meta")

    return MDCompiledFunc(compiled_func_return, sharded_graph)
