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
import threading
from functools import partial
from typing import Any, cast
from contextlib import nullcontext

import numpy
import rich
import torch
import torch.utils._pytree as pytree
import torch.distributed.rpc as rpc
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._tensor import (DeviceMesh, DTensor, Replicate, distribute_tensor)
from torch.fx._pytree import tree_flatten_spec
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.nn.utils import stateless
from torch._functorch.partitioners import default_partition

import easydist.config as mdconfig
from easydist.autoflow.solver import AutoFlowSolver
from easydist.torch.bridge import (get_torch_sharding_strategy, to_torch_spmd, torch2meta_graph)
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.PipelineStage import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import SplitPatcher, compile_pipeline
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states, set_backward_flag, set_step_flag, set_updated_params_states
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.init_helper import (SetParaInitHelper, init_contiguous_buf, materialize_zero)
from easydist.torch.passes import (eliminate_detach, fix_addmm_bias, fix_convoluation_bias,
                                   tile_comm, runtime_prof, fix_embedding, fix_meta_device,
                                   sharding_transform, sharding_transform_dtensor,
                                   AllocatorProfiler, ModuleProfilingInfo)
from easydist.torch.device_mesh import get_device_mesh, get_pp_group, get_pp_rank, get_pp_size, set_device_mesh, spmd_device_mesh
from easydist.torch.passes import comm_optimize, rule_override_by_graph, create_edinfo
from easydist.torch.passes.fix_sharding_node_order import fix_order, fix_sharding_node_order
from easydist.torch.schedule.ilp_memory_scheduler import ILPMemoryScheduler
from easydist.torch.schedule.efficient_memory_scheduler import EfficientMemoryScheduler
from easydist.torch.schedule.graph_mem_plan import GraphMemPlan
from easydist.torch.sharding_interpreter import EDTorchShardingAnn
from easydist.torch.utils import (_enable_compile, _rematerialize_optimizer, _sharding_ann_env)
from easydist.utils import rgetattr, rsetattr
from easydist.utils.testing import TorchMockDeviceMesh

# for pickle dump opt_strategy
import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)

sharding_sol = None
sol_rdy_cond = threading.Condition()

mem_sol = None
mem_addr_rdy_cond = threading.Condition()


def preprocess_traced_graph(fx_module: torch.fx.GraphModule):
    fx_module = fix_meta_device(fx_module)
    fx_module = fix_embedding(fx_module)
    fx_module = fix_addmm_bias(fx_module)
    # fx_module = fix_convoluation_bias(fx_module)
    fx_module = eliminate_detach(fx_module)

    fx_module.recompile()

    return fx_module


def easydist_shard(fx_module: torch.fx.GraphModule, state_tensor_num, input_signature, *args,
                   **kwargs):
    # only called by rank 0
    if mdconfig.enable_compile_cache:
        os.makedirs(mdconfig.compile_cache_dir, exist_ok=True)
        compiled_cache_file = os.path.join(mdconfig.compile_cache_dir, f"./{input_signature}.pkl")

    find_cache = mdconfig.enable_compile_cache and os.path.exists(compiled_cache_file)

    if find_cache:
        logger.info(f"load compiled result from {compiled_cache_file}.")
        shape_info, opt_strategy, sharding_strategy, args_strategy, state_io_map = pickle.load(
            open(compiled_cache_file, "rb"))
    else:
        if mdconfig.log_level <= logging.DEBUG:
            fx_module.print_readable()

        # (1) sharding annotation
        with _sharding_ann_env():
            start_t = time.perf_counter()
            sharding_interpreter = EDTorchShardingAnn(fx_module)
            flatten_args = tree_flatten_spec(
                list(args) + list(kwargs.values()), fx_module._in_spec)
            sharding_info, shape_info = sharding_interpreter.run(*flatten_args)
            logger.info(f"[EDTorchShardingAnn.time]:\t {time.perf_counter() - start_t} s.")
            if mdconfig.log_level <= logging.DEBUG:
                rich.print("sharding_info:\n", sharding_info)
                rich.print("shape_info:\n", shape_info)

        fx_module = create_edinfo(fx_module, sharding_info, shape_info)

        # (2) translate fx.GraphModule into MetaGraph
        meta_graph = torch2meta_graph(fx_module, state_tensor_num, sharding_info, shape_info)

        if mdconfig.log_level <= logging.DEBUG:
            rich.print(meta_graph)

        # (3) construct AutoFlowSolver and run ILP
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

        sharding_strategy = get_torch_sharding_strategy(fx_module, opt_strategy)

        if mdconfig.log_level <= logging.DEBUG:
            rich.print(sharding_strategy)

        args_strategy = meta_graph.get_input_strategy(opt_strategy)
        args_strategy = [[to_torch_spmd(i) for i in var_strategy]
                         for var_strategy in args_strategy]

        state_io_map = meta_graph.state_io_map

        if mdconfig.enable_compile_cache:
            pickle.dump([shape_info, opt_strategy, sharding_strategy, args_strategy, state_io_map],
                        open(compiled_cache_file, "wb"))
            logger.info(f"compiled result saved in {compiled_cache_file}.")

    return shape_info, opt_strategy, sharding_strategy, args_strategy, state_io_map


@torch.no_grad()
def sharded_tensor(tensor, strategy, mesh, materialize_fn):
    # if tensor is DTensor, redistribute it
    if isinstance(tensor, DTensor):
        return tensor.redistribute(mesh, strategy)

    # materialize FakeTensor and distribute_tensor
    if isinstance(tensor, FakeTensor):
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

def fetch_strategy():
    with sol_rdy_cond:
        sol_rdy_cond.wait()

    return sharding_sol

def fetch_mem_sol():
    with mem_addr_rdy_cond:
        mem_addr_rdy_cond.wait()

    return mem_sol


def _compile_auto(func,
                  tracing_mode,
                  init_helper,
                  input_signature,
                  args,
                  kwargs,
                  schedule_cls=ScheduleGPipe,
                  args_chunk_spec=None,
                  kwargs_chunk_spec=None,
                  outputs_chunk_spec=None,
                  num_chunks=1,
                  strict=False):
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

    # fix for sgd withtout momentum
    if all(state is None for state in flat_named_states):
        named_states = {}
        flat_named_states, _ = pytree.tree_flatten(named_states)

    state_tensor_num = len(params) + len(buffers) + len(flat_named_states)

    def stateless_func(func, params, buffers, named_states, args, kwargs):
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
        clear_pp_compile_states()
        traced_graph = make_fx(partial(stateless_func, func),
                               tracing_mode=tracing_mode,
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, named_states, args_split[0],
                                                             kwargs_split[0])

    traced_graph.graph.eliminate_dead_code()
    traced_graph = preprocess_traced_graph(traced_graph)
    traced_graph.recompile()

    save_graphviz_dot(traced_graph, 'traced_graph')

    if mdconfig.dump_fx_graph:
        print(f"node num in traced graph: {len(traced_graph.graph.nodes)}")
        drawer = FxGraphDrawer(traced_graph, "traced_fx", ignore_getattr=True)
        dot_graphs = drawer.get_all_dot_graphs()
        for name, dot_graph in dot_graphs.items():
            dot_graph.write_jpg(f"./tmp/{name}.jpg")
            dot_graph.write_raw(f"./tmp/{name}.txt")

        # seperate fwd/bwd graph
        fwd_graph, bwd_graph = default_partition(traced_graph, None, num_fwd_outputs=1)

        fwd_drawer = FxGraphDrawer(fwd_graph, "fwd_traced_fx", ignore_getattr=True)
        fwd_dot_graphs = fwd_drawer.get_all_dot_graphs()
        for name, dot_graph in fwd_dot_graphs.items():
            dot_graph.write_jpg(f"./tmp/{name}.jpg")
            dot_graph.write_raw(f"./tmp/{name}.txt")

        bwd_drawer = FxGraphDrawer(bwd_graph, "bwd_traced_fx", ignore_getattr=True)
        bwd_dot_graphs = bwd_drawer.get_all_dot_graphs()
        for name, dot_graph in bwd_dot_graphs.items():
            dot_graph.write_jpg(f"./tmp/{name}.jpg")
            dot_graph.write_raw(f"./tmp/{name}.txt")


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
            sharded_graph = comm_optimize(sharded_graph, 'rcpsp', grouping=True, mem_restrain=False)

        # override pytorch dtensor propagate rules to optimize dispater behavior
        if mdconfig.override_dtensor_rule is True:
            sharded_graph = rule_override_by_graph(sharded_graph, opt_strategy, shape_info)


    if mdconfig.log_level <= logging.DEBUG:
        sharded_graph.print_readable()

    if mdconfig.dump_fx_graph:
        print(f"node num in sharded graph: {len(sharded_graph.graph.nodes)}")
        drawer = FxGraphDrawer(sharded_graph, "shard_fx", ignore_getattr=True)
        dot_graphs = drawer.get_all_dot_graphs()
        for name, dot_graph in dot_graphs.items():
            dot_graph.write_jpg(f"./tmp/{name}.jpg")
            dot_graph.write_raw(f"./tmp/{name}.txt")

    # keep fake params, buffers, named_states
    fake_tensor_mode = FakeTensorMode()

    def wrap_fake(x):
        if isinstance(x, torch.Tensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    fake_params = pytree.tree_map(wrap_fake, params)
    fake_buffers = pytree.tree_map(wrap_fake, buffers)
    fake_named_states = pytree.tree_map(wrap_fake, named_states)

    with spmd_device_mesh():
        params, buffers, named_states = pre_shard(args_strategy, params, buffers, named_states,
                                                  init_helper, mdconfig.easydist_device)

    if schedule_cls is not None:
        pp_rank, pp_size = get_pp_rank(), get_pp_size()
        traced_graph_node_metas = {
            node.name: node.meta
            for node in traced_graph.graph.nodes
        }
        sharded_graph = fix_order(sharded_graph)
        stateless_func_args = (params, buffers, named_states, args, kwargs)
        save_graphviz_dot(sharded_graph, 'sharded_graph')
        pp_compiled_meta, pp_compiled_stages, pp_local_gm, _ = compile_pipeline(
            sharded_graph, pp_size, stateless_func_args, strict=strict)
        # sharded_graph = pp_local_gm
        save_graphviz_dot(pp_local_gm, f'pp_local_gm')
        pipe = PipelineStage(
            schedule_cls=schedule_cls,
            local_gm=pp_local_gm,
            compiled_meta=pp_compiled_meta,
            stage_idx=pp_rank,
            compiled_stage=pp_compiled_stages[pp_rank],
            node_metas=traced_graph_node_metas,
            num_chunks=num_chunks,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            returns_chunk_spec=outputs_chunk_spec,
            group=get_pp_group(),
            device=torch.device(f"cuda:{rank}"),
            sharded_graph=sharded_graph,
        )
        return pipe

    if mdconfig.enable_memory_opt:
        rpc.init_rpc(f"ed_worker{rank}", rank=rank, world_size=world_size)
        if rank == 0:
            logging.info("profiling fx module's memory...")

        import __main__
        # setting allocator to profiling mode
        __main__.allocator_mode = 'profile'

        # save all profiling information in this dict
        profiling_info = ModuleProfilingInfo(rank)
        alloc_profiler = AllocatorProfiler(sharded_graph, profiling_info)
        _ = alloc_profiler.run([])

        if rank == 0:
            logging.info("finish profiling fx module's memory")
            graph_mem_info = alloc_profiler.create_graph_mem_info()

            if mdconfig.mem_opt_by_solver:
                mem_sched = ILPMemoryScheduler(
                                    sharded_graph, graph_mem_info, 1024*128)
            else:
                mem_sched = EfficientMemoryScheduler(
                                    sharded_graph, graph_mem_info)

            required_memory, temp_memory, schedules, ordered_schedules, mem_alloc_info, mem_locations = \
                                                mem_sched.gen_mem_addresses()
            #print(f"master proposes required_memory: {required_memory}")
            #print(f"master creates mem locations:\n{mem_locations}")

            with mem_addr_rdy_cond:
                global mem_sol
                mem_sol = [
                    required_memory, temp_memory, ordered_schedules, mem_alloc_info, mem_locations
                ]
                mem_addr_rdy_cond.notify_all()

            assert len(ordered_schedules) == len(mem_sched.nodes_to_schedule), \
                f"schedule {len(ordered_schedules)} nodes, but totally {len(mem_sched.nodes_to_schedule)} nodes"
        else:
            required_memory, temp_memory, ordered_schedules, mem_alloc_info, mem_locations = rpc.rpc_sync(
                                "ed_worker0", fetch_mem_sol, args=(), timeout=0)
            #print(f"worker {rank} receives required_memory: {required_memory}")
            #print(f"worker {rank} receives mem locations:\n{mem_locations}")

        rpc.shutdown()

        graph_mem_plan = GraphMemPlan(required_memory, temp_memory)

        for name in ordered_schedules:
            if name in mem_alloc_info:
                alloc_list = mem_alloc_info[name]
                for alloc_info in alloc_list:
                    addr = alloc_info[1]
                    size = alloc_info[2]
                    if alloc_info[3] == 0:
                        is_temp_mem = True
                    else:
                        is_temp_mem = False

                    graph_mem_plan.append_addr_size(addr, size, is_temp_mem, name)

        #print("graph memory plan:")
        #print(str(graph_mem_plan))

        # setting allocator back to runtime mode
        alloc_profiler.load_memory_plan(graph_mem_plan)

        #print(f"graph details:\n{str(sharded_graph.graph)}\nend of graph\n")

    class EDCompiledFunc:

        def __init__(self, graph) -> None:
            self.graph = graph

        @torch.no_grad()
        def compiled_func(self, graph, *args, **kwargs):
            nonlocal params, buffers, named_states

            # tensor to DTensor
            # (TODO) the redistributed of params, buffers, named_states should in graph
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

            # run the sharded_graph
            if mdconfig.enable_memory_opt:
                __main__.allocator_mode = 'runtime'
                __main__.is_customized = True
                params, buffers, named_states, grads, sharded_out = graph(
                    params, buffers, named_states, args, kwargs)
                __main__.is_customized = False
            else:
                params, buffers, named_states, grads, sharded_out = graph(
                    params, buffers, named_states, args, kwargs)

            for para_name in params:
                params[para_name].grad = grads[para_name]

            # out from DTensor to Tensor
            local_out = pytree.tree_map(dtensor_to_tensor, sharded_out)
            #print(f"local_out: {local_out}, \nshape: {local_out.shape}")

            return local_out

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.compiled_func(self.graph, *args, **kwargs)

        def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
            return self.compiled_func(graph, *args, **kwargs)

        def get_state(self):
            return params, buffers, named_states

        def compile_mono_graph(self, *args: Any, **kwargs: Any):
            with _enable_compile():
                fx_module = make_fx(partial(stateless_func, func),
                                    tracing_mode=tracing_mode,
                                    decomposition_table=EASYDIST_DECOMP_TABLE,
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
    if module is not None and not isinstance(module.parameters().__next__(), FakeTensor):
        module.to("meta")

    return EDCompiledFunc(sharded_graph)


def pre_shard(args_strategy, params, buffers, named_states, init_helper,
              device):
    device_mesh = get_device_mesh()
    device = mdconfig.easydist_device

    # pre-shard params, buffers, named_states
    params_strategy = args_strategy[:len(params)]
    buffers_strategy = args_strategy[len(params):len(params) + len(buffers)]

    if mdconfig.use_contiguous_buffer:
        contiguous_buf = init_contiguous_buf(params, params_strategy,
                                             device_mesh)

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
            contiguous_buf[index:index +
                           size] = params[param_name]._local_tensor.view(-1)
            params[param_name]._local_tensor = contiguous_buf[
                index:index + size].view(
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
            flat_named_states[i] = sharded_tensor(
                flat_named_states[i],
                args_strategy[state_tensor_num],
                get_device_mesh(),
                materialize_fn=materialize_fn)
            if not mdconfig.use_dtensor:
                flat_named_states[i] = flat_named_states[i]._local_tensor

            state_tensor_num += 1

    named_states = pytree.tree_unflatten(flat_named_states, named_states_spec)

    return params, buffers, named_states
