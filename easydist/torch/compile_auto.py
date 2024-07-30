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
from functools import partial, reduce
from typing import Any, Union, cast, Set, Dict, List, Tuple
from contextlib import nullcontext

import rich
import intervaltree
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
from easydist.autoflow.solver import AutoFlowSolver1D
from easydist.torch.bridge import (get_torch_sharding_strategy, to_torch_spmd, torch2meta_graph)
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.runtime import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import SplitPatcher, compile_pipeline
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states, set_backward_flag, set_step_flag, set_updated_params_states
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.init_helper import (SetParaInitHelper, init_contiguous_buf, materialize_zero)
from easydist.torch.passes import (eliminate_detach, fix_addmm_bias, fix_convoluation_bias, decouple_view,
                                   tile_comm, runtime_prof, fix_embedding, fix_meta_device,
                                   sharding_transform, sharding_transform_dtensor,
                                   AllocatorProfiler, ModuleProfilingInfo)
from easydist.torch.device_mesh import get_device_mesh
from easydist.torch.passes import comm_optimize, rule_override_by_graph, create_edinfo
from easydist.torch.passes.fix_node_order import fix_node_order
from easydist.torch.schedule.ilp_memory_scheduler import ILPMemoryScheduler
from easydist.torch.schedule.efficient_memory_scheduler import EfficientMemoryScheduler
from easydist.torch.schedule.graph_mem_plan import GraphMemPlan
from easydist.torch.sharding_interpreter import EDTorchShardingAnn
from easydist.torch.utils import (_enable_compile, _rematerialize_optimizer,
                                  _sharding_ann_env, extract_tensor_meta_info)
from easydist.utils import rgetattr, rsetattr
from easydist.utils.testing import TorchMockDeviceMesh
from easydist.torch.mem_allocation_info import OutVar
import easydist.torch.profiler.stream_tracer as ed_stream_tracer
from easydist.torch.meta_allocator import profiling_allocator
from easydist.torch.schedule.lifetime_info import mem_owner_tracer

# for pickle dump opt_strategy
import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)

sharding_sol = None
sol_rdy_cond = threading.Condition()

mem_sol = None
mem_addr_rdy_cond = threading.Condition()

def preprocess_traced_graph(fx_module: torch.fx.GraphModule):
    fx_module = decouple_view(fx_module)
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

        spmd_mesh = get_device_mesh('spmd')
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory

        opt_strtg_per_dim = []
        for dim, dim_size in enumerate(spmd_mesh.shape):
            # (2) translate fx.GraphModule into MetaGraph
            meta_graph = torch2meta_graph(fx_module, state_tensor_num, sharding_info, shape_info, opt_strtg_per_dim)

            if mdconfig.log_level <= logging.DEBUG:
                rich.print(meta_graph) 
            
            # (3) construct AutoFlowSolver and run ILP
            solver = AutoFlowSolver1D(dim_size, total_memery=total_memory)

            if mdconfig.enable_graph_coarsen:
                logger.info(f"enable graph coarsen with level {mdconfig.coarsen_level}.")
                solver.add_coarsen_graph(meta_graph)
            else:
                solver.add_graph(meta_graph, opt_strategy)

            start_t = time.perf_counter()
            if mdconfig.enable_graph_coarsen:
                opt_strategy_cur_dim = solver.ilp_solve()
            else:
                opt_strategy_cur_dim = solver.ilp_optimize()
            logger.info(f"[AutoFlowSolver.time]:\t {dim} round {time.perf_counter() - start_t} s.")

            opt_strtg_per_dim.append(opt_strategy_cur_dim)

        def reduce_fn(global_strtg, cur_dim_opt_strgt):
            assert set(global_strtg.keys()) == set(cur_dim_opt_strgt.keys()), f"{set(global_strtg.keys()) - set(cur_dim_opt_strgt.keys())}"
            for k in global_strtg.keys():
                for i, in_var_spmd_strtg in enumerate(cur_dim_opt_strgt[k]['strategy'].in_strtg_group):
                    global_strtg[k]['strategy'].in_strtg_group[i] += in_var_spmd_strtg
                for i, out_var_spmd_strtg in enumerate(cur_dim_opt_strgt[k]['strategy'].out_strtg_group):
                    if global_strtg[k]['strategy'].out_strtg_group[i]:
                        global_strtg[k]['strategy'].out_strtg_group[i] += out_var_spmd_strtg
            return global_strtg

        opt_strategy = reduce(reduce_fn, opt_strtg_per_dim)
        if mdconfig.log_level <= logging.DEBUG:
            rich.print(opt_strategy)

        sharding_strategy = get_torch_sharding_strategy(fx_module, opt_strategy)

        if mdconfig.log_level <= logging.DEBUG:
            rich.print(sharding_strategy)

        args_strategy = meta_graph.get_input_strategy(opt_strategy, spmd_mesh.mesh.shape)
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

@torch.fx.has_side_effect
def start_customized_allocator():
    profiling_allocator._set_customized_flag(True)
    return None

@torch.fx.has_side_effect
def stop_customized_allocator():
    profiling_allocator._set_customized_flag(False)
    return None

def insert_customized_allocator_flag(
                              gm: torch.fx.GraphModule,
                              back_alloced_nodes: Set[str]):
    start_before_nodes = []
    stop_before_nodes = []
    customized_alloc_active = False
    for node in gm.graph.nodes:
        if node.op == 'placeholder' or node.op == 'get_attr' or node.op == 'output':
            if customized_alloc_active:
                stop_before_nodes.append(node)
                customized_alloc_active = False
        elif node.name in back_alloced_nodes:
            if customized_alloc_active:
                stop_before_nodes.append(node)
                customized_alloc_active = False
        else:
            if not customized_alloc_active:
                start_before_nodes.append(node)
                customized_alloc_active = True

    #print(f"start: {start_before_nodes}")
    #print(f"stop: {stop_before_nodes}")

    for start_nd in start_before_nodes:
        with gm.graph.inserting_before(start_nd):
            start_flag_nd = gm.graph.call_function(start_customized_allocator)
            #print(f"add start flag: {start_flag_nd}")

    for stop_nd in stop_before_nodes:
        with gm.graph.inserting_before(stop_nd):
            stop_flag_nd = gm.graph.call_function(stop_customized_allocator)
            #print(f"add stop flag: {stop_flag_nd}")

    #print("updated graph node list:")
    #for node in gm.graph.nodes:
    #    print(node)

@torch.fx.has_side_effect
def op_mem_checker(op_name: str, args, arg_mem_owner, new_mem_owner):
    assert mem_owner_tracer != None
    profiling_allocator._set_cur_op_name(op_name)
    print(f"cur node: {op_name}")
    for arg, arg_name in args:
        if isinstance(arg, torch.Tensor):
            #print(f"  arg: {arg_name}, type: {type(arg)}, ptr: {arg.data_ptr}, shape: {arg.shape}, size: {arg.element_size() * arg.numel()}")
            for iv in arg_mem_owner:
                expected_op_name = iv.data
                for overlap_iv in mem_owner_tracer.overlap(iv):
                    if overlap_iv.data != expected_op_name:
                        print(f"lifetimes overlap between {arg_name} and {overlap_iv.data}\n"
                              f"op {op_name} expects range {iv} was written by {arg_name},"
                              f" but it actually was written by {overlap_iv.data}")

    # update mem_owner_tracer
    if new_mem_owner == None:
        return None

    for new_mem_iv in new_mem_owner:
        overlap_ivs = mem_owner_tracer.overlap(new_mem_iv.begin, new_mem_iv.end)
        for overlap_iv in overlap_ivs:
            if overlap_iv.begin < new_mem_iv.begin:
                mem_owner_tracer[overlap_iv.begin:new_mem_iv.begin] = overlap_iv.data
            if overlap_iv.end > new_mem_iv.end:
                mem_owner_tracer[new_mem_iv.end:overlap_iv.end] = overlap_iv.data

        # debug only
        new_overlap_ivs = mem_owner_tracer.overlap(new_mem_iv.begin, new_mem_iv.end)
        assert overlap_ivs == new_overlap_ivs
        mem_owner_tracer.remove_overlap(new_mem_iv.begin, new_mem_iv.end)
        mem_owner_tracer[new_mem_iv.begin:new_mem_iv.end] = new_mem_iv.data

    return None

def insert_op_mem_checker(gm: torch.fx.GraphModule,
                          inter_op_mems: Dict[str, List[Tuple[int, OutVar]]]):
    nodes = [node for node in gm.graph.nodes]
    for node in nodes:
        if node.op == 'call_function':
            node_args_flatten = pytree.tree_flatten(node.args)[0]
            invars = [(arg, arg.name) for arg in node_args_flatten if isinstance(arg, torch.fx.Node)]
            #print(f"invars of node {node}: {invars}")
        else:
            invars = []

        arg_mem_owner = intervaltree.IntervalTree()
        for invar, invar_name in invars:
            if invar_name not in inter_op_mems:
                continue
            invar_addrs = inter_op_mems[invar_name]
            # check if the data in addresses are generated by invar op
            for invar_addr in invar_addrs:
                if invar_addr == None:
                    # ignore context memory
                    continue
                mem_start = invar_addr[0]
                mem_stop = invar_addr[0] + invar_addr[1].size()
                assert arg_mem_owner.overlap(mem_start, mem_stop) == set()
                arg_mem_owner[mem_start:mem_stop] = invar_name

        if node.name in inter_op_mems:
            new_mem_owner = intervaltree.IntervalTree()
            outvar_addrs = inter_op_mems[node.name]
            for outvar_addr in outvar_addrs:
                if outvar_addr == None:
                    # ignore context memory
                    continue
                mem_start = outvar_addr[0]
                mem_stop = outvar_addr[0] + outvar_addr[1].size()
                new_mem_owner[mem_start:mem_stop] = node.name
        else:
            new_mem_owner = None

        with gm.graph.inserting_before(node):
            checker = gm.graph.call_function(
                        op_mem_checker,
                        #args=(node.name, invars, arg_mem_owner, mem_owner_tracer,
                        args=(node.name, invars, arg_mem_owner,
                              new_mem_owner))
            if mdconfig.log_level <= logging.DEBUG:
                print(f"add op mem checker: {checker}")

def memory_opt(gm: torch.fx.GraphModule):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    rpc.init_rpc(f"ed_worker{rank}", rank=rank, world_size=world_size)
    if rank == 0:
        logging.info("profiling fx module's memory...")

    # setting allocator to profiling mode
    profiling_allocator._set_allocator_mode(profiling_allocator.AllocatorMode.PROFILE)

    # save all profiling information in this dict
    profiling_info = ModuleProfilingInfo(rank)
    alloc_profiler = AllocatorProfiler(gm, profiling_info)
    with ed_stream_tracer.StreamTracer() as stream_tracer:
        _ = alloc_profiler.run([])
        trace_data = stream_tracer.get_stream_trace_data()

    if mdconfig.log_level <= logging.DEBUG:
        print(f"py op_streams:\n{trace_data.op_streams}")
        print(f"py op_extra_streams:\n{trace_data.op_extra_streams}")

    if rank == 0:
        logging.info("finish profiling fx module's memory")
        graph_mem_info = alloc_profiler.create_graph_mem_info()
        #print(f"graph_mem_info:\n{str(graph_mem_info)}")

        op_streams = {}
        for op_name, streams in trace_data.op_streams.items():
            op_streams[op_name] = streams[0]

        for op_name, streams in trace_data.op_extra_streams.items():
            if op_name not in op_streams:
                op_streams[op_name] = streams[0]

        if mdconfig.mem_opt_by_solver:
            mem_sched = ILPMemoryScheduler(gm, graph_mem_info,
                                           1024*128, op_streams)
        else:
            mem_sched = EfficientMemoryScheduler(gm, graph_mem_info, op_streams)

        required_memory, temp_memory, schedules, ordered_schedules, mem_alloc_info, inter_op_mems, back_alloced_nodes = \
                                            mem_sched.gen_mem_addresses()
        #print("ordered_schedules:")
        #for idx, nd in enumerate(ordered_schedules):
        #    print(f"{idx}: {nd}")

        #print(f"master proposes required_memory: {required_memory}")
        #print(f"master creates mem locations:\n{inter_op_mems}")

        with mem_addr_rdy_cond:
            global mem_sol
            mem_sol = [
                required_memory, temp_memory, ordered_schedules, mem_alloc_info, inter_op_mems, back_alloced_nodes
            ]
            mem_addr_rdy_cond.notify_all()

        assert len(ordered_schedules) == len(mem_sched.nodes_to_schedule), \
            f"schedule {len(ordered_schedules)} nodes, but totally {len(mem_sched.nodes_to_schedule)} nodes"
    else:
        required_memory, temp_memory, ordered_schedules, mem_alloc_info, inter_op_mems, back_alloced_nodes = rpc.rpc_sync(
                            "ed_worker0", fetch_mem_sol, args=(), timeout=0)
        #print(f"worker {rank} receives required_memory: {required_memory}")
        #print(f"worker {rank} receives mem locations:\n{inter_op_mems}")

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

    #print(f"graph details:\n{str(gm.graph)}\nend of graph\n")

    insert_customized_allocator_flag(gm, back_alloced_nodes)
    if mdconfig.enable_runtime_trace:
        insert_op_mem_checker(gm, inter_op_mems)
        def transform_wrapper(code):
            return ["from intervaltree import IntervalTree, Interval\n",
                    *code]
        gm.graph.on_generate_code(lambda _: transform_wrapper)

    gm.recompile()
    if mdconfig.log_level <= logging.DEBUG:
        print(f"python codes\n{gm.code}")


def _compile_auto(func,
                  tracing_mode,
                  init_helper,
                  input_signature,
                  args,
                  kwargs,
                  schedule_cls=None,
                  args_chunk_spec=None,
                  kwargs_chunk_spec=None,
                  outputs_chunk_spec=None,
                  num_chunks=1,
                  return_to_all_stages=True,
                  accumulate_grads_inplace=True,
                  strict=True) -> Union[PipelineStage, Any]:
    args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, num_chunks,
                                                             args_chunk_spec, kwargs_chunk_spec)
    args, kwargs = args_split[0], kwargs_split[0]
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

        #print(f"initial params: {params}")
        #print(f"initial buffers: {buffers}")
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

    with _enable_compile(), SplitPatcher(module, opt) if schedule_cls else nullcontext():
        traced_graph = make_fx(partial(stateless_func, func),
                               tracing_mode=tracing_mode,
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, named_states, args,
                                                             kwargs)

    assert len(list(traced_graph.buffers())) == 0, f"{set(traced_graph.named_buffers().keys())}"
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
    master_address = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    rpc.init_rpc(f"ed_worker{rank}", rank=rank, world_size=world_size,rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{master_address}:{master_port}",
            _transports=["uv", "shm"],
            _channels=["basic"]
    ))
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

    if mdconfig.use_dtensor:
        sharded_gm = sharding_transform_dtensor(traced_graph, sharding_strategy)
    else:
        sharded_gm = sharding_transform(traced_graph, opt_strategy, state_io_map)
        if mdconfig.enable_tile_comm:
            sharded_gm = runtime_prof(sharded_gm, tiling_prof=True)
            sharded_gm = tile_comm(sharded_gm)

    save_graphviz_dot(sharded_gm, f'sharded_graph_raw_{rank}')
    sharded_gm = fix_embedding(sharded_gm, recover=True)

    if not mdconfig.use_dtensor:
        if schedule_cls is None and mdconfig.comm_optimization is True:
            sharded_gm = runtime_prof(sharded_gm)
            sharded_gm = comm_optimize(sharded_gm, 'rcpsp', grouping=True, mem_restrain=False)

        # override pytorch dtensor propagate rules to optimize dispater behavior
        if mdconfig.override_dtensor_rule is True:
            sharded_gm = rule_override_by_graph(sharded_gm, opt_strategy, shape_info)


    if mdconfig.log_level <= logging.DEBUG:
        sharded_gm.print_readable()

    if mdconfig.dump_fx_graph:
        print(f"node num in sharded graph: {len(sharded_gm.graph.nodes)}")
        drawer = FxGraphDrawer(sharded_gm, f"shard_fx-{rank}", ignore_getattr=True)
        dot_graphs = drawer.get_all_dot_graphs()
        for name, dot_graph in dot_graphs.items():
            dot_graph.write_jpg(f"./tmp/{name}.jpg")
            dot_graph.write_raw(f"./tmp/{name}.txt")
        if mdconfig.log_level <= logging.DEBUG:
            print(f"sharded_gm._code: {sharded_gm._code}")

    spmd_mesh = get_device_mesh('spmd')
    device = mdconfig.easydist_device

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
        contiguous_buf = init_contiguous_buf(params, params_strategy, spmd_mesh)

    index = 0
    for idx, param_name in enumerate(params):
        materialize_fn = init_helper.get_materialize_fn()
        materialize_fn = partial(materialize_fn,
                                param_buf_key=param_name,
                                materialization_device=device)
        params[param_name] = sharded_tensor(params[param_name],
                                            params_strategy[idx],
                                            spmd_mesh,
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
                                            spmd_mesh,
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
                                                spmd_mesh,
                                                materialize_fn=materialize_fn)
            if not mdconfig.use_dtensor:
                flat_named_states[i] = flat_named_states[i]._local_tensor

            state_tensor_num += 1

    named_states = pytree.tree_unflatten(flat_named_states, named_states_spec)

    if schedule_cls is not None:
        pp_mesh = get_device_mesh('pp')
        pp_rank, pp_size = pp_mesh.get_coordinate()[0], pp_mesh.size()
        traced_graph_node_metas = {
            node.name: node.meta
            for node in traced_graph.graph.nodes
        }
        sharded_gm = fix_node_order(sharded_gm)
        stateless_func_args = (params, buffers, named_states, args, kwargs)
        save_graphviz_dot(sharded_gm, 'sharded_graph')
        pp_compiled_meta, pp_compiled_stages, pp_local_gm, _ = compile_pipeline(
            sharded_gm,
            pp_size,
            stateless_func_args,
            phs_stragegies=args_strategy,
            strict=strict)
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
            pp_group=pp_mesh.get_group('pp'),
            device=torch.device(f"cuda:{rank}"),
            sharded_graph=sharded_gm,
            return_to_all_stages=return_to_all_stages,
            accumulate_grads_inplace=accumulate_grads_inplace
        )
        return pipe

    if mdconfig.enable_memory_opt:
        memory_opt(sharded_gm)

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
                                                     spmd_mesh,
                                                     materialize_fn=materialize_fn)
                    if not mdconfig.use_dtensor:
                        flatten_args[i] = flatten_args[i]._local_tensor
                    args_strategy_idx += 1

            args, kwargs = pytree.tree_unflatten(flatten_args, args_specs)

            # run the sharded_gm
            if mdconfig.enable_memory_opt:
                profiling_allocator._set_allocator_mode(profiling_allocator.AllocatorMode.RUNTIME)
                params, buffers, named_states, grads, sharded_out = graph(
                    params, buffers, named_states, args, kwargs)
            else:
                params, buffers, named_states, grads, sharded_out = graph(
                    params, buffers, named_states, args, kwargs)

            for para_name in params:
                params[para_name].grad = grads[para_name]

            # out from DTensor to Tensor
            local_out = pytree.tree_map(dtensor_to_tensor, sharded_out)

            return local_out

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if mem_owner_tracer:
                mem_owner_tracer.clear()

            return self.compiled_func(self.graph, *args, **kwargs)

        def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
            if mem_owner_tracer:
                mem_owner_tracer.clear()

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

    return EDCompiledFunc(sharded_gm)

