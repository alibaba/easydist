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

import time
import logging
from functools import partial
from typing import Any, cast
from contextlib import nullcontext

import rich
import torch
import torch._custom_ops
import torch.utils._pytree as pytree
from torch.fx._pytree import tree_flatten_spec
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils import stateless

import easydist.config as mdconfig
from easydist.torch.sharding_interpreter import EDTorchShardingAnn
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.passes import (tile_comm, runtime_prof, comm_optimize, eliminate_detach,
                                   annotation_edinfo, create_edinfo)
from easydist.torch.passes.sharding import (all_reduce_start, all_reduce_end, scatter_wrapper,
                                            all_gather_start, all_gather_end, reduce_scatter_start,
                                            reduce_scatter_end)
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer, _sharding_ann_env
from easydist.torch.passes.process_tag import process_tag
from easydist.utils import rgetattr, rsetattr
from easydist.torch.device_mesh import get_device_mesh

logger = logging.getLogger(__name__)


def get_shape_info(node_output):
    if isinstance(node_output, torch.Tensor) or isinstance(node_output,
                                                           torch.nn.parameter.Parameter):
        return {"shape": node_output.shape, "dtype": node_output.dtype}
    else:
        return {}


def transform_ddp(traced_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # (TODO) only support _fused_adam here
    dp_mesh = get_device_mesh('dp')
    for node in traced_graph.graph.nodes:
        if node.target == torch.ops.aten._fused_adam.default:
            grad_nodes = node.args[1]
            synced_node = []
            for grad_node in grad_nodes:
                with traced_graph.graph.inserting_before(node):
                    ranks = dp_mesh.mesh.flatten().tolist()
                    reduceOp = "avg"
                    all_reduce_start_node = traced_graph.graph.call_function(all_reduce_start,
                                                                             args=(grad_node,
                                                                                   reduceOp,
                                                                                   ranks))
                    all_reduce_end_node = traced_graph.graph.call_function(
                        all_reduce_end, args=(all_reduce_start_node, reduceOp, ranks))
                    synced_node.append(all_reduce_end_node)

            node.update_arg(1, synced_node)

    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    return traced_graph


def transform_fsdp(traced_graph: torch.fx.GraphModule, shard_param: bool) -> torch.fx.GraphModule:
    # (TODO) only support _fused_adam here
    dp_mesh = get_device_mesh('dp')

    dp_rank = dp_mesh.get_coordinate()[0]
    dp_size = dp_mesh.size(0)

    opt_state_nodes = []

    # recode the parameter node
    if shard_param is True:
        param_nodes_map = {}
        for node in traced_graph.graph.nodes:
            if node.target == torch.ops.aten._fused_adam.default:
                for para_node in node.args[0]:
                    param_nodes_map[para_node.name] = para_node

        ranks = dp_mesh.mesh.flatten().tolist()
        for node in traced_graph.graph.nodes:
            if node.op == 'call_function':
                # only insert all-gather for the backward and forward
                if node.target == torch.ops.aten._fused_adam.default:
                    break
                flatten_args = pytree.tree_flatten(node.args)[0] + pytree.tree_flatten(
                    node.kwargs)[0]
                param_arg_nodes = [
                    i for i in flatten_args
                    if isinstance(i, torch.fx.Node) and i.name in param_nodes_map
                ]
                for param_node in param_arg_nodes:
                    with traced_graph.graph.inserting_before(node):
                        gather_dim = 0
                        all_gather_start_node = traced_graph.graph.call_function(all_gather_start,
                                                                                 args=(param_node,
                                                                                       gather_dim,
                                                                                       ranks))
                        all_gather_end_node = traced_graph.graph.call_function(
                            all_gather_end, args=(all_gather_start_node, gather_dim, ranks))
                        view_node = traced_graph.graph.call_function(
                            torch.ops.aten.view,
                            args=(all_gather_end_node, param_node.meta['val'].shape))
                    node.replace_input_with(param_node, view_node)

    for node in traced_graph.graph.nodes:
        if node.target == torch.ops.aten._fused_adam.default:
            # scatter the param when zero-2, which we don't shard the parameters
            if shard_param is False:
                param_nodes = node.args[0]
                scattered_nodes = []
                for param_node in param_nodes:
                    with traced_graph.graph.inserting_before(node):
                        scatter_dim = 0
                        flatten_node = traced_graph.graph.call_function(torch.ops.aten.flatten,
                                                                        args=(param_node, ))
                        scatter_node = traced_graph.graph.call_function(scatter_wrapper,
                                                                        args=(flatten_node,
                                                                              dp_size, 0, dp_rank))
                        scattered_nodes.append(scatter_node)

                node.update_arg(0, scattered_nodes)

            ranks = dp_mesh.mesh.flatten().tolist()

            # reduce_scatter the gradient
            grad_nodes = node.args[1]
            synced_nodes = []
            for grad_node in grad_nodes:
                with traced_graph.graph.inserting_before(node):
                    reduceOp = "avg"
                    scatter_dim = 0
                    flatten_node = traced_graph.graph.call_function(torch.ops.aten.flatten,
                                                                    args=(grad_node, ))
                    reduce_scatter_start_node = traced_graph.graph.call_function(
                        reduce_scatter_start, args=(flatten_node, reduceOp, scatter_dim, ranks))
                    reduce_scatter_end_node = traced_graph.graph.call_function(
                        reduce_scatter_end,
                        args=(reduce_scatter_start_node, reduceOp, scatter_dim, ranks))
                    synced_nodes.append(reduce_scatter_end_node)

            node.update_arg(1, synced_nodes)

            opt_state_nodes = opt_state_nodes + node.args[2] + node.args[3]

            if shard_param is False:
                # all_gather the first list of output.
                for used_node in node.users.keys():
                    # the param output list
                    if used_node.args[1] == 0:
                        for out_param_node in used_node.users.keys():
                            with traced_graph.graph.inserting_after(out_param_node):
                                all_gather_start_node = traced_graph.graph.call_function(
                                    all_gather_start, args=(out_param_node, 0, ranks))
                            with traced_graph.graph.inserting_after(all_gather_start_node):
                                all_gather_end_node = traced_graph.graph.call_function(
                                    all_gather_end, args=(all_gather_start_node, 0, ranks))
                            with traced_graph.graph.inserting_after(all_gather_end_node):
                                view_node = traced_graph.graph.call_function(
                                    torch.ops.aten.view,
                                    args=(all_gather_end_node, out_param_node.meta['val'].shape))

                                out_param_node.replace_all_uses_with(view_node)
                                all_gather_start_node.update_arg(0, out_param_node)

    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    # fix the args meta
    if shard_param is True:
        for param_node in param_nodes_map.values():
            fake_val = scatter_wrapper(param_node.meta['val'].flatten(), dp_size, 0, dp_rank)
            param_node.meta = {'val': fake_val, 'tensor_meta': _extract_tensor_metadata(fake_val)}

    for opt_state_node in opt_state_nodes:
        fake_val = scatter_wrapper(opt_state_node.meta['val'].flatten(), dp_size, 0, dp_rank)
        opt_state_node.meta = {'val': fake_val, 'tensor_meta': _extract_tensor_metadata(fake_val)}

    return traced_graph


def _compile_dp(func, parallel_mode, tracing_mode, args, kwargs):
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
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, named_states, args,
                                                             kwargs)

    traced_graph = eliminate_detach(traced_graph)
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    traced_graph = process_tag(traced_graph)

    shard_opt_state = False
    shard_param = False

    # pre shard the parameter ot optimizer state
    device_mesh = get_device_mesh()
    assert "dp" in device_mesh.mesh_dim_names
    device_mesh = device_mesh["dp"]
    dp_rank = device_mesh.get_coordinate()[0]
    dp_size = device_mesh.size(0)

    if mdconfig.enable_tile_comm:
        with _sharding_ann_env():
            start_t = time.perf_counter()
            sharding_interpreter = EDTorchShardingAnn(traced_graph)
            flatten_args = tree_flatten_spec(
                [params, buffers, named_states, args, kwargs], traced_graph._in_spec)
            sharding_info, shape_info = sharding_interpreter.run(*flatten_args)
            logger.info(f"[EDTorchShardingAnn.time]:\t {time.perf_counter() - start_t} s.")
            if mdconfig.log_level <= logging.DEBUG:
                rich.print("sharding_info:\n", sharding_info)
                rich.print("shape_info:\n", shape_info)

        traced_graph = create_edinfo(traced_graph, sharding_info, shape_info)

    if dp_size > 1:
        if parallel_mode == "ddp":
            logger.info(f"parallel_mode: {parallel_mode}")
            sharded_graph = transform_ddp(traced_graph)
        elif parallel_mode in ["zero2", "zero3"]:
            shard_opt_state = True
            shard_param = (parallel_mode == "zero3")
            logger.info(f"parallel_mode: {parallel_mode}")
            sharded_graph = transform_fsdp(traced_graph, shard_param=shard_param)
    else:
        sharded_graph = traced_graph

    sharded_graph = annotation_edinfo(sharded_graph)

    if mdconfig.log_level <= logging.DEBUG:
        sharded_graph.print_readable()

    shape_info = {}
    for node in sharded_graph.graph.nodes:
        if hasattr(node, "meta") and 'val' in node.meta:
            shape_info[node.name] = pytree.tree_map(get_shape_info, node.meta['val'])
        else:
            shape_info[node.name] = {}

    if mdconfig.enable_tile_comm:
        sharded_graph = runtime_prof(sharded_graph, tiling_prof=True)
        sharded_graph = tile_comm(sharded_graph)

    if mdconfig.comm_optimization is True:
        sharded_graph = runtime_prof(sharded_graph)
        sharded_graph = comm_optimize(sharded_graph,
                                      'rcpsp',
                                      grouping=True,
                                      mem_restrain=False)

    if mdconfig.log_level <= logging.DEBUG:
        sharded_graph.print_readable()

    # (TODO) only support Adam here
    if shard_param is True:
        for param_name, param_var in params.items():
            params[param_name] = scatter_wrapper(param_var.detach().flatten(), dp_size, 0, dp_rank)

    # (TODO) only support Adam here
    if shard_opt_state is True:
        for param_name, param_opt_var in named_states.items():
            for opt_var_name, opt_var in param_opt_var.items():
                if opt_var_name != "step":
                    named_states[param_name][opt_var_name] = scatter_wrapper(
                        opt_var.flatten(), dp_size, 0, dp_rank)

    class EDCompiledFunc:

        def __init__(self, graph) -> None:
            self.graph = graph

        @torch.no_grad()
        def compiled_func(self, graph, *args, **kwargs):
            nonlocal params, buffers, named_states

            # run the sharded_graph
            params, buffers, named_states, grads, out = graph(params, buffers, named_states, args,
                                                              kwargs)

            for para_name in params:
                params[para_name].grad = grads[para_name]

            return out

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.compiled_func(self.graph, *args, **kwargs)

        def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
            return self.compiled_func(graph, *args, **kwargs)

        def get_state(self):
            return params, buffers, named_states

        def parameters(self):
            return params.values()

        def named_parameters(self):
            return params

    return EDCompiledFunc(sharded_graph)
