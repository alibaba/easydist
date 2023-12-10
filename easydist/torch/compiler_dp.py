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
import operator
from functools import partial
from typing import Any, cast
from contextlib import nullcontext

import torch
import torch._custom_ops
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless

import easydist.config as mdconfig
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.passes import tile_comm, runtime_prof, comm_optimize, eliminate_detach
from easydist.torch.passes.sharding import all_reduce_start, all_reduce_end, COMM_FUNCS, CUSTOM_FUNCS
from easydist.torch.utils import (_enable_compile, _rematerialize_optimizer, EDInfo, EDNodeType, create_meta_from_node)
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
    device_mesh = get_device_mesh()
    for node in traced_graph.graph.nodes:
        if node.target == torch.ops.aten._fused_adam.default:
            grad_nodes = node.args[1]
            synced_node = []
            for grad_node in grad_nodes:
                with traced_graph.graph.inserting_before(node):
                    ranks = device_mesh.mesh.flatten().tolist()
                    reduceOp = "avg"
                    all_reduce_start_node = traced_graph.graph.call_function(all_reduce_start,
                                                                         args=(grad_node, reduceOp,
                                                                               ranks))
                    all_reduce_end_node = traced_graph.graph.call_function(
                        all_reduce_end, args=(all_reduce_start_node, reduceOp, ranks))
                    synced_node.append(all_reduce_end_node)

            node.update_arg(1, synced_node)

    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    return traced_graph


def transform_fsdp(traced_graph: torch.fx.GraphModule, shard_param: bool) -> torch.fx.GraphModule:
    device_mesh = get_device_mesh()
    raise NotImplementedError("ZeRO not implement yet.")
    return traced_graph


def annotation_edinfo(traced_graph: torch.fx.GraphModule):
    for node in traced_graph.graph.nodes:
        if not hasattr(node, "ed_info"):
            node.ed_info = EDInfo(ori_meta=node.meta)

        if node.op == 'placeholder':
            node.ed_info.node_type = EDNodeType.AUXILIARY
        elif node.op == 'call_function':
            # create meta for custom function
            if node.target in CUSTOM_FUNCS:
                node.meta = create_meta_from_node(node)
            # annotate node type
            if node.target in COMM_FUNCS:
                node.ed_info.node_type = EDNodeType.COMMUNICATION
            elif node.target == operator.getitem:
                node.ed_info.node_type = EDNodeType.AUXILIARY
            else:
                node.ed_info.node_type = EDNodeType.COMPUTATION
        elif node.op == 'output':
            node.ed_info.node_type = EDNodeType.AUXILIARY

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

    if parallel_mode == "ddp":
        logger.info(f"parallel_mode: {parallel_mode}")
        sharded_graph = transform_ddp(traced_graph)
    elif parallel_mode in ["zero2", "zero3"]:
        logger.info(f"parallel_mode: {parallel_mode}")
        sharded_graph = transform_fsdp(traced_graph, shard_param=(parallel_mode=="zero3"))

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
        sharded_graph = runtime_prof(sharded_graph)
        sharded_graph = tile_comm(sharded_graph)

    if mdconfig.comm_optimization is True:
        sharded_graph = runtime_prof(sharded_graph)
        sharded_graph = comm_optimize(sharded_graph, shape_info, 'rcpsp', grouping=True, mem_restrain=False)

    if mdconfig.log_level <= logging.DEBUG:
        sharded_graph.print_readable()

    class EDCompiledFunc:

        def __init__(self, graph) -> None:
            self.graph = graph

        @torch.no_grad()
        def compiled_func(self, graph, *args, **kwargs):
            nonlocal params, buffers, named_states

            # run the sharded_graph
            params, buffers, named_states, grads, out = graph(
                params, buffers, named_states, args, kwargs)

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
