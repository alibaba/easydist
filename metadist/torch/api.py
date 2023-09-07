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

import rich
import torch
from torch._functorch.aot_autograd import aot_function, make_boxed_compiler

from metadist.metashard import metair
from metadist.autoflow.solver import AutoFlowSolver
import metadist.config as mdconfig
from .sharding_interpreter import MDTorchShardingAnn
from .utils import _sharding_ann_env
from .bridge import torch2meta_graph, get_torch_sharding_strategy
from .passes import eliminate_detach, fix_addmm_bias, fix_convoluation_bias, fix_embedding, sharding_transform
from .passes.sharding import get_device_mesh

logger = logging.getLogger(__name__)

ENABLE_TRANSFORM = False
BW_CONSTRAINTS = None
INPUT_STRATEGY = None


def get_input_strategy():
    global INPUT_STRATEGY
    return INPUT_STRATEGY


def enable_transform():
    global ENABLE_TRANSFORM
    ENABLE_TRANSFORM = True


def _get_output_strategy(opt_strategy, meta_graph, input_strategy):
    partial_strategy = {}
    for op in meta_graph.op_list:
        op_key = op.unique_key()
        if op_key in opt_strategy:
            for idx, var in enumerate(op.outvars):
                if var in meta_graph.output_list:
                    strategy = opt_strategy[op_key]['strategy'].out_strtg_group.get_var_strtg(idx)
                    partial_strategy[var] = strategy

    for var in meta_graph.input_list:
        if var in meta_graph.output_list:
            if var in input_strategy:
                partial_strategy[var] = input_strategy[var]

    return partial_strategy


def _get_input_strategy(opt_strategy, meta_graph):
    partial_strategy = {}
    for op in reversed(meta_graph.op_list):
        op_key = op.unique_key()
        if op_key in opt_strategy:
            for idx, var in enumerate(op.invars):
                if var in meta_graph.input_list:
                    strategy = opt_strategy[op_key]['strategy'].in_strtg_group.get_var_strtg(idx)
                    partial_strategy[var] = strategy

    partial_strategy_list = []

    for var in meta_graph.input_list:
        if var in partial_strategy:
            partial_strategy_list.append(partial_strategy[var])
        else:
            partial_strategy_list.append(
                [metair.SPMD(metair.SPMD.REPLICATE),
                 metair.SPMD(metair.SPMD.REPLICATE)])

    return partial_strategy, partial_strategy_list


@make_boxed_compiler
def metadist_shard(fx_module: torch.fx.GraphModule, inps):

    global BW_CONSTRAINTS, INPUT_STRATEGY

    fx_module = fix_embedding(fx_module)
    fx_module = fix_addmm_bias(fx_module)
    fx_module = fix_convoluation_bias(fx_module)
    fx_module = eliminate_detach(fx_module)
    fx_module.recompile()
    if mdconfig.log_level <= logging.DEBUG:
        print(fx_module.graph)

    with _sharding_ann_env():
        start_t = time.perf_counter()
        sharding_interpreter = MDTorchShardingAnn(fx_module)
        sharding_info, fwd_shape_info = sharding_interpreter.run(*inps)
        logger.info(f"[MDTorchShardingAnn.time]:\t {time.perf_counter() - start_t} s.")
        if mdconfig.log_level <= logging.DEBUG:
            rich.print("sharding_info:\n", sharding_info)
            rich.print("fwd_shape_info:\n", fwd_shape_info)

    meta_graph = torch2meta_graph(fx_module, sharding_info, fwd_shape_info)

    if mdconfig.log_level <= logging.DEBUG:
        rich.print(meta_graph)

    device_mesh = get_device_mesh()
    device_mesh_shape = (device_mesh.size(0), device_mesh.size(1))

    solver = AutoFlowSolver(device_mesh_shape, BW_CONSTRAINTS)
    solver.add_graph(meta_graph)
    start_t = time.perf_counter()
    count_invars = BW_CONSTRAINTS is None
    opt_strategy = solver.ilp_optimize(count_invars)
    logger.info(f"[AutoFlowSolver.time]:\t {time.perf_counter() - start_t} s.")
    # start_t = time.perf_counter()
    # beam_search_strategy = solver.beam_search()
    # logger.info(f"[AutoFlowSolver.beam_search]: {time.perf_counter() - start_t} s.")

    if mdconfig.log_level <= logging.DEBUG:
        rich.print(opt_strategy)

    if BW_CONSTRAINTS is None:
        strategy_map, INPUT_STRATEGY = _get_input_strategy(opt_strategy, meta_graph)
        BW_CONSTRAINTS = _get_output_strategy(opt_strategy, meta_graph, strategy_map)

    if ENABLE_TRANSFORM:
        sharding_strategy = get_torch_sharding_strategy(fx_module, opt_strategy)

        if mdconfig.log_level <= logging.DEBUG:
            print(sharding_strategy)

        fx_module = sharding_transform(fx_module, sharding_strategy)
        fx_module = fix_embedding(fx_module, recover=True)

        if mdconfig.log_level <= logging.DEBUG:
            print(fx_module.graph)

    return fx_module


def md_aot_module(mod: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)
    compiled_f = aot_function(functional_call,
                              num_params_buffers=num_params_buffers,
                              *args,
                              **kwargs)

    class AOTModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            named_params = dict(mod.named_parameters(remove_duplicate=False))
            named_buffers = dict(mod.named_buffers(remove_duplicate=False))
            return compiled_f(
                named_params,
                named_buffers,
                *args,
                **kwargs,
            )

    return AOTModule()


def compile(mod: torch.nn.Module):
    return md_aot_module(mod, fw_compiler=metadist_shard, bw_compiler=metadist_shard)
