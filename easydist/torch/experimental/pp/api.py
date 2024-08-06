from contextlib import nullcontext
from functools import partial
from typing import cast
import warnings

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn.utils import stateless
from torch.fx.experimental.proxy_tensor import make_fx

from easydist.torch.compile import compile_train_step
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (EDGraphModule, SplitPatcher,
                                                             compile_pipeline,
                                                             graph_outputs_to_func_outputs)
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.runtime import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.init_helper import SetParaInitHelper
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.utils import rgetattr, rsetattr

from functools import partial
from typing import Any, cast
from contextlib import nullcontext

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless

import easydist.config as mdconfig
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.runtime import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import SplitPatcher, compile_pipeline
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states
from easydist.torch.init_helper import (SetParaInitHelper)
from easydist.torch.utils import (_enable_compile, _rematerialize_optimizer)
from easydist.utils import rgetattr, rsetattr


def _compile_pp(func,
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
                return_to_all_stages=True,
                accumulate_grads_inplace=True,
                strict=True,
                local=False,
                local_pp_stage_cnt=None) -> PipelineStage:
    args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, num_chunks,
                                                             args_chunk_spec, kwargs_chunk_spec)
    args, kwargs = args_split[0], kwargs_split[0]

    if local_pp_stage_cnt is not None:
        world_size = local_pp_stage_cnt
        rank = 0
    else:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    module, opt = None, None

    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, torch.nn.Module):
            assert module is None, "Only support single nn.Module in args now"
            module = arg
        if isinstance(arg, torch.optim.Optimizer):
            assert opt is None, "Only support single Optimizer in args now"
            opt = arg

    params, buffers, named_states, _, traced_stateless_func = compile_train_step(func, tracing_mode, init_helper, args, kwargs, schedule_cls, module, opt)

    traced_stateless_func_node_metas = {
        node.name: node.meta
        for node in traced_stateless_func.graph.nodes
    }
    stateless_func_args = [params, buffers, named_states, args, kwargs]

    if local:
        assert num_chunks == 1, "num_chunks should be 1 in local mode"
        compiled_meta, compiled_stages, local_gm, _ = compile_pipeline(traced_stateless_func,
                                                                       local_pp_stage_cnt,
                                                                       stateless_func_args,
                                                                       strict=strict)

        class EDCompiledFunc:

            def __init__(self, graph) -> None:
                self.graph = graph

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                self.graph(*args, **kwargs)
                returns = {}
                for stage in compiled_stages:
                    returns.update({
                        node_name: val
                        for node_name, val in stage.outputs.items()
                        if node_name in stage.compiled_meta.returns_nodes_flatten
                    })
                ret = graph_outputs_to_func_outputs(stage.compiled_meta, returns, strict=False)[-1]
                return ret

            def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
                graph(*args, **kwargs)
                returns = {}
                for stage in compiled_stages:
                    returns.update({
                        node_name: val
                        for node_name, val in stage.outputs.items()
                        if node_name in stage.compiled_meta.returns_nodes_flatten
                    })
                ret = graph_outputs_to_func_outputs(stage.compiled_meta, returns, strict=False)[-1]
                return ret

        return EDCompiledFunc(local_gm)

    compiled_meta, compiled_stages, local_gm, _ = compile_pipeline(traced_stateless_func,
                                                                   world_size,
                                                                   stateless_func_args,
                                                                   strict=strict)

    # materialize stage and move states to device
    device = torch.device(f"cuda:{rank}")
    # materialize_fn = init_helper.get_materialize_fn()
    compiled_stage = compiled_stages[rank]
    for gm_type in ['fw_gm', 'bw_gm', 'step_gm']:
        if hasattr(compiled_stage, gm_type):
            gm: EDGraphModule = getattr(compiled_stage, gm_type)
            for state_type in gm.injected_states:
                for name, state in gm.injected_states[state_type].items():
                    gm.injected_states[state_type][name] = state.to(device)

    save_graphviz_dot(local_gm, "local_gm")
    pipe = PipelineStage(schedule_cls=schedule_cls,
                         local_gm=local_gm,
                         compiled_meta=compiled_meta,
                         stage_idx=rank,
                         compiled_stage=compiled_stage,
                         node_metas=traced_stateless_func_node_metas,
                         num_chunks=num_chunks,
                         args_chunk_spec=args_chunk_spec,
                         kwargs_chunk_spec=kwargs_chunk_spec,
                         returns_chunk_spec=outputs_chunk_spec,
                         pp_group=None,
                         device=device,
                         sharded_graph=traced_stateless_func,
                         return_to_all_stages=return_to_all_stages,
                         accumulate_grads_inplace=accumulate_grads_inplace)

    return pipe
