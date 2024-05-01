from contextlib import nullcontext
from functools import partial
from typing import cast

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn.utils import stateless
from torch.fx.experimental.proxy_tensor import make_fx

import easydist.config as mdconfig
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, compile_pipeline, graph_outputs_to_func_outputs)
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.PipelineStage import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.init_helper import SetParaInitHelper, materialize_zero
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.utils import rgetattr, rsetattr

import logging
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
from easydist.torch.experimental.pp.PipelineStage import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import SplitPatcher, compile_pipeline
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states
from easydist.torch.init_helper import (SetParaInitHelper, materialize_zero)
from easydist.torch.passes import (fix_embedding, sharding_transform)
from easydist.torch.device_mesh import get_device_mesh
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
                all_gather_output=True,
                strict=True,
                local=False,
                local_pp_stage_cnt=None) -> PipelineStage:

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

        with mode, _enable_compile():
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

    with _enable_compile(), SplitPatcher(module, opt):
        traced_stateless_func = make_fx(partial(stateless_func, func),
                                           tracing_mode=tracing_mode,
                                           decomposition_table=EASYDIST_DECOMP_TABLE,
                                           _allow_non_fake_inputs=False)(params, buffers,
                                                                         named_states,
                                                                         args_split[0],
                                                                         kwargs_split[0])
    assert len(list(traced_stateless_func.named_buffers())) == 0, "Make sure there is no tensor created in the forward function"
    traced_stateless_func = preprocess_traced_graph(traced_stateless_func)
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
                        for node_name, val in stage.outputs.items() if node_name in stage.compiled_meta.returns_nodes_flatten
                    })
                ret = graph_outputs_to_func_outputs(stage.compiled_meta, returns, strict=False)[-1]
                return ret

            
            def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
                graph(*args, **kwargs)
                returns = {}
                for stage in compiled_stages:
                    returns.update({
                        node_name: val
                        for node_name, val in stage.outputs.items() if node_name in stage.compiled_meta.returns_nodes_flatten
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
    pipe = PipelineStage(
        schedule_cls=schedule_cls,
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
        all_gather_output=all_gather_output
    )

    return pipe