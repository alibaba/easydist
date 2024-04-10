from contextlib import nullcontext
from functools import partial
from typing import cast

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn.utils import stateless
from torch.fx.experimental.proxy_tensor import make_fx
from easydist.torch.compile_auto import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.compile_pipeline import (SplitPatcher, compile_pipeline, get_updated_params,
                                                             set_backward_flag, set_step_flag, set_updated_params_states)
from easydist.torch.experimental.pp.microbatch import \
    split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.PipelineStage import PipelineStage
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states
from easydist.torch.init_helper import SetParaInitHelper
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.utils import rgetattr, rsetattr


# TODO @botbw: how to deal with dict?
def _compile_pp(func,
                tracing_mode,
                init_helper,
                input_signature,
                args,
                kwargs,
                schedule_cls,
                args_chunk_spec,
                kwargs_chunk_spec,
                outputs_chunk_spec,
                num_chunks=1,
                strict=True) -> PipelineStage:

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
        params, named_states = get_updated_params()
        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, num_chunks,
                                                             args_chunk_spec, kwargs_chunk_spec)

    with _enable_compile(), SplitPatcher(module, opt):
        set_backward_flag(False)
        set_step_flag(False)
        traced_stateless_func = make_fx(partial(stateless_func, func),
                                           tracing_mode=tracing_mode,
                                           decomposition_table=EASYDIST_DECOMP_TABLE,
                                           _allow_non_fake_inputs=False)(params, buffers,
                                                                         named_states,
                                                                         args_split[0],
                                                                         kwargs_split[0])

    traced_stateless_func.graph.eliminate_dead_code()
    traced_stateless_func = preprocess_traced_graph(traced_stateless_func)
    traced_stateless_func.recompile()

    assert len(list(traced_stateless_func.named_buffers())) == 0, "Make sure there is no tensor created in the forward function"
    traced_stateless_func_node_metas = {
        node.name: node.meta
        for node in traced_stateless_func.graph.nodes
    }
    stateless_func_args = [params, buffers, named_states, args, kwargs]

    compiled_meta, compiled_stages, local_gm, _ = compile_pipeline(traced_stateless_func,
                                                                world_size,
                                                                stateless_func_args,
                                                                init_helper=init_helper,
                                                                strict=strict)

    pipe = PipelineStage(schedule_cls=schedule_cls,
                             local_gm=local_gm,
                             compiled_meta=compiled_meta,
                             stage_idx=rank,
                             compiled_stage=compiled_stages[rank],
                             node_metas=traced_stateless_func_node_metas,
                             num_chunks=num_chunks,
                             args_chunk_spec=args_chunk_spec,
                             kwargs_chunk_spec=kwargs_chunk_spec,
                             returns_chunk_spec=outputs_chunk_spec,
                             device=torch.device(f"cuda:{rank % torch.cuda.device_count()}"))

    return pipe
