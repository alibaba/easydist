import warnings
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.split_utils import SplitPatcher
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import make_fx
from functools import partial
from easydist.torch.experimental.pp.split_utils import clear_pp_compile_states, get_updated_params_states
from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.torch.init_helper import SetParaInitHelper
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer


import torch
from torch.nn.utils import stateless


from contextlib import nullcontext
from typing import cast

from easydist.utils import rgetattr, rsetattr


def stateless_func(func, module, opt, params, buffers, named_states, args, kwargs):
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


def ed_compile_func(func, tracing_mode, init_helper, args, kwargs, schedule_cls, module, opt):
    params, buffers = {}, {}
    if module is not None:
        params = dict(module.named_parameters())
        buffers = dict(module.named_buffers())

        if isinstance(init_helper, SetParaInitHelper):
            init_helper.module = module

    named_states = {}

    if opt is not None:
        # assign grad and warm up optimizer
        for name in dict(module.named_parameters()):
            with torch.no_grad():
                rsetattr(module, name + ".grad", torch.zeros_like(rgetattr(module, name).data))
                if isinstance(rgetattr(module, name).data, FakeTensor):
                    mode = rgetattr(module, name).data.fake_mode

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

    with _enable_compile(), SplitPatcher(module, opt) if schedule_cls else nullcontext():
        traced_graph = make_fx(partial(stateless_func, func, module, opt),
                               tracing_mode=tracing_mode,
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, named_states, args,
                                                             kwargs)

    if len(list(traced_graph.named_buffers())) != 0:
        warnings.warn("No buffer should be found in the traced graph, please check if the model is correctly traced")

    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    save_graphviz_dot(traced_graph, 'traced_graph')

    return params,buffers,named_states,state_tensor_num,traced_graph