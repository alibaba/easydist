import copy
import functools
import inspect
import os
from contextlib import nullcontext
from functools import partial
from typing import Any, cast
# make easydist happy without torchrun
os.environ['MASTER_PORT'] = '-1'

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.autograd import Function
from torch.fx.experimental.proxy_tensor import (PreDispatchTorchFunctionMode,
                                                ProxyTorchDispatchMode,
                                                PythonKeyTracer, decompose,
                                                disable_autocast_cache,
                                                disable_proxy_modes_tracing,
                                                dispatch_trace,
                                                enable_pre_dispatch,
                                                enable_python_dispatcher,
                                                fake_signature, wrap_key)
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn.utils import stateless

from easydist.torch.compiler import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp.IR import (_from_traced,
                                               get_pipeline_tracer, pipe_split,
                                               set_pipeline_tracer)
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer

# for pickle dump opt_strategy


def my_make_fx(f,
               decomposition_table=None,
               tracing_mode="real",
               _allow_non_fake_inputs=False,
               *,
               pre_dispatch=False,
               _allow_fake_constant=False):
    assert tracing_mode in ["real", "fake", "symbolic"]

    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        # type: ignore[attr-defined]
        phs = pytree.tree_map(lambda _: fx.PH, args)
        fx_tracer = PythonKeyTracer()
        fake_tensor_mode: Any = nullcontext()
        if tracing_mode == "real":
            fake_tensor_mode = nullcontext()
        elif tracing_mode == "fake":
            import torch._dynamo
            fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
            if fake_tensor_mode is None:
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True,
                                                  allow_non_fake_inputs=_allow_non_fake_inputs)
        elif tracing_mode == "symbolic":
            import torch._dynamo
            fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
            if fake_tensor_mode is None:
                shape_env = ShapeEnv()
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False,
                                                  allow_non_fake_inputs=_allow_non_fake_inputs,
                                                  shape_env=shape_env)
            else:
                shape_env = fake_tensor_mode.shape_env
                assert shape_env is not None, "shape_env should be set if tracing with 'symbolic'"

        else:
            raise AssertionError(f"Unexpected tracing type: {tracing_mode}")

        python_dispatcher_mode: Any = nullcontext()
        pre_dispatch_mode: Any = nullcontext()
        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if tracing_mode == "symbolic" or pre_dispatch:
            python_dispatcher_mode = enable_python_dispatcher()
        if pre_dispatch:
            pre_dispatch_mode = enable_pre_dispatch()

        proxy_function_mode: Any = nullcontext()
        if pre_dispatch:
            proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        proxy_mode = ProxyTorchDispatchMode(fx_tracer,
                                            tracing_mode,
                                            pre_dispatch=pre_dispatch,
                                            _allow_fake_constant=_allow_fake_constant)

        arg_count = 0

        def wrap_fake(x):
            nonlocal arg_count
            if isinstance(x, torch.Tensor):
                # TODO: it would be nice to line these up with the names
                # FX will choose for the placeholders, but we don't
                # actually know what the names will be at this point yet
                # NB: the Source here is actually meaningless
                from torch._dynamo.source import ConstantSource
                source = ConstantSource(f"input{arg_count}")
                arg_count += 1
                # type: ignore[attr-defined]
                return fake_tensor_mode.from_tensor(x, source=source)

            return x

        sym_mode = proxy_mode.sym_mode

        wrap_fn_map = {
            "real": lambda x: x,
            "fake": wrap_fake,
            "symbolic": wrap_fake,
        }
        args = pytree.tree_map(wrap_fn_map[tracing_mode], args)

        if not hasattr(inspect.unwrap(f),
                       '__code__') or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS:
            # FX doesn't support varargs, so we gotta fake up a wrapper
            # TODO: Would be nice to fix this at the source...
            func = fake_signature(f, len(phs))
        else:
            func = f

        # We disable the autocast cache as the autocast cache causes type conversions on parameters to
        # check a cache, which introduces untracked tensors into the graph
        #
        # We also disable tracing by any other tensor proxy-based tracers except the current. The
        # purpose of `make_fx` is to produce graphmodules as a side effect; its internal execution is
        # thus irrelevant to any external functional trace.
        with decompose(decomposition_table), fake_tensor_mode, python_dispatcher_mode, pre_dispatch_mode, proxy_function_mode, \
                sym_mode, proxy_mode, disable_autocast_cache(), disable_proxy_modes_tracing(enable_current=True):
            old_tracer = get_pipeline_tracer()
            set_pipeline_tracer(fx_tracer)
            try:
                t = dispatch_trace(wrap_key(func, args, fx_tracer, pre_dispatch),
                                   tracer=fx_tracer,
                                   concrete_args=tuple(phs))
            finally:
                set_pipeline_tracer(old_tracer)

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if tracing_mode == "symbolic":
            t.shape_env = shape_env  # type: ignore[assignment]
        return t

    return wrapped


class SplitFunc(Function):

    @staticmethod
    def forward(ctx, *input):
        pipe_split()
        if len(input) == 1:
            return input[0]
        return input

    @staticmethod
    def backward(ctx, *grad_output):
        pipe_split()
        if len(grad_output) == 1:
            return grad_output[0]
        return grad_output


def split_func(*input):
    return SplitFunc.apply(*input)


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm([1024])
        self.linear1_0 = torch.nn.Linear(1024, 256)
        self.linear1_1 = torch.nn.Linear(256, 128)
        self.linear1_2 = torch.nn.Linear(128, 10)

        self.norm0 = torch.nn.BatchNorm1d(1024)
        self.linear0_0 = torch.nn.Linear(1024, 512)
        self.linear0_1 = torch.nn.Linear(512, 256)
        self.linear0_2 = torch.nn.Linear(256, 128)
        self.linear0_3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x0 = self.norm0(x)
        x0 = self.linear0_0(x0)
        x0 = self.linear0_1(x0)
        x0 = self.linear0_2(x0)
        x0 = self.linear0_3(x0)

        x, x0 = split_func(x, x0)

        x1 = self.norm1(x)
        x1 = self.linear1_0(x1)
        x1 = self.linear1_1(x1)
        x1 = self.linear1_2(x1)

        return (x0 + x1).relu()


if __name__ == '__main__':
    model = Foo().cuda().train().double()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    model_copy = copy.deepcopy(model)
    optim_copy = torch.optim.SGD(model_copy.parameters(), lr=0.1)
    args = (torch.rand(16, 1024, dtype=torch.double).cuda(), model, optim)
    kwargs = {}
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def train_step(input, model, opt):
        out = model(input).mean()
        pipe_split()
        out.backward()
        pipe_split()
        opt.step()
        opt.zero_grad()
        return out

    named_states = {}
    flat_named_states, named_states_spec = pytree.tree_flatten(named_states)

    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(
                cast(torch.nn.Module, model), {
                    **params,
                    **buffers
                }, tie_weights=True) if model else nullcontext(), _rematerialize_optimizer(
                    optim, named_states, params) if optim else nullcontext():
            ret = func(*args, **kwargs)

        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

    with _enable_compile():
        traced_graph = my_make_fx(partial(stateless_func, train_step),
                                  tracing_mode='fake',
                                  decomposition_table=EASYDIST_DECOMP_TABLE,
                                  _allow_non_fake_inputs=False)(params, buffers, named_states,
                                                                args, kwargs)

    splited = _from_traced(model, traced_graph, None)
    for name, submod in splited.named_children():
        submod.graph.eliminate_dead_code()
        submod = preprocess_traced_graph(submod)
        submod.recompile()
        setattr(splited, name, submod)
    print(traced_graph.code)
    print(splited.code)

    model.eval()
    params_detached = {k: v.detach() for k, v in params.items()}
    buffers_detached = {k: v.detach() for k, v in buffers.items()}
    flatten_args, _ = pytree.tree_flatten(
        (params_detached, buffers_detached, named_states, args, kwargs))
    out = splited(*flatten_args)
    out_copy = model_copy(args[0]).mean()
    out_copy.backward()
    optim_copy.step()

    assert torch.allclose(out[-1], out_copy)
    assert len(params_detached) == len(dict(model_copy.named_parameters())) and all(
        torch.allclose(params_detached[name], v) for name, v in model_copy.named_parameters())
    assert len(buffers_detached) == len(dict(model_copy.named_buffers())) and all(
        torch.allclose(buffers_detached[name], v) for name, v in model_copy.named_buffers())

    print("passed")
