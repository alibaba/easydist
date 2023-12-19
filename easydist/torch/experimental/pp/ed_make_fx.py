import functools
import inspect
import os
from contextlib import nullcontext
from typing import Any
# make easydist happy without torchrun
os.environ['MASTER_PORT'] = '-1'

import torch.fx as fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (PreDispatchTorchFunctionMode,
                                                ProxyTorchDispatchMode, PythonKeyTracer, decompose,
                                                disable_autocast_cache,
                                                disable_proxy_modes_tracing, dispatch_trace,
                                                enable_pre_dispatch, enable_python_dispatcher,
                                                fake_signature, wrap_key)
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from easydist.torch.experimental.pp.compile_pipeline import (get_tracer_global, set_tracer_global)


# copied from torch.fx.experimental.proxy_tensor
def ed_make_fx(f,
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
            old_tracer = get_tracer_global()
            set_tracer_global(fx_tracer)
            try:
                t = dispatch_trace(wrap_key(func, args, fx_tracer, pre_dispatch),
                                   tracer=fx_tracer,
                                   concrete_args=tuple(phs))
            finally:
                set_tracer_global(old_tracer)

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if tracing_mode == "symbolic":
            t.shape_env = shape_env  # type: ignore[assignment]
        return t

    return wrapped
