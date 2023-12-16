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
import importlib
from functools import update_wrapper
from typing import Any

import numpy
import torch
import torch.utils._pytree as pytree
from torch.distributed._tensor import DeviceMesh

import easydist.config as mdconfig
from easydist.torch.device_mesh import (device_mesh_world_size, get_device_mesh, set_device_mesh)
from easydist.torch.compile_auto import _compile_auto
from easydist.torch.compiler_dp import _compile_dp
from easydist.torch.init_helper import (CpuModuleInitHelper, SetParaInitHelper)
from easydist.torch.utils import get_input_signature

logger = logging.getLogger(__name__)

PARALLEL_EXTENTION = dict()


def register_parallel_method(parallel_mode: str, compiler_func=None):

    def wrapper(compiler_func):
        global PARALLEL_EXTENTION
        PARALLEL_EXTENTION[parallel_mode] = compiler_func
        logger.info(f"Register parallel method [{parallel_mode}]: {compiler_func}")
        return compiler_func

    if compiler_func is None:
        return wrapper
    else:
        return wrapper(compiler_func)


class CompiledFuncWrapper:

    def __init__(self,
                 func,
                 parallel_mode="auto",
                 tracing_mode="fake",
                 cuda_graph=True,
                 enable_mono_graph=False,
                 compile_only=False) -> None:
        update_wrapper(self, func)
        self.original_func = func

        self.compiled_func = None
        self.parallel_mode = parallel_mode
        self.tracing_mode = tracing_mode
        self.enable_cuda_graph = cuda_graph
        self.enable_mono_graph = enable_mono_graph
        self.compile_only = compile_only

        self.init_helper = SetParaInitHelper()

        self.all_input_signature = []
        self.graph_list = {}

        self.cuda_graph_space = {}
        self.graph_pool = None

    def register_input_signature(self, *args: Any, **kwargs: Any) -> str:

        input_signature = get_input_signature(*args, **kwargs)

        if input_signature not in self.all_input_signature:

            self.all_input_signature.append(input_signature)
            logger.info(f"[Compile API] register input signature: {input_signature}")

            if self.enable_cuda_graph:
                self.cuda_graph_space[input_signature] = {
                    "cuda_graph": None,
                    "cuda_graph_input": None,
                    "cuda_graph_output": None,
                }

        return input_signature

    def register_cpu_module(self, cpu_module):
        self.init_helper = CpuModuleInitHelper(cpu_module)

    def set_init_helper(self, init_helper):
        self.init_helper = init_helper

    def __call__(self, *args: Any, **kwargs: Any) -> Any:

        input_signature = self.register_input_signature(*args, **kwargs)

        # skip compile when only one device
        world_size = torch.distributed.get_world_size()
        need_compile = world_size >= 2
        if get_device_mesh() is not None:
            need_compile = device_mesh_world_size() >= 2

        # override need_compile with forced_compile env var
        if mdconfig.forced_compile:
            need_compile = True

        if need_compile and self.compiled_func is None:
            mesh_shape = numpy.array(range(world_size)).reshape(1, -1)
            mesh = DeviceMesh("cuda", mesh_shape.tolist())

            if get_device_mesh() == None:
                set_device_mesh(mesh)

            if self.parallel_mode == "auto":
                self.compiled_func = _compile_auto(self.original_func, self.tracing_mode,
                                                   self.init_helper, input_signature, args, kwargs)
            elif self.parallel_mode in ["ddp", "zero2", "zero3"]:
                self.compiled_func = _compile_dp(self.original_func, self.parallel_mode,
                                                 self.tracing_mode, args, kwargs)
            elif self.parallel_mode in PARALLEL_EXTENTION:
                self.compiled_func = PARALLEL_EXTENTION[self.parallel_mode](self.original_func,
                                                                            self.parallel_mode,
                                                                            self.tracing_mode,
                                                                            args, kwargs)
            else:
                raise NotImplementedError()
            self.graph_list[input_signature] = self.compiled_func.graph
            # release the cpu module when finised pre-shard in _compiler
            self.init_helper = None
        if self.compile_only:
            return self.compiled_func

        def run_func(*args, **kwargs):
            if self.compiled_func:
                if input_signature not in self.graph_list:
                    if self.enable_mono_graph:
                        mono_graph = self.compiled_func.compile_mono_graph(*args, **kwargs)
                        self.graph_list[input_signature] = mono_graph
                        logger.info(f"[Compile API] compile mono graph for {input_signature}")
                    else:
                        raise RuntimeError(
                            "Input mismatch. If you are sure that different inputs do not change the graph, "
                            "you can try turning on the enable_mono_graph option.")
                graph = self.graph_list[input_signature]
                output = self.compiled_func.run_with_graph(graph, *args, **kwargs)
                if importlib.util.find_spec("torch.distributed._functional_collectives"):
                    # wait all AsyncCollectiveTensor and unwrap AsyncCollectiveTensor
                    from torch.distributed._functional_collectives import AsyncCollectiveTensor
                    from torch.distributed._functional_collectives_impl import _wait_all
                    _wait_all()

                    def wait_unwarp_fn(async_coll_tensor_):
                        return async_coll_tensor_.elem

                    return pytree.tree_map_only(AsyncCollectiveTensor, wait_unwarp_fn, output)
                else:
                    return output
            else:
                return self.original_func(*args, **kwargs)

        if self.enable_cuda_graph:
            current_space = self.cuda_graph_space[input_signature]
            if current_space["cuda_graph"] is None:
                logger.info("[Compile API] cuda graph warming up...")

                flatten_args, args_specs = pytree.tree_flatten([args, kwargs])
                current_space["cuda_graph_input"] = []
                for f_args in flatten_args:
                    if isinstance(f_args, torch.Tensor):
                        current_space["cuda_graph_input"].append(
                            torch.empty_like(f_args).copy_(f_args))
                    else:
                        current_space["cuda_graph_input"].append(f_args)
                args, kwargs = pytree.tree_unflatten(current_space["cuda_graph_input"], args_specs)

                # warm up
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    current_space["cuda_graph_output"] = run_func(*args, **kwargs)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                # sleep to prevent cuda error when cuda graph capture.
                time.sleep(2)

                current_space["cuda_graph"] = torch.cuda.CUDAGraph()

                with torch.cuda.graph(current_space["cuda_graph"], self.graph_pool):
                    current_space["cuda_graph_output"] = run_func(*args, **kwargs)

                # assign the pool of first cuda graph as self.graph_pool
                if self.graph_pool is None:
                    self.graph_pool = current_space["cuda_graph"].pool()
            else:
                flatten_args, args_specs = pytree.tree_flatten([args, kwargs])
                for static_input, outside_input in zip(current_space["cuda_graph_input"],
                                                       flatten_args):
                    if isinstance(static_input, torch.Tensor):
                        static_input.copy_(outside_input)

            current_space["cuda_graph"].replay()
            return current_space["cuda_graph_output"]
        else:
            return run_func(*args, **kwargs)


def easydist_compile(func=None,
                     parallel_mode="auto",
                     tracing_mode="fake",
                     cuda_graph=True,
                     enable_mono_graph=False,
                     use_hint=False,
                     liveness_only_input=False,
                     max_solver_time=float("inf"),
                     compile_only=False):

    mdconfig.use_hint = use_hint
    mdconfig.liveness_only_input = liveness_only_input
    mdconfig.max_seconds_same_incumbent = max_solver_time

    if parallel_mode not in ["auto", "ddp", "zero2", "zero3"
                             ] and parallel_mode not in PARALLEL_EXTENTION:
        raise NotImplementedError(
            "please use [auto, ddp, zero2, zero3] for `parallel_mode` or register your parallel extention"
        )
    if func:
        return CompiledFuncWrapper(func, parallel_mode, tracing_mode, cuda_graph,
                                   enable_mono_graph, compile_only)
    else:

        def wrapper(func):
            return CompiledFuncWrapper(func, parallel_mode, tracing_mode, cuda_graph,
                                       enable_mono_graph, compile_only)

        return wrapper
