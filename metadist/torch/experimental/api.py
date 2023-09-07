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
from functools import update_wrapper
from typing import Any

import numpy
import torch
import torch.utils._pytree as pytree
from torch.distributed._tensor import DeviceMesh

import metadist.config as mdconfig
from metadist.torch.device_mesh import (device_mesh_world_size, get_device_mesh, set_device_mesh)
from metadist.torch.experimental.compiler import _compile
from metadist.torch.experimental.init_helper import (CpuModuleInitHelper, SetParaInitHelper)
from metadist.torch.utils import get_input_signature

logger = logging.getLogger(__name__)


class CompiledFuncWrapper:

    def __init__(self,
                 func,
                 tracing_mode="fake",
                 cuda_graph=True,
                 enable_mono_graph=False) -> None:
        update_wrapper(self, func)
        self.original_func = func

        self.compiled_func = None
        self.tracing_mode = tracing_mode
        self.enable_cuda_graph = cuda_graph
        self.enable_mono_graph = enable_mono_graph

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
        if need_compile and self.compiled_func is None:
            mesh_shape = numpy.array(range(world_size)).reshape(1, -1)
            mesh = DeviceMesh("cuda", mesh_shape.tolist())

            if get_device_mesh() == None:
                set_device_mesh(mesh)

            self.compiled_func = _compile(self.original_func, self.tracing_mode, self.init_helper,
                                          input_signature, args, kwargs)
            self.graph_list[input_signature] = self.compiled_func.graph
            # release the cpu module when finised pre-shard in _compiler
            self.init_helper = None

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
                return self.compiled_func.run_with_graph(graph, *args, **kwargs)
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


def metadist_compile(func=None,
                     tracing_mode="fake",
                     cuda_graph=True,
                     enable_mono_graph=False,
                     use_hint=False,
                     liveness_only_input=False,
                     max_solver_time=float("inf")):

    mdconfig.use_hint = use_hint
    mdconfig.liveness_only_input = liveness_only_input
    mdconfig.max_seconds_same_incumbent = max_solver_time

    if func:
        return CompiledFuncWrapper(func, tracing_mode, cuda_graph, enable_mono_graph)
    else:

        def wrapper(func):
            return CompiledFuncWrapper(func, tracing_mode, cuda_graph, enable_mono_graph)

        return wrapper
