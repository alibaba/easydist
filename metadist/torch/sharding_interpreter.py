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

import copy
import logging
import os
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.utils._pytree as pytree
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter
from torch.fx.node import Argument, Target, _get_qualified_name
from torch._subclasses.fake_tensor import FakeTensorMode

from metadist.metashard import MetaOp
from metadist.torch.utils import to_meta
import metadist.config as mdconfig
from .preset_propagation import preset_meta_spmd
from .device_mesh import device_mesh_world_size

logger = logging.getLogger(__name__)


def get_shape_info(node_output):
    if isinstance(node_output, torch.Tensor) or isinstance(node_output,
                                                           torch.nn.parameter.Parameter):
        return {"shape": node_output.shape, "dtype": node_output.dtype}
    else:
        return {}


def meta_exec(self, flat_meta_input=None, enable_fallback=True):

    fake_tensor_mode = FakeTensorMode()

    def wrap_fake(x):
        if isinstance(x, torch.Tensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    if flat_meta_input is None:
        flat_meta_input = self.flat_input_args
    try:
        fake_flat_args = pytree.tree_map(wrap_fake, flat_meta_input)
        fake_args, fake_kwargs = pytree.tree_unflatten(fake_flat_args, self.input_pytree_spec)
        with fake_tensor_mode:
            out = self.func(*fake_args, **fake_kwargs)
    except:
        if enable_fallback:
            out = self.exec()
        else:
            raise RuntimeError(f"failed meta_exec with MetaOp {self}")
    return out


# register the meta_exec function in MetaOp
MetaOp.meta_exec = meta_exec


def to_real(tensor, size=None):

    device = mdconfig.metadist_device

    if size is None:
        size = tensor.size()

    if isinstance(tensor, torch.Tensor) and tensor.is_meta:
        if tensor.dtype == torch.bool:
            return torch.rand(size, dtype=torch.float, device=device) > 0.5
        elif torch.is_floating_point(tensor):
            return torch.rand(size, dtype=tensor.dtype, device=device)
        else:
            return torch.randint(high=1, size=size, dtype=tensor.dtype, device=device)
    return tensor


class MDTorchShardingAnn(Interpreter):

    def __init__(self, module: GraphModule, use_cache=True, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.use_cache = use_cache
        self.shape_info = {}
        self.sharding_info = {}

        self.pass_by_ops = ["_operator.getitem"]

    def run(self, *args) -> Any:
        """
        Run `module` via interpretation and return the result.
        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
        Returns:
            Any: The value returned from executing the Module
        """
        self.env = {}

        args = pytree.tree_map(to_meta, args)

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        self.args_iter: Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:

            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            node_output = self.run_node(node)
            self.env[node] = pytree.tree_map(to_meta, node_output)
            self.shape_info[node.name] = pytree.tree_map(get_shape_info, self.env[node])

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == 'output':
                # output_val = self.env[node]
                self.env = {}
                return self.sharding_info, self.shape_info

    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)

        if "placeholder" not in self.sharding_info:
            self.sharding_info["placeholder"] = {}

        if target.startswith('*'):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            try:
                output = next(self.args_iter)
                if isinstance(output, torch.Tensor):
                    out_meta_str = str(output)
                    if out_meta_str not in self.sharding_info["placeholder"]:
                        sharding_ann, combination_ann = preset_meta_spmd("placeholder", (output, kwargs))
                        if sharding_ann or combination_ann:
                            self.sharding_info["placeholder"][out_meta_str] = {
                                "sharding_ann": sharding_ann,
                                "combination_ann": combination_ann,
                            }
                return output
            except StopIteration as si:
                if len(args) > 0:
                    return args[0]
                else:
                    raise RuntimeError(f'Expected positional argument for parameter {target}, but one was not passed in!') from si

    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        assert isinstance(target, str)
        output = self.fetch_attr(target)

        if "get_attr" not in self.sharding_info:
            self.sharding_info["get_attr"] = {}

        if isinstance(output, torch.Tensor):
            out_meta_str = str(output)
            if out_meta_str not in self.sharding_info["get_attr"]:
                sharding_ann, combination_ann = preset_meta_spmd("get_attr", (output, kwargs))
                if sharding_ann or combination_ann:
                    self.sharding_info["get_attr"][out_meta_str] = {
                        "sharding_ann": sharding_ann,
                        "combination_ann": combination_ann,
                    }
        return output

    def call_function(self, target: 'Target', args: Tuple[Argument, ...],
                      kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.
        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        ops_name = _get_qualified_name(target)

        if ops_name in self.pass_by_ops:
            real_out = target(*args, **kwargs)
            return real_out

        args_meta, kwargs_meta = args, kwargs

        op_perf_key = {
            "ops_name": ops_name,
            "args_meta": str(args_meta) + ' | ' + str(kwargs_meta)
        }
        logger.debug(op_perf_key)

        def get_hint_size(args_list, mock_dim_value=1024):
            size_dict = {}
            element_number = 0
            max_dim_value = 0
            for arg_idx in range(len(args_list)):
                if isinstance(args_list[arg_idx], torch.Tensor):
                    tensor_size = args_list[arg_idx].size()
                    size_dict[arg_idx] = tensor_size
                    element_number += tensor_size.numel()
                    if len(args_list[arg_idx].size()) > 0:
                        max_dim_value = max(max_dim_value, max(tensor_size))

            # if element number of all tensor larger then 1024 ** 3,
            hint_size = size_dict
            if mdconfig.use_hint and element_number >= 1024**3:
                hint_size = {}
                for arg_idx in size_dict:
                    hint_size[arg_idx] = torch.Size([
                        mock_dim_value if dim == max_dim_value else dim
                        for dim in size_dict[arg_idx]
                    ])

                logger.info(f"use mock shape for {op_perf_key}, {size_dict} -> {hint_size}")

            return size_dict, hint_size

        # (CAUTION !!) use mock (smaller size) args/kwargs here for OOM issue of giant Op.
        # materialize the input of ops, execute the function and return the result
        def materialize_args_kwargs(args, kwargs):
            flat_args, args_specs = pytree.tree_flatten((args, kwargs))

            ori_size, hint_size = get_hint_size(flat_args)

            def materialize_with_size_dict(size_dict):
                for arg_idx in range(len(flat_args)):
                    if isinstance(flat_args[arg_idx], torch.Tensor):
                        real_tensor = to_real(flat_args[arg_idx], size_dict[arg_idx])

                        # clip for embedding op prevent indexerror in CUDA kernel.
                        if ops_name in [
                                "torch.ops.aten.embedding.default",
                                "metadist.torch.passes.fix_embedding.md_embedding"
                        ]:
                            real_tensor.clip_(max=args[0].shape[0])

                        flat_args[arg_idx] = real_tensor

            materialize_with_size_dict(hint_size)

            # try to run the metaop on the mock size (hint_size)
            # if failed with materialize with the original size
            try:
                meta_op = MetaOp(func=target, input_args=(args, kwargs), name=ops_name)
                meta_op.meta_exec(flat_meta_input=flat_args)
            except:
                materialize_with_size_dict(ori_size)

            return pytree.tree_unflatten(flat_args, args_specs)

        args, kwargs = materialize_args_kwargs(args_meta, kwargs_meta)
        meta_op = MetaOp(func=target,
                         input_args=(args, kwargs),
                         shard_size=device_mesh_world_size(),
                         name=ops_name)
        # user fake tensor here, maybe use shape/dtype info from `make_fx`
        meta_out = meta_op.meta_exec(flat_meta_input=pytree.tree_flatten((args_meta,
                                                                          kwargs_meta))[0])

        # sharding discovery
        if op_perf_key["ops_name"] not in self.sharding_info:
            self.sharding_info[op_perf_key["ops_name"]] = {}

        if not self.use_cache or op_perf_key["args_meta"] not in self.sharding_info[
                op_perf_key["ops_name"]]:
            prompt_annotation = None
            if self.use_cache and len(self.sharding_info[op_perf_key["ops_name"]]) >= 1:
                prompt_annotation = list(
                    self.sharding_info[op_perf_key["ops_name"]].values())[0]["sharding_ann"]

            sharding_ann, combination_ann = preset_meta_spmd(meta_op, (args_meta, kwargs_meta))
            if sharding_ann is None and combination_ann is None:
                sharding_ann, combination_ann = meta_op.sharding_discovery(
                    prompt_annotation=copy.deepcopy(prompt_annotation))

            self.sharding_info[op_perf_key["ops_name"]][op_perf_key["args_meta"]] = {
                "sharding_ann": sharding_ann,
                "combination_ann": combination_ann,
            }

        return pytree.tree_map(to_meta, meta_out)
