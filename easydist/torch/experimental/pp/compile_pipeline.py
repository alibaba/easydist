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
import operator
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.distributed._tensor import Replicate
from torch.distributed._tensor.placement_types import Placement

from easydist.metashard.metair import VarSPMDStrategy
from easydist.torch.experimental.pp.ed_split_module import ed_split_module
from easydist.torch.experimental.pp.split_utils import (
    get_backward_flag,
    split,
    split_func_with_bw,
)
from easydist.torch.experimental.pp.utils import (
    OneToOneMap,
    _to_tuple,
    do_spmd_comm,
    ordered_gi_users,
    save_graphviz_dot,
)
from easydist.utils import rgetattr, rsetattr

# ================================= section start ========================================
# Functions in this section are modified from
# https://github.com/pytorch/PiPPy/blob/e9e2d5f0164a2e5d952a1424a3926da543365801/pippy/IR.py#L1206


def split_after_forward(module: torch.nn.Module):
    def hook(module, args, ret):
        ret_on_new_stage = split(ret)
        return ret_on_new_stage
    module.register_forward_hook(hook, prepend=True)
    return module

# Copyright (c) Meta Platforms, Inc. and affiliates

def annotate_split_points(module: torch.nn.Module, spec: Set[str]):
    if not isinstance(spec, set):
        raise TypeError(f"spec should be a set of strings, found {type(spec)=} {spec=}")

    # TODO: make this implementation out-of-place?
    for qualname in iter(spec):
        atoms = qualname.split(".")
        predecessor_module = module
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(
                    f'Specified target {qualname} referenced nonexistent module {".".join(atoms[:i+1])}'
                )

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        wrapped_mod = split_after_forward(mod_to_wrap)
        setattr(predecessor_module, atoms[-1], wrapped_mod)


def split_into_equal_size(nstages: int = 1, ) -> Callable[[torch.nn.Module], torch.fx.GraphModule]:

    def _split_into_nstages_equal_size(module: torch.nn.Module) -> torch.fx.GraphModule:
        tracer = torch.fx.Tracer()
        g = tracer.trace(module)
        gm = torch.fx.GraphModule(module, g)
        param_size = 0
        for param in gm.parameters():
            param_size += param.numel()
        buffer_size = 0
        for buffer in gm.buffers():
            buffer_size += buffer.numel()

        total_size = param_size + buffer_size
        per_stage_size = total_size // nstages
        logging.debug(f"Total model size: {total_size}, "
                      f"per stage size: {per_stage_size}")

        gm, rv_nstages = _split_on_size_threshold_with_max_stages(gm, per_stage_size, nstages)
        assert rv_nstages == nstages
        return nstages, gm

    return _split_into_nstages_equal_size


def _analyze_node_size(gm: torch.fx.GraphModule, ) -> Dict[torch.fx.Node, Dict[str, int]]:
    # state_dict helps us to get parameter sizes
    state_dict = gm.state_dict()

    # Function Parameter Usage
    node_param_sizes: Dict[torch.fx.Node, Dict[str, int]] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":  # a parameter node
            param_name = node.target
            assert param_name in state_dict
            param = state_dict[param_name]
            # Find use site of this parameter
            for user in node.users:
                func_param_sizes = node_param_sizes.setdefault(user, {})
                func_param_sizes.setdefault(param_name, param.numel())

    # Module Parameter Usage
    for node in gm.graph.nodes:
        # We calcuate size of a user-defined submodule as a whole
        if node.op == "call_module":
            mod_param_sizes: Dict[str, int] = {}
            submod: torch.nn.Module = gm.get_submodule(node.target)
            for param_name, param in submod.named_parameters():
                mod_param_sizes.setdefault(param_name, param.numel())
            if mod_param_sizes:
                node_param_sizes.setdefault(node, mod_param_sizes)

    for node, param_sizes in node_param_sizes.items():
        logging.debug(f"{node} has params: {param_sizes}")

    return node_param_sizes


def _split_on_size_threshold_with_max_stages(
    gm: torch.fx.GraphModule,
    threshold: int,
    max_stages: int = -1,
) -> Tuple[torch.fx.GraphModule, int]:
    # Analyze size of parameters/buffers used by each node in the graph
    node_param_sizes = _analyze_node_size(gm)

    # Record split positions
    insert_after_nodes: List[torch.fx.Node] = []

    def new_stage_after(node):
        insert_after_nodes.append(node)

    # Track the parameters we have seen in the current bucket and their total size
    accumulate_size = 0
    accumulate_params: Dict = {}
    checked_nodes = []
    for node in gm.graph.nodes:
        if node not in node_param_sizes:
            # The callsite of this node does not involve parameters or buffers
            continue

        # Track the new parameters we see at this node as well as parameters that we have seen in current bucket
        new_size = 0
        new_params: Dict = {}
        repeated_size = 0
        repeated_params: Dict = {}
        param_sizes = node_param_sizes[node]
        if node.op == "call_function":
            checked_nodes.append(node)
            # For function, the parameter it uses can be shared with other functions seen previously
            for param_name, size in param_sizes.items():
                if param_name not in accumulate_params:  # new parameter
                    new_params.setdefault(param_name)
                    new_size += size
                else:  # repeated parameter; mark down; use later
                    repeated_params.setdefault(param_name)
                    repeated_size += size
        elif node.op == "call_module":
            checked_nodes.append(node)
            # For module, we count its paramters as a single whole
            for param_name, size in param_sizes.items():
                new_size += size

        if (accumulate_size + new_size
                <= threshold):  # can accommodate this node in current bucket
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif (accumulate_size == 0 and new_size > threshold):  # this node becomes a stage
            new_stage_after(node)
        else:  # cannot accommodate this node
            try:
                new_stage_after(checked_nodes[-2])
            except IndexError:
                raise RuntimeError(
                    f"Cannot split graph into stages with size threshold {threshold} and max stages {max_stages}"
                )
            accumulate_size = repeated_size + new_size
            accumulate_params.clear()
            accumulate_params.update(repeated_params)
            accumulate_params.update(new_params)

    def gen_func_wrapper(target_func):

        def wrapped_func(*args, **kwargs):
            ret = target_func(*args, **kwargs)
            ret = _to_tuple(ret)
            ret = split_func_with_bw(*ret)
            return ret[0] if len(ret) == 1 else ret

        return wrapped_func

    def gen_module_wrapper(target_module):
        return split_after_forward(target_module)

    nstages = 1
    for node in insert_after_nodes:
        if nstages == max_stages:
            break
        if node.op == "call_function":
            node.target = gen_func_wrapper(node.target)
        else:
            assert node.op == "call_module"
            rsetattr(gm, node.target, gen_module_wrapper(rgetattr(gm, node.target)))

        nstages += 1

    # Since we transformed the graph, we need to recompile the module
    gm.recompile()

    return gm, nstages


# ================================= section end ========================================


class StateType(Enum):
    PARAMS = "params"
    BUFFERS = "buffers"
    OPTIMSTATES = "optimstates"

    @staticmethod
    def get_empty_dict() -> Dict:
        return {
            StateType.PARAMS: {},
            StateType.BUFFERS: {},
            StateType.OPTIMSTATES: {}
        }


class SubmodType(Enum):
    FW = "fw"
    BW = "bw"
    STEP = "step"


@dataclass
class CompiledMeta:
    # pp_size
    nstages: int

    # stateless_func spec
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

    # torch name to node name mapping
    input_params_map: OneToOneMap
    input_buffers_map: OneToOneMap
    input_optimstates_map: OneToOneMap
    output_params_map: OneToOneMap
    output_buffers_map: OneToOneMap
    output_optimstates_map: OneToOneMap
    output_grads_map: OneToOneMap

    # step graph
    optim_state_types: Set[str]
    input_node_to_step_input_params: OneToOneMap
    input_node_to_step_input_grads: OneToOneMap

    # stateless_func node names
    output_nodes_flatten: Tuple[str, ...]
    args_nodes_unflatten: Tuple[str, ...]
    kwargs_nodes_unflatten: Dict[str, str]
    return_nodes_flatten: Tuple[str, ...]
    return_nodes_spec: pytree.TreeSpec

    # spmd strategy
    tensors_spmd_strategies: Dict[str, List[Placement]]

    # input node to output node mapping
    input_to_output_params_map: OneToOneMap=None
    input_to_output_buffers_map: OneToOneMap=None
    input_to_output_optimstates_map: OneToOneMap=None

    def __post_init__(self):
        assert self.input_to_output_params_map is None
        assert self.input_to_output_buffers_map is None
        assert self.input_to_output_optimstates_map is None
        self.input_to_output_params_map = self.input_params_map.inverse().apply(self.output_params_map)
        self.input_to_output_buffers_map = self.input_buffers_map.inverse().apply(self.output_buffers_map)
        self.input_to_output_optimstates_map = self.input_optimstates_map.inverse().apply(self.output_optimstates_map)


@dataclass
class EDGraphModule:
    gm: torch.fx.GraphModule
    submod_type: SubmodType
    inputs_spec: Set[str]
    node_states: Dict[StateType, Dict[str, torch.Tensor]]
    outputs_spec: List[str]
    call_module_users: Dict[str, Set[str]]
    name: str

    def __call__(self, *args, **kwargs):
        return self.gm(*args, **kwargs)


class CompiledStage:

    def __init__(self,
                 compiled_meta: CompiledMeta,
                 stage_fw_gm: EDGraphModule,
                 stage_bw_gm: Optional[EDGraphModule],
                 full_step_gm: Optional[EDGraphModule]):
        self.has_bw = False
        self.has_step = False
        self.compiled_meta = compiled_meta
        self.fw_gm = stage_fw_gm
        stage_param_nodes = set(stage_fw_gm.node_states[StateType.PARAMS].keys())
        stage_buffer_nodes = set(stage_fw_gm.node_states[StateType.BUFFERS].keys())
        stage_param_and_buffer_nodes = stage_param_nodes | stage_buffer_nodes
        self.fw_func_args = set(stage_fw_gm.inputs_spec) - stage_param_and_buffer_nodes  # user inputs for self.forward
        self.fw_func_returns = set(
            output for output, users in self.fw_gm.call_module_users.items()
            if len(set(users) - set([stage_bw_gm.name] if stage_bw_gm is not None else [])) > 0)  # not just used by bw, need to pass to next stage

        if stage_bw_gm is not None:
            self.has_bw = True
            self.bw_gm = stage_bw_gm
            from_fw_inputs = set(stage_fw_gm.inputs_spec) & set(stage_bw_gm.inputs_spec)
            from_fw_outputs = set(stage_fw_gm.outputs_spec) & set(stage_bw_gm.inputs_spec)
            self.tensors_to_save_bw = from_fw_inputs | from_fw_outputs
            self.bw_func_args = set(stage_bw_gm.inputs_spec) - stage_param_and_buffer_nodes - set(self.tensors_to_save_bw)
            self.bw_func_returns = set(self.bw_gm.call_module_users.keys())

        if full_step_gm is not None:
            self.has_step = True
            stage_optim_input_params = set(
                compiled_meta.input_node_to_step_input_params.get(node_name) for node_name in stage_param_nodes
            )
            stage_optim_input_grads = set(
                compiled_meta.input_node_to_step_input_grads.get(node_name) for node_name in stage_param_nodes
            )
            stage_optim_input_states = set(
                compiled_meta.input_optimstates_map.get(
                    (compiled_meta.input_params_map.inv(node_name), state_type)
                ) for node_name in stage_param_nodes for state_type in compiled_meta.optim_state_types
            )
            self.optim_grads = stage_optim_input_grads
            self.step_func_args = (stage_optim_input_params | stage_optim_input_grads) | stage_optim_input_states
            self.stage_step_gm = _extract_step_subgraph_from_args(full_step_gm, self.step_func_args)
            save_graphviz_dot(self.stage_step_gm.gm, self.fw_gm.name + '(step)')

    @torch.no_grad
    def forward(self, saved_tensors_bw: Optional[Dict]=None, returns_chunk: Optional[Dict]=None, **kwargs):
        assert set(kwargs.keys()) == self.fw_func_args, f"known kwargs {kwargs}, {self.fw_func_args} are required"

        if saved_tensors_bw is None or returns_chunk is None:  # for local run
            assert saved_tensors_bw is None and returns_chunk is None
            self.saved_tensors_bw = {}
            self.returns = {}
            saved_tensors_bw = self.saved_tensors_bw
            returns_chunk = self.returns

        kwargs_gm = {}
        for arg_name in self.fw_gm.inputs_spec:
            if arg_name in kwargs:
                kwargs_gm[arg_name] = kwargs.pop(arg_name)
            elif arg_name in self.fw_gm.node_states[StateType.PARAMS]:  # params
                kwargs_gm[arg_name] = self.fw_gm.node_states[StateType.PARAMS][arg_name]
            elif arg_name in self.fw_gm.node_states[StateType.BUFFERS]:  # buffers
                kwargs_gm[arg_name] = self.fw_gm.node_states[StateType.BUFFERS][arg_name]
            else:
                raise RuntimeError(f"arg {arg_name} not found")

            if self.has_bw and arg_name in self.tensors_to_save_bw:
                saved_tensors_bw[arg_name] = kwargs_gm[arg_name]

        with torch.profiler.record_function("actual_compute"):
            output_from_gm = _to_tuple(self.fw_gm(**kwargs_gm))

        ret = {}
        assert len(output_from_gm) == len(
            self.fw_gm.outputs_spec
        ), "output_from_gm should have the same length as self.fw_gm.outputs_spec"
        for output_name, output in zip(self.fw_gm.outputs_spec, output_from_gm):
            if output_name in self.fw_func_returns:
                    ret[output_name] = output

            if output_name in self.compiled_meta.return_nodes_flatten:
                assert output_name not in returns_chunk
                returns_chunk[output_name] = output

            if output_name in self.compiled_meta.output_buffers_map.inv_keys():  # updated buffers
                input_name = self.compiled_meta.input_buffers_map.get(
                    self.compiled_meta.output_buffers_map.inv(output_name)
                )
                self.fw_gm.node_states[StateType.BUFFERS][input_name] = output

            if self.has_bw and output_name in self.tensors_to_save_bw:
                saved_tensors_bw[output_name] = output

        return ret

    @torch.no_grad
    def backward(self, saved_tensors_bw: Optional[Dict]=None, grads: Optional[Dict]=None, **kwargs):
        if not self.has_bw:
            raise NotImplementedError("This compiled stage doesn't contain bw_gm")

        assert set(kwargs.keys()) == self.bw_func_args, "backward args should be saved for fw"

        if saved_tensors_bw is None or grads is None:  # for local run
            assert saved_tensors_bw is None and grads is None
            self.grads = {}
            saved_tensors_bw = self.saved_tensors_bw
            grads = self.grads

        kwargs_gm = {}
        for arg_name in self.bw_gm.inputs_spec:
            if arg_name in kwargs:
                kwargs_gm[arg_name] = kwargs.pop(arg_name)
            elif arg_name in saved_tensors_bw:
                kwargs_gm[arg_name] = saved_tensors_bw.pop(arg_name)
            else:
                raise RuntimeError(f"arg {arg_name} not found")

        assert len(saved_tensors_bw) == 0, f"all backward args should be used, but found {saved_tensors_bw} {len(saved_tensors_bw)}"
        with torch.profiler.record_function("actual_compute"):
            output_gm = _to_tuple(self.bw_gm(**kwargs_gm))

        ret = {}
        assert len(output_gm) == len(
            self.bw_gm.outputs_spec
        ), "output_from_gm should have the same length as self.bw_gm.outputs_spec"
        for output_name, output in zip(self.bw_gm.outputs_spec, output_gm):
            if output_name in self.bw_func_returns:
                ret[output_name] = output
            else:
                grads[output_name] = output

        return ret

    @torch.no_grad
    def step(self, grads: Optional[Dict]=None):
        if not self.has_step:
            raise NotImplementedError("This compiled stage doesn't contain step_gm")

        if grads is None:
            grads = self.grads

        with torch.profiler.record_function("actual_compute"):
            output_gm = self.stage_step_gm(**grads, **self.stage_step_gm.node_states[StateType.OPTIMSTATES], **self.fw_gm.node_states[StateType.PARAMS])
        grads.clear()

        for output_name, output in zip(self.stage_step_gm.outputs_spec, output_gm):
            if output_name in self.compiled_meta.output_params_map.inv_keys():  # updated params
                input_name = self.compiled_meta.input_params_map.get(
                    self.compiled_meta.output_params_map.inv(output_name)
                )
                self.fw_gm.node_states[StateType.PARAMS][input_name] = output  # this is updated in place when there is no comm node
            elif output_name in self.compiled_meta.output_optimstates_map.inv_keys():  # updated optim states
                input_name = self.compiled_meta.input_optimstates_map.get(
                    self.compiled_meta.output_optimstates_map.inv(output_name)
                )
                self.stage_step_gm.node_states[StateType.OPTIMSTATES][input_name] = output # this is updated in place when there is no comm node

        return None

    def has_step(self):
        return hasattr(self, 'step_gm')

    def has_bw(self):
        return hasattr(self, 'bw_gm')

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict.update(self.named_parameters())
        state_dict.update(self.named_buffers())
        return state_dict

    def _load_params(self, state_dict: Dict[str, Any]):
        to_pop = []
        for torch_name, tensor in state_dict.items():
            node_name = self.compiled_meta.input_params_map.get(torch_name)
            if node_name in self.compiled_meta.tensors_spmd_strategies:
                src_specs = [Replicate()] * len(src_specs)
                tgt_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                tensor = do_spmd_comm(tensor, src_specs, tgt_specs)
            self.fw_gm.node_states[StateType.PARAMS][node_name] = tensor
            to_pop.append(torch_name)
        for torch_name in to_pop:
            state_dict.pop(torch_name)

    def _load_buffers(self, state_dict: Dict[str, Any]):
        to_pop = []
        for torch_name, tensor in state_dict.items():
            node_name = self.compiled_meta.input_buffers_map.get(torch_name)
            if node_name in self.compiled_meta.tensors_spmd_strategies:
                src_specs = [Replicate()] * len(src_specs)
                tgt_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                tensor = do_spmd_comm(tensor, src_specs, tgt_specs)
            self.fw_gm.node_states[StateType.BUFFERS][node_name] = tensor
            to_pop.append(torch_name)
        for torch_name in to_pop:
            state_dict.pop(torch_name)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool=True):
        self._load_params(state_dict)
        self._load_buffers(state_dict)
        if strict and len(state_dict) != 0:
            raise RuntimeError(f"Unexpected keys in state_dict {state_dict.keys()}")

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Dict[str, Any]], strict: bool=True):
        for torch_name, states in state_dict.items():
            for state_type, tensor in states.items():
                node_name = self.compiled_meta.input_optimstates_map.get((torch_name, state_type))
                if node_name in self.compiled_meta.tensors_spmd_strategies:
                    src_specs = [Replicate()] * len(src_specs)
                    tgt_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                    tensor = do_spmd_comm(tensor, src_specs, tgt_specs)
                self.stage_step_gm.node_states[StateType.OPTIMSTATES][node_name] = tensor

    def named_parameters(self) -> Dict[str, Any]:
        if self.compiled_meta.tensors_spmd_strategies:
            params = {}
            for node_name, tensor in self.fw_gm.node_states[StateType.PARAMS].items():
                src_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                tgt_specs = [Replicate()] * len(src_specs)
                tensor = do_spmd_comm(tensor, src_specs, tgt_specs)
                torch_name = self.compiled_meta.input_params_map.inv(node_name)
                params[torch_name] = tensor
            return params
        else:
            return {
                self.compiled_meta.input_params_map.inv(node_name): tensor
                for node_name, tensor in self.fw_gm.node_states[StateType.PARAMS].items()
            }

    def named_buffers(self) -> Dict[str, Any]:
        if self.compiled_meta.tensors_spmd_strategies:
            buffers = {}
            for node_name, tensor in self.fw_gm.node_states[StateType.BUFFERS].items():
                src_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                tgt_specs = [Replicate()] * len(src_specs)
                tensor = do_spmd_comm(tensor, src_specs, tgt_specs)
                torch_name = self.compiled_meta.input_buffers_map.inv(node_name)
                buffers[torch_name] = tensor
            return buffers
        else:
            return {
                self.compiled_meta.input_buffers_map.inv(node_name): tensor
                for node_name, tensor in self.fw_gm.node_states[StateType.BUFFERS].items()
            }

    def optimizer_state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _optimizer_state_dict(self) -> Dict[str, Any]:
        if not self.has_step:
            raise RuntimeError(f"Optimizer wasn't compiled")
        optim_state = defaultdict(dict)
        if self.compiled_meta.tensors_spmd_strategies:
            for node_name, tensor in self.stage_step_gm.node_states[StateType.OPTIMSTATES].items():
                src_specs = self.compiled_meta.tensors_spmd_strategies[node_name]
                tgt_specs = [Replicate()] * len(src_specs)
                tensor = do_spmd_comm(tensor, src_specs, tgt_specs)
                torch_name, state_type = self.compiled_meta.input_optimstates_map.inv(node_name)
                optim_state[torch_name][state_type] = tensor
        else:
            for node_name, tensor in self.stage_step_gm.node_states[StateType.OPTIMSTATES].items():
                torch_name, state_type = self.compiled_meta.input_optimstates_map.inv(node_name)
                optim_state[torch_name][state_type] = tensor

        return dict(optim_state)


def split_by(traced: torch.fx.GraphModule, split_point: Callable):
    # avoid looking at next node by keeping track of previous split point
    prev_pipe_split_idx = -1
    pipe_split_nodes_to_erase = set()
    for i, node in enumerate(traced.graph.nodes):
        if (node.op, node.target) == ("call_function", split_point):
            if prev_pipe_split_idx == i - 1:
                pipe_split_nodes_to_erase.add(node)
            prev_pipe_split_idx = i

    for node in pipe_split_nodes_to_erase:
        traced.graph.erase_node(node)

    traced.recompile()

    part_idx = 0

    def split_callback(n: torch.fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == ("call_function", split_point):
            part_idx += 1
        return part_idx

    # split module
    split = ed_split_module(traced, None, split_callback, {})

    # remove pipe_split point
    for submodule in split.children():
        for node in submodule.graph.nodes:
            if (node.op, node.target) == ("call_function", split_point):
                assert len(node.args) == 1 and len(node.args[0]) == len(
                    node.users), f"split_point should have only one argument (list) or None, found {node} {node.args} {node.users}"
                tensor_list = node.args[0]
                to_erase = []
                for gi in node.users:
                    assert gi.op == "call_function" and gi.target == operator.getitem
                    gi_index = gi.args[1]
                    gi.replace_all_uses_with(tensor_list[gi_index])
                    to_erase.append(gi)
                for gi in to_erase:
                    submodule.graph.erase_node(gi)
                submodule.graph.erase_node(node)

        submodule.recompile()

    split.graph.eliminate_dead_code()
    split.delete_all_unused_submodules()
    split.graph.lint()
    split.recompile()

    return split, part_idx + 1


def _extract_step_subgraph_from_args(ed_gm: EDGraphModule, inputs_spec: Set[str]):
    new_graph = fx.Graph()
    env = {}
    outputs = []
    gm = ed_gm.gm
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if node.name in inputs_spec:  # only those in inputs
                env[node.name] = new_graph.placeholder(node.name)
        elif node.op == 'call_function':
            if node.target == operator.getitem:
                pass  # getitem is handled in foreach operators
            elif '_foreach_' == node.name[:
                                          9]:  # handle foreach operators
                list_args_kwargs_mask = []
                args = []
                for arg in node.args:
                    # foreach operators in torch/_C/_VariableFunctions.pyi.in
                    if isinstance(arg, (list, tuple)):  # list of Tensors
                        args.append([env[x.name] for x in arg if x.name in env])
                        list_args_kwargs_mask.append(tuple(x.name in env for x in arg))
                    elif isinstance(arg, torch.fx.Node):  # Tensors
                        args.append(env[arg.name])
                    else:  # Number or _complex
                        args.append(arg)

                kwargs = {}
                for kwarg_name, kwarg in node.kwargs.items():
                    if isinstance(kwarg, (list, tuple)):
                        kwargs[kwarg_name] = [env[x.name] for x in kwarg if x.name in env]
                        list_args_kwargs_mask.append(tuple(x.name in env for x in kwarg))
                    elif isinstance(kwarg, torch.fx.Node):
                        kwargs[kwarg_name] = env[kwarg.name]
                    else:
                        kwargs[kwarg_name] = kwarg

                assert len(set(list_args_kwargs_mask)
                           ) == 1, f"Input list of foreach operators should have the same mask, but found {list_args_kwargs_mask} {node} {node.args} {node.kwargs}"

                env[node.name] = new_graph.create_node(op='call_function',
                                                       name=node.name,
                                                       target=node.target,
                                                       args=tuple(args),
                                                       kwargs=kwargs)

                output_mask = list_args_kwargs_mask[0]
                getitem_idx = 0
                for getitem_user, kept in zip(ordered_gi_users(node), output_mask):
                    assert getitem_user.op == 'call_function' and getitem_user.target == operator.getitem
                    if kept:
                        env[getitem_user.name] = new_graph.create_node(op='call_function',
                                                                       name=getitem_user.name,
                                                                       target=operator.getitem,
                                                                       args=(env[node.name],
                                                                             getitem_idx))
                        getitem_idx += 1
            else:  # normal nodes like mul, add, etc.
                args = []
                for arg in node.args:
                    if isinstance(arg, (list, tuple)):  # list of Tensors
                        if any(isinstance(x, fx.Node) for x in arg):
                            assert all(isinstance(x, fx.Node) for x in arg)
                            args.append([env[x.name] for x in arg if x.name in env])
                        else:
                            args.append(arg)
                    elif isinstance(arg, torch.fx.Node):  # Tensor
                        if arg.name in env:
                            args.append(env[arg.name])
                    else:  # Number or _complex
                        args.append(arg)

                kwargs = {}
                for kwarg_name, kwarg in node.kwargs.items():
                    if isinstance(kwarg, (list, tuple)):
                        if any(isinstance(x, fx.Node) for x in kwarg):
                            assert all(isinstance(x, fx.Node) for x in kwarg)
                            kwargs[kwarg_name] = [env[x.name] for x in kwarg if x.name in env]
                        else:
                            kwargs[kwarg_name] = kwarg
                    elif isinstance(kwarg, torch.fx.Node):
                        if kwarg.name in env:
                            kwargs[kwarg_name] = env[kwarg.name]
                    else:
                        kwargs[kwarg_name] = kwarg

                if len(args) != len(node.args) or len(kwargs) != len(node.kwargs):
                    assert (not any(isinstance(arg, torch.fx.Node) for arg in args)) and (
                        not any(isinstance(kwarg, torch.fx.Node) for kwarg in kwargs.values())
                    ), f"This node shall be completed removed since it has no tensor args and kwargs {node} {args} {kwargs}"
                else:
                    env[node.name] = new_graph.create_node(op='call_function',
                                                           name=node.name,
                                                           target=node.target,
                                                           args=tuple(args),
                                                           kwargs=kwargs)
        elif node.op == 'output':
            for output in node.args[0]:  # output is a tuple (node.args[0])
                if output.name in env:
                    outputs.append(env[output.name])
        else:
            raise RuntimeError(f"op {node.op} not supported")

    new_graph.output(tuple(outputs))

    new_graph.eliminate_dead_code()
    new_graph.lint()
    new_gm = fx.GraphModule(gm, new_graph)

    injected_states = StateType.get_empty_dict()
    to_pop = []
    for name, val in ed_gm.node_states[StateType.OPTIMSTATES].items():
        if name in inputs_spec:
            injected_states[StateType.OPTIMSTATES][name] = val
            to_pop.append(name)
    for name in to_pop:
        ed_gm.node_states[StateType.OPTIMSTATES].pop(name)

    output_spec = [node.name for node in outputs]

    return EDGraphModule(new_gm, ed_gm.submod_type, inputs_spec, injected_states, output_spec,
                         ed_gm.call_module_users, 'partial_' + ed_gm.name)


def compile_pipeline(
    traced_stateless_func: fx.GraphModule,  # traced stateless function with split op
    nstages: int,  # number of stages, should be num_of_split_op * 2
    stateless_func_args,  # args for stateless function
    tensors_spmd_strategies: Optional[List[VarSPMDStrategy]] = None,
    strict=True  # report error if not all params and buffers are used
) -> Tuple[CompiledMeta, List[CompiledStage], fx.GraphModule, Set[str]]:
    '''
    This method split a graph module into multiple submodule according to split nodes traced in graph.
    It's specialized to deal with func:
        def stateless_func(func, module, opt, params, buffers, named_states, args, kwargs):
            with stateless._reparametrize_module(
                    cast(torch.nn.Module, module), {
                        **params,
                        **buffers
                    }, tie_weights=True) if module else nullcontext(), _rematerialize_optimizer(
                        opt, named_states, params) if opt else nullcontext():
                ret = func(*args, **kwargs)
            grads = {k: v.grad for k, v in params.items()}
            return params, buffers, named_states, grads, ret
    where func is a classical training step in pytorch:
        def train_step(input, model, opt):
            out = model(input)
            loss = out.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return out
    '''
    inputs_nodes_flatten = tuple(ph.name for ph in traced_stateless_func.graph.nodes
                                if ph.op == 'placeholder')
    inputs_nodes_unflatten = pytree.tree_unflatten(inputs_nodes_flatten,
                                                   traced_stateless_func._in_spec)

    output_nodes_flatten = tuple(node.name if node else None
                                 for node in list(traced_stateless_func.graph.nodes)[-1].args[0])
    output_nodes_unflatten = pytree.tree_unflatten(output_nodes_flatten,
                                                   traced_stateless_func._out_spec)

    # get spmd strategy if proviced
    if tensors_spmd_strategies:
        tensors_spmd_strategies = {
            ph_name: ph_strategy for ph_name, ph_strategy in zip(inputs_nodes_flatten, tensors_spmd_strategies)
        }

    # node name of input and output in stateless_func
    params, buffers, optimstates, args, kwargs = stateless_func_args
    input_params_nodes_unflatten, input_buffers_nodes_unflatten, input_optimstates_nodes_unflatten, args_nodes_unflatten, kwargs_nodes_unflatten = inputs_nodes_unflatten
    output_params_nodes_unflatten, output_buffers_nodes_unflatten, output_optimstates_nodes_unflatten, output_grads_unflatten, returns_nodes_unflatten = output_nodes_unflatten

    return_nodes_flatten, return_nodes_spec = pytree.tree_flatten(returns_nodes_unflatten)

    # one-to-one mapping between torch name and node name
    input_params_map = OneToOneMap.from_dict({
        torch_name: node_name for torch_name, node_name in input_params_nodes_unflatten.items()
    })
    input_buffers_map = OneToOneMap.from_dict({
        torch_name: node_name for torch_name, node_name in input_buffers_nodes_unflatten.items()
    })
    input_optimstates_map = OneToOneMap.from_dict({
        (torch_name, state_type): node_name for torch_name, states in input_optimstates_nodes_unflatten.items() for state_type, node_name in states.items()
    })
    optim_state_types = set(state_type for _, states in input_optimstates_nodes_unflatten.items() for state_type, _ in states.items())

    # given output node name of updated params/params, find corresponding input node name (and then find torch name)
    output_params_map = OneToOneMap.from_dict({
        torch_name: node_name for torch_name, node_name in output_params_nodes_unflatten.items()
    })
    output_buffers_map = OneToOneMap.from_dict({
        torch_name: node_name for torch_name, node_name in output_buffers_nodes_unflatten.items()
    })
    output_optimstates_map = OneToOneMap.from_dict({
        (torch_name, state_type): node_name for torch_name, states in output_optimstates_nodes_unflatten.items() for state_type, node_name in states.items()
    })
    output_grads_map = OneToOneMap.from_dict({
        torch_name: node_name for torch_name, node_name in output_grads_unflatten.items()
    })

    # split fw_bw and step
    splited_global, global_partition_cnt = split_by(traced_stateless_func,
                                        torch.ops.easydist.step_split.default)
    save_graphviz_dot(splited_global, "splited_global")
    assert global_partition_cnt <= 2 and global_partition_cnt == len(list(splited_global.children())), f"global_partition_cnt should be 1 (fw or fw_bw) or 2 (fw_bw + step), but found {global_partition_cnt}"
    states_used_by = defaultdict(list)

    def _extract_output(node):
        # process output
        outputs_spec = []
        call_module_users = defaultdict(set)
        getitem_users = [
            user.op == 'call_function' and user.target == operator.getitem for user in node.users
        ]
        if any(getitem_users):
            assert all(getitem_users), "Output shoule be tuple, which can be infered by ed_split_module"
            for gi in node.users:
                outputs_spec.append(gi.name)
                for gi_user in gi.users:
                    if gi_user.op == 'call_module':
                        call_module_users[gi.name].add(gi_user.name)
        else:  # output is tensor
            assert len(node.users) == 1, "Output should be tensor"
            user = next(iter(node.users))
            outputs_spec.append(user.name)
            for uuser in user.users:
                if uuser.op == 'call_module':
                    call_module_users[user.name].add(uuser.name)
        return outputs_spec, call_module_users

    def _extract_fw_submod(node, submod, stage_idx):
        save_graphviz_dot(submod, f"stage_{stage_idx}_fw")
        # process input
        inputs_spec, inputs_users = [], []
        injected_states = StateType.get_empty_dict()

        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            states_used_by[arg.name].append(node.name)
            try:
                if arg.name in input_params_map.inv_keys():
                    injected_states[StateType.PARAMS][arg.name] = params.pop(input_params_map.inv(arg.name))
                elif arg.name in input_buffers_map.inv_keys():  # inject states to the first submod
                    injected_states[StateType.BUFFERS][arg.name] = buffers.pop(
                        input_buffers_map.inv(arg.name))
            except KeyError:
                name = input_params_map.inv(arg.name) if arg.name in input_params_map.inv_keys() else input_buffers_map.inv(arg.name)
                typ = StateType.PARAMS if arg.name in input_params_map.inv_keys() else StateType.BUFFERS
                raise RuntimeError(
                    f"{typ}: {name} ({arg.name}) is found used by multiple forward submods {states_used_by[arg.name]}"
                )

        # process output
        outputs_spec, call_module_users = _extract_output(node)

        return EDGraphModule(submod, SubmodType.FW, inputs_spec, injected_states, outputs_spec,
                             call_module_users, node.target)

    def _extract_bw_submod(node, submod, stage_idx):
        save_graphviz_dot(submod, f"stage_{stage_idx}_bw")
        # process input
        inputs_spec, inputs_users = [], []
        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            states_used_by[arg.name].append(node.name)
        # process output
        outputs_spec, call_module_users = _extract_output(node)
        injected_states = StateType.get_empty_dict()
        return EDGraphModule(submod, SubmodType.BW, inputs_spec, injected_states, outputs_spec,
                             call_module_users, node.target)

    def _extract_step_submod(node, submod):
        # process input
        inputs_spec, inputs_users = [], []

        injected_states = StateType.get_empty_dict()
        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            states_used_by[arg.name].append(node.name)
            if arg.name in input_optimstates_map.inv_keys():  # inject states to the first submod (might be fw_gm or step_gm but not bw_gm)
                try:
                    torch_name, state_type = input_optimstates_map.inv(arg.name)
                    injected_states[StateType.OPTIMSTATES][arg.name] = optimstates[torch_name].pop(state_type)
                except KeyError:
                    torch_name, state_type = input_optimstates_map.inv(arg.name)
                    typ = StateType.OPTIMSTATES
                    raise RuntimeError(f"Please report this, {typ}:{torch_name} ({state_type}) is found used by multiple step submods {states_used_by[arg.name]}")

        # process output
        outputs_spec, call_module_users = _extract_output(node)

        return EDGraphModule(submod, SubmodType.STEP, inputs_spec, injected_states, outputs_spec,
                             call_module_users, 'step')

    has_backward = get_backward_flag()
    fw_and_bw_gm, stage_cnt = split_by(
                    splited_global.submod_0, torch.ops.easydist.fw_bw_split.default)
    save_graphviz_dot(fw_and_bw_gm, 'fw_and_bw_gm')
    if not (stage_cnt == (nstages * 2 if has_backward else nstages)):
        raise RuntimeError(
            f"Backward is called and there should be 2 * {nstages} submodule (found {stage_cnt}), please check split annotation or report this bug" if has_backward
            else f"Backward is not called and there should be {nstages} submodule (found {stage_cnt}), please check split annotation or report this bug"
        )

    stateful_fw_bw = []
    submod_idx = 0
    for n in fw_and_bw_gm.graph.nodes:  # for each submod in fw_and_bw_gm
        if n.op == 'call_module':  # extract stateful submods
            if (not has_backward) or (submod_idx < nstages):  # forward
                stage_idx = submod_idx
                submod = getattr(fw_and_bw_gm, n.target)
                stateful_fw_bw.append(_extract_fw_submod(n, submod, stage_idx))
            else:  # backward
                stage_idx = 2 * nstages - submod_idx - 1
                submod = getattr(fw_and_bw_gm, n.target)
                stateful_fw_bw.append(_extract_bw_submod(n, submod, stage_idx))
            submod_idx += 1

    step_gm_global = getattr(splited_global, 'submod_1', None)
    input_node_to_step_input_params = OneToOneMap.from_dict({})
    input_node_to_step_input_grads = OneToOneMap.from_dict({})
    if step_gm_global:
        save_graphviz_dot(step_gm_global, 'step_gm_global')
        step_gm_global = _extract_step_submod([n for n in splited_global.graph.nodes if n.name == 'submod_1'][0], step_gm_global)
        step_gm_phs = [node.name for node in splited_global.submod_1.graph.nodes if node.op == 'placeholder']
        input_params_nodes_flatten, _ = pytree.tree_flatten(input_params_nodes_unflatten)
        num_params = len(input_params_nodes_flatten)
        input_node_to_step_input_params = OneToOneMap.from_dict({
            input_node_name: step_node_name for input_node_name, step_node_name in zip(input_params_nodes_flatten, step_gm_phs[:num_params])
        })
        input_node_to_step_input_grads = OneToOneMap.from_dict({
            input_node_to_step_input_params.inv(param_node_name): grad_node_name for param_node_name, grad_node_name in zip(step_gm_phs[:num_params], step_gm_phs[num_params:2*num_params])
        })

    # meta data
    compiled_meta = CompiledMeta(
        nstages=nstages,
        in_spec=traced_stateless_func._in_spec,
        out_spec=traced_stateless_func._out_spec,
        output_nodes_flatten=output_nodes_flatten,
        args_nodes_unflatten=args_nodes_unflatten,
        kwargs_nodes_unflatten=kwargs_nodes_unflatten,
        input_params_map=input_params_map,
        input_buffers_map=input_buffers_map,
        input_optimstates_map=input_optimstates_map,
        optim_state_types=optim_state_types,
        input_node_to_step_input_params=input_node_to_step_input_params,
        input_node_to_step_input_grads=input_node_to_step_input_grads,
        output_params_map=output_params_map,
        output_buffers_map=output_buffers_map,
        output_optimstates_map=output_optimstates_map,
        output_grads_map=output_grads_map,
        tensors_spmd_strategies=tensors_spmd_strategies,
        return_nodes_flatten=return_nodes_flatten,
        return_nodes_spec=return_nodes_spec,
    )

    compiled_stages: List[CompiledStage] = []
    for fw_gm, bw_gm in zip(stateful_fw_bw[:nstages], reversed(stateful_fw_bw[nstages:])):
        compiled_stage = CompiledStage(compiled_meta,
                                        fw_gm,
                                        bw_gm,
                                        step_gm_global
                                        )
        compiled_stages.append(compiled_stage)

    # post check on params, buffers, optimstates
    erased_tensor_keys = set(params.keys()) | set(buffers.keys()) | set(
        k for k, v in optimstates.items() if v)
    if len(erased_tensor_keys) > 0:
        debugging_info = textwrap.dedent(f"""
            Some states will be erased, please make sure this is intended behaviour
            Erased:
                Params:
                    {' '.join(params)}
                Buffers:
                    {' '.join(buffers)}
                Optimstates:
                    {' '.join(k for k, v in optimstates.items() if len(v) > 0)}
            """)
        if strict:
            raise RuntimeError(debugging_info)
        else:
            logging.warning(debugging_info)

    # construct a local graph representing pipeline
    g = fx.Graph()
    env = {}
    for node_name in pytree.tree_flatten([args_nodes_unflatten, kwargs_nodes_unflatten])[0]:
        env[node_name] = g.placeholder(node_name)
        submod_idx = 0
    for node in fw_and_bw_gm.graph.nodes:
        if node.op == 'call_module':
            if (not has_backward) or (submod_idx < nstages):  # forward
                stage_idx = submod_idx
                stage = compiled_stages[stage_idx]
                out_dict = g.create_node(
                    name=f'stage_{stage_idx}_fw',
                    op='call_function',
                    target=stage.forward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.fw_func_args})
                out_dict.meta['stage_idx'] = stage_idx
                for output in stage.fw_func_returns:
                    env[output] = g.create_node(name=output,
                                                op='call_function',
                                                target=operator.getitem,
                                                args=(out_dict, output))
            else:  # backward
                stage_idx = 2 * nstages - submod_idx - 1
                stage = compiled_stages[stage_idx]
                out_dict = g.create_node(
                    name=f'stage_{stage_idx}_bw',
                    op='call_function',
                    target=stage.backward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.bw_func_args})
                out_dict.meta['stage_idx'] = stage_idx
                for output in stage.bw_func_returns:
                    if not output in output_nodes_flatten:
                        env[output] = g.create_node(name=output,
                                                    op='call_function',
                                                    target=operator.getitem,
                                                    args=(out_dict, output))
                if stage.has_step:
                    g.create_node(name=f'stage_{stage_idx}_step',
                                    op='call_function',
                                    target=stage.step)
            submod_idx += 1

    def gather_outputs():
        outputs = reduce(lambda cur, stage: {**cur,  **stage.returns}, compiled_stages, {})
        outputs = [outputs[node_name] for node_name in compiled_meta.return_nodes_flatten]
        return pytree.tree_unflatten(outputs, compiled_meta.return_nodes_spec)

    def eliminate_dead_code():
        raise RuntimeError("This method should be called since the graph doesn't have output node")

    setattr(g, 'eliminate_dead_code', eliminate_dead_code)
    g.output(g.call_function(gather_outputs))
    local_gm = fx.GraphModule({}, g)

    save_graphviz_dot(local_gm, f'pp_local_gm')

    return compiled_meta, compiled_stages, local_gm, erased_tensor_keys
