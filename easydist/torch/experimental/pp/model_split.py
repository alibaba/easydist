'''
Adapted from https://github.com/pytorch/PiPPy/blob/83a2308f4a53ae36eba2f0c1b2b262d5d697d37b/pippy/IR.py#L280
'''
import copy
import itertools
import logging
import operator
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast)
from unittest.mock import patch

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import Tensor
from torch._decomp.decompositions_for_rng import (PhiloxStateTracker, rng_decompositions)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import (dynamo_timed, lazy_format_graph_code, preserve_rng_state)
from torch._guards import detect_fake_mode, tracing
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module
from torch.nn.parameter import Parameter
from torch.nn.utils import stateless
from torchviz import make_dot

from easydist.torch.compiler import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp import (get_pipeline_tracer, set_pipeline_tracer)
from easydist.torch.experimental.pp.backward import (BackStage, _insert_stage_symbolic_backward)
from easydist.torch.experimental.pp.get_activations import get_activations
from easydist.torch.experimental.pp.loss_wrapper import TrivialLossWrapper
from easydist.torch.utils import _enable_compile


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


def save_dot(dot, fname):
    with open(fname, 'w') as f:
        f.write(dot.source)


MultiUseParamSpec = Union[MultiUseParameterConfig, Dict[str, MultiUseParameterConfig]]


def pipe_split():
    pipeline_tracer = get_pipeline_tracer()
    if pipeline_tracer is not None and hasattr(pipeline_tracer, "graph"):
        pipeline_tracer.graph.call_function(pipe_split, (), {})


class PipeSplitWrapper(torch.nn.Module):

    class SplitPoint(Enum):
        BEGINNING = 1
        END = 2

    def __init__(
        self,
        mod: torch.nn.Module,
        split_point: SplitPoint = SplitPoint.BEGINNING,
    ):
        super().__init__()
        self.mod = mod
        self.split_point = split_point

    def forward(self, *args, **kwargs):
        try:
            if self.split_point == self.SplitPoint.BEGINNING:
                pipe_split()

            return self.mod(*args, **kwargs)
        finally:
            if self.split_point == self.SplitPoint.END:
                pipe_split()


def annotate_split_points(mod: torch.nn.Module, spec: Dict[str, PipeSplitWrapper.SplitPoint]):
    # TODO: make this implementation out-of-place?
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        predecessor_module = mod
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(
                    f'Specified target {qualname} referenced nonexistent module {".".join(atoms[:i+1])}'
                )

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        wrapped_mod = PipeSplitWrapper(mod_to_wrap, split_type)
        setattr(predecessor_module, atoms[-1], wrapped_mod)


def _find_loss_from_output_and_spec(output_val, spec_val):
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, torch.fx.Node):
            raise RuntimeError(f"Loss spec must specify a dynamic value but got {output_val}")
        return output_val

    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(f"Output value {output_val} must match type of loss specification "
                               f"{spec_val}")
        if len(output_val) != len(spec_val):
            raise RuntimeError(
                f"Output value {output_val} must match length of loss specification "
                f"{spec_val}")
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(f"Output value {output_val} must match type of loss specification "
                               f"{spec_val}")
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(f"Output value {output_val} must match keys of loss specification "
                               f"{spec_val}")
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(output_val[k], spec_val[k])
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    raise RuntimeError(f"Unsupported type {type(spec_val)} in loss specification")


def _number_and_count_forward_stages(gm: torch.fx.GraphModule):
    num_stages = 0
    found_idxs: Dict[int, None] = {}
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith("submod_"):
            node.meta["stage_idx"] = int(node.target[len("submod_"):])
            found_idxs.setdefault(node.meta["stage_idx"])
            num_stages += 1

    # this assert will fail if a split point is inserted before the first layer, which creates empty first submodule
    assert all(i in found_idxs for i in range(num_stages))

    return num_stages


def _find_loss_output(mod: torch.nn.Module, g: torch.fx.Graph, output_loss_value_spec):
    output_nodes = [n for n in g.nodes if n.op == "output"]
    assert len(output_nodes) == 1
    output_node = output_nodes[0]
    output_val = output_node.args[0]
    generated_spec: Any = None

    if isinstance(mod, TrivialLossWrapper):
        # TrivialLossWrapper is pre-defined by PiPPy.
        # It has loss as the only output so we can safely assume the first output arg is the loss.
        assert len(output_node.args) == 1
        loss_node = output_val
        generated_spec = TrivialLossWrapper.loss_spec
    elif output_loss_value_spec is None:
        # Use default spec, i.e. search for "loss" in output values
        if isinstance(output_val, dict) and "loss" in output_val.keys():
            loss_node = output_val["loss"]
            generated_spec = {k: k == "loss" for k in output_val}
        else:
            loss_node = None
            generated_spec = None
    else:
        loss_node = _find_loss_from_output_and_spec(output_val, output_loss_value_spec)
        generated_spec = output_loss_value_spec

    return loss_node, output_node, generated_spec


def _from_traced(
    mod: torch.nn.Module,
    traced: torch.fx.GraphModule,
    multi_use_param_spec: Optional[MultiUseParamSpec] = None,
    output_loss_value_spec=None,
    return_to_0: bool = True,
):
    """
    Additionally, the ``output_loss_value_spec`` value can be specified to disambiguate
    which value in the output of `forward` is the loss value on which PiPPy should apply
    backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
    you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
    a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
    ``output_loss_value_spec={'loss': True, 'model_out': False}``
    """

    # Deduplicate `get_attr` nodes that refer to the same parameter . Downstream code for moving
    # parameters relies on the invariant that parameter accesses happen once. This is not necessarily
    # the case (especially with custom tracers), so fix that up here.
    get_attr_nodes: Dict[str, torch.fx.Node] = {}
    for node in traced.graph.nodes:
        if node.op == "get_attr":
            get_attr_nodes.setdefault(node.target, node)

            if get_attr_nodes[node.target] != node:
                node.replace_all_uses_with(get_attr_nodes[node.target])
                traced.graph.erase_node(node)

    # avoid looking at next node by keeping track of previous pipe_split
    prev_pipe_split_idx = -1
    pipe_split_nodes_to_erase = set()
    for i, node in enumerate(traced.graph.nodes):
        if (node.op, node.target) == ("call_function", pipe_split):
            if prev_pipe_split_idx == i - 1:
                pipe_split_nodes_to_erase.add(node)
            prev_pipe_split_idx = i

    for node in pipe_split_nodes_to_erase:
        traced.graph.erase_node(node)

    traced.recompile()

    part_idx = 0

    def split_callback(n: torch.fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == ("call_function", pipe_split):
            part_idx += 1
        return part_idx

    # Ask split_module to return mapping from new qualname to old qualname
    qualname_map: Dict[str, str] = {}
    # TODO: what does split do with module invocations? does it move the modules
    # into the submodules?
    split = split_module(traced, mod, split_callback, qualname_map)
    # a (custom) tracer can produce dead code like orphan get_attr nodes
    split.graph.eliminate_dead_code()

    # peephole to remove pipe_split
    for submodule in split.modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ("call_function", pipe_split):
                    submodule.graph.erase_node(node)
            submodule.recompile()

    # lift single-use parameter fetches into the modules that use them
    # TODO: backport this into split_module
    def delete_user_reference(node, user, delete_node=True):
        assert len(user.kwargs) == 0
        use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
        assert len(use_idxs) == 1
        args_copy = list(user.args)
        args_copy.pop(use_idxs[0])
        user.args = tuple(args_copy)
        if delete_node:
            node.graph.erase_node(node)

        return use_idxs[0]

    def move_param_to_callee(root, callee_name, param_val, use_idx, is_buffer):
        assert isinstance(param_val, torch.Tensor), (
            f"Expected '{node.target}' to be {torch.Tensor} but got {type(param_val)}." +
            (f" It might happen if module '{node.target}' was passed to some 'leaf function'"
             f"(see https://pytorch.org/docs/stable/fx.html#torch.fx.wrap). Please inspect "
             f"usages of '{node.target}' in the traced graph." if isinstance(
                 param_val, torch.nn.Module) else ""))
        callee = root.get_submodule(callee_name)
        new_param_name = f"moved_{node.target.replace('.', '_')}"
        assert not hasattr(
            callee,
            new_param_name), f"Module {callee_name} already has a parameter named {new_param_name}"
        if is_buffer:
            callee.register_buffer(new_param_name, param_val)
        else:
            setattr(callee, new_param_name, param_val)

        # Update qualname mapping
        # New qualname will have submodule prefix
        new_qualname = f"{callee_name}.{new_param_name}"
        if node.target in qualname_map:
            # Just in case the target name is already in the qualname_map
            # returned by split_module() -- we update the mapping using the
            # new name as a new key
            qualname_map[new_qualname] = qualname_map.pop(node.target)
        else:
            qualname_map[new_qualname] = node.target

        ph_counter = 0
        for sn in callee.graph.nodes:
            if sn.op == "placeholder":
                if ph_counter == use_idx:
                    with callee.graph.inserting_before(sn):
                        get_attr = callee.graph.get_attr(new_param_name)
                        sn.replace_all_uses_with(get_attr)
                        callee.graph.erase_node(sn)
                ph_counter += 1
        callee.graph.lint()
        callee.recompile()

        return get_attr

    to_delete = list()  # a list of nodes for deferral deletion

    for node in split.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 1:
            user = list(node.users)[0]
            assert user.op == "call_module"
            use_idx = delete_user_reference(node, user)

            # Move parameter into submodule and replace PH with a get_attr
            atoms = node.target.split(".")
            mod_itr = split
            for atom in atoms[:-1]:
                mod_itr = getattr(mod_itr, atom)
            param_val = getattr(mod_itr, atoms[-1])
            is_buffer = atoms[-1] in mod_itr._buffers

            move_param_to_callee(split, user.target, param_val, use_idx, is_buffer)

            to_delete.append((mod_itr, atoms))

    # deferral deletion
    for mod_itr, atoms in to_delete:
        delattr(mod_itr, atoms[-1])

    split.graph.lint()
    split.recompile()

    # Handle multi-use parameters based on user's configuration
    # TODO: generalize this to sequential
    multi_use_param_spec = multi_use_param_spec or {}

    multi_use_params_qualnames: Dict[str, Optional[MultiUseParameterConfig]] = {}
    for node in split.graph.nodes:
        if node.op == "get_attr" and len(node.users) > 1:
            multi_use_params_qualnames.setdefault(node.target)

    for param in multi_use_params_qualnames:
        if isinstance(multi_use_param_spec, MultiUseParameterConfig):
            multi_use_params_qualnames[param] = multi_use_param_spec
        elif isinstance(multi_use_param_spec, dict):
            multi_use_params_qualnames[param] = multi_use_param_spec.get(
                param, MultiUseParameterConfig.TRANSMIT)
        else:
            raise ValueError("multi_use_param_spec must be MultiUseParamSpec enum or dict")

    # TODO: do we maintain the invariant that `Node.users` is topologically ordered? I don't think so
    node_to_first_user: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in split.graph.nodes:
        for input in node.all_input_nodes:
            if input not in node_to_first_user:
                node_to_first_user[input] = node

    for node in split.graph.nodes:
        if (node.op == "get_attr" and node.target in multi_use_params_qualnames):
            reuse_type = multi_use_params_qualnames[node.target]
            if reuse_type == MultiUseParameterConfig.TRANSMIT:
                first_user = node_to_first_user[node]
                assert first_user.op == "call_module"

                use_idx = delete_user_reference(node, first_user, delete_node=False)

                atoms = node.target.split(".")
                mod_itr = split
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                param_val = getattr(mod_itr, atoms[-1])
                is_buffer = atoms[-1] in mod_itr._buffers

                callee_param_def = move_param_to_callee(split, first_user.target, param_val,
                                                        use_idx, is_buffer)

                delattr(mod_itr, atoms[-1])

                # Add extra output to the callee and switch references to the parameter
                # access in the pipeline graph to use this.
                submod = split.get_submodule(first_user.target)
                callee_output_nodes = [n for n in submod.graph.nodes if n.op == "output"]
                assert len(callee_output_nodes) == 1
                callee_output_node = callee_output_nodes[0]

                # TODO: zero outputs?
                if isinstance(callee_output_node.args[0], tuple):
                    new_output_args = callee_output_node.args[0] + (callee_param_def, )
                    callee_output_node.args = (new_output_args, )
                    new_output_idx = len(new_output_args) - 1
                    promoted_to_tuple = False
                else:
                    new_output_args = (
                        callee_output_node.args[0],
                        callee_param_def,
                    )
                    callee_output_node.args = (new_output_args, )
                    new_output_idx = len(new_output_args) - 1
                    promoted_to_tuple = True

                submod.graph.lint()
                submod.recompile()

                with split.graph.inserting_after(first_user):
                    if promoted_to_tuple:
                        # TODO: test this code path
                        orig_output_getitem = split.graph.call_function(
                            operator.getitem, (first_user, 0))
                        first_user.replace_all_uses_with(orig_output_getitem)
                        # HACK because the above replace_all_uses with ALSO replaced the instance
                        # of first_user within the getitem node we just added
                        orig_output_getitem.args = (first_user, ) + orig_output_getitem.args[1:]

                    transmitted_value_getitem = split.graph.call_function(
                        operator.getitem, (first_user, new_output_idx))
                    node.replace_all_uses_with(transmitted_value_getitem)
                    split.graph.erase_node(node)
            elif reuse_type == MultiUseParameterConfig.REPLICATE:
                for user in copy.copy(node.users):
                    use_idx = delete_user_reference(node, user, delete_node=False)
                    atoms = node.target.split(".")
                    mod_itr = split
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)
                    param_val = getattr(mod_itr, atoms[-1])
                    is_buffer = atoms[-1] in mod_itr._buffers

                    move_param_to_callee(split, user.target, param_val, use_idx, is_buffer)

                atoms = node.target.split(".")
                mod_itr = split
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)

                delattr(mod_itr, atoms[-1])

                split.graph.erase_node(node)
            else:
                raise ValueError(
                    f"Unknown multi-use config value {reuse_type} specified for {node.target}")

    split.delete_all_unused_submodules()

    split.graph.lint()
    split.recompile()

    # num_stages = _number_and_count_forward_stages(split)

    # has_loss_and_backward = False
    # generated_loss_spec = output_loss_value_spec

    # if mod.training or output_loss_value_spec is not None:
    #     loss_node, output_node, generated_loss_spec = _find_loss_output(
    #         mod, split.graph, output_loss_value_spec)
    #     if loss_node is not None:
    #         _insert_stage_symbolic_backward(split.graph, loss_node, output_node, return_to_0,
    #                                         num_stages)
    #         split.recompile()
    #         has_loss_and_backward = True
    #         logging.info("Pipeline is in training mode, backward pass generated")
    #     else:
    #         logging.warning(
    #             "Did not find any loss value from model output, your pipeline will be in inference mode. "
    #             "If you want your pipeline to be in training mode, please specify a loss value via "
    #             "`output_loss_value_spec`.")
    # else:
    #     logging.info("Pipeline is in evaluation mode, backward pass not generated")

    return split


def run_local_split_gm(split_gm, *args):
    executor = DetachExecutor(split_gm)
    return executor.run(*args)


class EDCompiledForward(torch.nn.Module):

    def __init__(self, stateless_forward, params, buffers) -> None:
        super().__init__()
        self.stateless_forward = [stateless_forward]
        self.orig_map = {}
        self.update_params(params)
        self.update_buffers(buffers)

    def map_name(self, name):
        return name.replace('.', '_')

    def map_states(self, iter: Iterator[Tuple[str, Tensor]]) -> Iterator[Tuple[str, Tensor]]:
        return map(lambda pair: (self.map_name(pair[0]), pair[1]), iter)

    def update_params(self, params):
        for pname, param in params.items():
            mapped_name = self.map_name(pname)
            self.orig_map[mapped_name] = pname
            self.register_parameter(mapped_name, param)

    def update_buffers(self, buffers):
        for bname, buffer in buffers.items():
            mapped_name = self.map_name(bname)
            self.orig_map[mapped_name] = bname
            self.register_buffer(mapped_name, buffer)

    def named_parameters_orig(self,
                              prefix: str = '',
                              recurse: bool = True,
                              remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        iter_unmapped = super().named_parameters(prefix, recurse, remove_duplicate)
        return map(lambda pair: (self.orig_map[pair[0]], pair[1]), iter_unmapped)

    def named_buffers_orig(self,
                           prefix: str = '',
                           recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        iter_unmapped = super().named_buffers(prefix, recurse)
        return map(lambda pair: (self.orig_map[pair[0]], pair[1]), iter_unmapped)

    def __call__(self, *args, **kwargs):
        params = dict(self.named_parameters_orig())
        buffers = dict(self.named_buffers_orig())

        output, buffers = self.stateless_forward[0](params, buffers, args, kwargs)

        self.update_buffers(buffers)

        return output


class EDCompiledBackward(nn.Module):
    ed_forward: List[EDCompiledForward]

    def __init__(self, ed_forward: EDCompiledForward, stateless_backward: nn.Module):
        super().__init__()
        # hide children modules
        self.ed_forward = [ed_forward]
        self.stateless_backward = [stateless_backward]

    def __call__(self, **kwargs):
        params = dict(self.ed_forward[0].named_parameters_orig())
        buffers = dict(self.ed_forward[0].named_buffers_orig())
        save_dot(make_dot(kwargs['stage_output'], params, True, True), 'runtime.dot')
        activations = get_activations(kwargs["stage_output"], kwargs['outputs_with_grads_idxs'],
                                      params, buffers)

        grad_inputs, barrier_token, grads = self.stateless_backward[0](params, buffers,
                                                                       list(activations.values()),
                                                                       kwargs)

        for pname in params:
            params[pname].grad = grads[pname]

        self.ed_forward[0].update_params(params)

        return grad_inputs, barrier_token


def trace_backward(tracing_mode: str, ed_forward: EDCompiledForward,
                   maybe_faked_params: Dict[str,
                                            nn.Parameter], maybe_faked_buffers: Dict[str,
                                                                                     torch.Tensor],
                   activations, backward_mod: BackStage, kwargs: Dict):
    ret_for_interpreter = None

    def stateless_backward(params, buffers, activations, kwargs):
        nonlocal ret_for_interpreter
        grad_inputs, barrier_token = backward_mod(**kwargs)
        ret_for_interpreter = {
            "tracing_output": (grad_inputs, barrier_token),
        }
        grads = {k: v.grad for k, v in params.items()}
        return grad_inputs, barrier_token, grads

    with _enable_compile():
        # args only
        gm = make_fx(stateless_backward,
                     tracing_mode=tracing_mode,
                     decomposition_table=EASYDIST_DECOMP_TABLE,
                     _allow_non_fake_inputs=False)(maybe_faked_params, maybe_faked_buffers,
                                                   activations, kwargs)

    # TODO @botbw: get_activations depends on unoptimized graph (run once more after tracing?)
    # gm.graph.eliminate_dead_code()
    # gm = preprocess_traced_graph(gm)
    # gm.recompile()

    return ret_for_interpreter, EDCompiledBackward(ed_forward, gm)


def trace_forward(tracing_mode: str, module: nn.Module, args: Tuple, kwargs: Dict):
    # TODO @botbw: better way to do this
    ret_for_interpreter: Dict[str, Any] = {}

    def stateless_forward(params, buffers, args, kwargs):
        nonlocal ret_for_interpreter
        with stateless._reparametrize_module(cast(torch.nn.Module, module), {
                **params,
                **buffers
        },
                                             tie_weights=True):
            output = module(*args, **kwargs)

        # need to get the faked tensors for backward
        ret_for_interpreter = {
            "tracing_output": output,
            "maybe_faked_params": params,
            "maybe_faked_buffers": buffers,
            "maybe_faked_args": args,
            "maybe_faked_kwargs": kwargs
        }

        return output, buffers

    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    with _enable_compile():
        gm = make_fx(stateless_forward,
                     tracing_mode=tracing_mode,
                     decomposition_table=EASYDIST_DECOMP_TABLE,
                     _allow_non_fake_inputs=False)(params, buffers, args, kwargs)

    # TODO @botbw: get_activations depends on unoptimized graph (run once more after tracing?)
    # gm.graph.eliminate_dead_code()
    # gm = preprocess_traced_graph(gm)
    # gm.recompile()

    return ret_for_interpreter, EDCompiledForward(gm, params, buffers)


from torch._functorch.aot_autograd import (AOT_COUNTER, AOTConfig, create_aot_dispatcher_function,
                                           create_tree_flattened_fn,
                                           run_functionalized_fw_and_collect_metadata,
                                           track_graph_compiling, PytreeThunk,
                                           create_runtime_wrapper, OutputType,
                                           call_func_with_args, TensorAlias, functionalized_rng_runtime_epilogue)
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch._functorch.partitioners import default_partition
from torch._prims_common import CUDARngStateHelper
from torch._functorch import config

def compile_submod(module, args, kwargs):
    named_params = dict(module.named_parameters())
    named_buffers = dict(module.named_buffers())
    num_params_buffers = len(named_params) + len(named_buffers)

    args = (named_params, named_buffers, *args)

    def fn(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(module, params_and_buffers, args, kwargs)

    flat_args, _ = pytree.tree_flatten((args, kwargs))
    flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)

    aot_config = AOTConfig(
        is_export=True, # to produce fx_g
        decompositions=EASYDIST_DECOMP_TABLE,
        partition_fn=default_partition,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=False,
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        dynamic_shapes=False,
        aot_autograd_arg_pos_to_source=None,
        no_tangents=False,
        enable_log=False,
    )

    fx_g, fw_metadata = create_aot_dispatcher_function(
        flat_fn,
        flat_args,
        aot_config,
    )

    # Copied from aot_dispatch_autograd_graph.
    traced_tangents = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        fw_metadata.traced_tangents,
    )
    joint_inputs = (flat_args, traced_tangents)

    with torch.no_grad():
        with track_graph_compiling(aot_config, "joint"):
            num_inner_fwd_outputs = (fw_metadata.num_mutated_inputs + fw_metadata.num_outputs +
                                     fw_metadata.num_intermediate_bases +
                                     fw_metadata.num_outputs_rng_offset)
            fw_module, bw_module = aot_config.partition_fn(fx_g,
                                                           joint_inputs,
                                                           num_fwd_outputs=num_inner_fwd_outputs)
            fw_outs = [n for n in fw_module.graph.nodes if n.op == "output"][0].args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            symint_outs_saved_for_bw = [n for n in fw_outs_saved_for_bw if is_sym_node(n)]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

    _indices_of_inps_to_detach = []
    bw_outs = [n for n in bw_module.graph.nodes if n.op == "output"][0].args[0]
    assert len(bw_outs) == len(fw_metadata.input_info) + fw_metadata.num_outputs_rng_offset
    for i, (bw_out) in enumerate(bw_outs):
        if bw_out is None:
            _indices_of_inps_to_detach.append(i)

    class CompiledForward:
        def __init__(self):
            super().__init__()
            self.fw_gm = fw_module
            self.fw_metadata = fw_metadata
            self.named_params = named_params
            self.named_buffers = named_buffers
            self.runtime_fw = create_runtime_wrapper(
                    self.forward,
                    runtime_metadata=fw_metadata,
                    indices_of_inps_to_detach=_indices_of_inps_to_detach,
                    trace_joint=True,
                    keep_input_mutations=aot_config.keep_inference_input_mutations,
                    disable_amp=torch._C._is_any_autocast_enabled()
                )

        def init_before_forward(self):
            self.saved_tensors = []
            self.symints = None
            self.non_differentiable = None

        def __call__(self, *args, **kwargs):
            self.init_before_forward()
            args = (named_params, named_buffers, *args)
            in_flat, in_spec = pytree.tree_flatten((args, kwargs))
            out_flat = self.runtime_fw(*in_flat)
            return out_spec.unflatten(out_flat)
        
        def forward(self, *deduped_flat_tensor_args):
            args = deduped_flat_tensor_args
            if self.fw_metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                args = (*args, seed, offset)
            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
            # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
            #   of the original view, and not the synthetic base
            fw_outs = call_func_with_args(
                self.fw_gm.forward,
                args,
                disable_amp=torch._C._is_any_autocast_enabled(),
            )

            num_outputs = self.fw_metadata.num_outputs
            num_outputs_aliased = self.fw_metadata.num_outputs_aliased
            num_intermediate_bases = self.fw_metadata.num_intermediate_bases
            num_symints_saved_for_bw = _num_symints_saved_for_bw
            num_mutated_inputs = self.fw_metadata.num_mutated_inputs
            num_mutated_metadata_only_inputs = (
                self.fw_metadata.num_mutated_metadata_only_inputs
            )
            num_forward_returns = self.fw_metadata.num_forward_returns
            num_forward = self.fw_metadata.num_forward

            assert num_forward_returns == len(
                self.fw_metadata.requires_grad_info
            ) + num_intermediate_bases

            # Partitioners must put symint arguments at the end separate from tensor arguments
            tensors_saved_for_backwards = fw_outs[
                self.fw_metadata.tensors_saved_for_backwards_slice
            ]
            assert all(
                isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards
            )
            # See Note [Detaching saved tensors in AOTAutograd]
            self.saved_tensors = (x.detach() if x._is_view() else x for x in tensors_saved_for_backwards)
            symint_outs = fw_outs[self.fw_metadata.symints_saved_for_backwards_slice]
            assert all(
                isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                for x in symint_outs
            ), str([type(x) for x in symint_outs])
            self.symints = symint_outs

            raw_returns = fw_outs[0:num_forward_returns]

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            if num_mutated_metadata_only_inputs > 0:
                for i, idx in enumerate(
                    self.fw_metadata.mutated_inp_indices
                ):
                    # We could make this faster by only looping over inputs with metadata-only mutations
                    # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                    info = self.fw_metadata.input_info[idx]
                    if info.mutates_metadata and not info.mutates_data:
                        raw_returns[i] = TensorAlias(raw_returns[i])

                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_inputs]
                    mut_inp_infos = [
                        x for x in self.fw_metadata.input_info if x.mutates_data or x.mutates_metadata
                    ]
                    assert len(user_mutated_inputs_raw) == len(mut_inp_infos)

            if self.fw_metadata.num_unsafe_view_outputs > 0:
                for idx in self.fw_metadata.unsafe_view_out_indices:
                    raw_return_idx = num_mutated_inputs + idx
                    o = raw_returns[raw_return_idx]
                    raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(o, o.shape)

            if num_outputs_aliased > 0:
                for idx in self.fw_metadata.aliased_out_indices:
                    raw_return_idx = num_mutated_inputs + idx
                    raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])

                if config.debug_assert:
                    intermediates_raw = raw_returns[num_mutated_inputs + num_outputs:]
                    assert not any(isinstance(x, TensorAlias) for x in intermediates_raw)

            # invariant: intermediate bases always require gradients, so we don't have to
            # consider marking them as non-differentiable.
            raw_returns_not_including_intermediate_bases = raw_returns[:num_mutated_inputs + num_outputs]
            fw_outs_not_requiring_grad = [
                x
                for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
                if isinstance(x, torch.Tensor)
                and not self.fw_metadata.requires_grad_info[i]
            ]
            self.non_differentiable = fw_outs_not_requiring_grad

            functionalized_rng_runtime_epilogue(
                self.fw_metadata,
                fw_outs[num_forward_returns:num_forward],
                return_new_outs=False
            )
            return tuple(raw_returns)

    compiled_fw = CompiledForward()

    class CompiledBackward:
        def __init__(self):
            super().__init__()
            self.bw_gm = bw_module
            self.compiled_fw = compiled_fw

        def __call__(self, *args, **kwargs):
            return self.bw_gm(*args, **kwargs)
        
        def backward(self, *flat_args):
            num_mutated_inps = self.compiled_fw.fw_metadata.num_mutated_inputs
            num_intermediate_bases = self.compiled_fw.fw_metadata.num_intermediate_bases
            expected_grad_outs = (
                self.compiled_fw.fw_metadata.num_outputs + num_mutated_inps + num_intermediate_bases
            )

            assert len(flat_args) == expected_grad_outs
            out_info = self.compiled_fw.fw_metadata.output_info
            if (
                self.compiled_fw.fw_metadata.num_mutated_metadata_only_inputs > 0
                or self.compiled_fw.fw_metadata.num_outputs_aliased > 0
            ):
                inp_tangents, out_tangents, intermediate_base_tangents = (
                    flat_args[0:num_mutated_inps],
                    flat_args[num_mutated_inps:num_mutated_inps + self.compiled_fw.fw_metadata.num_outputs],
                    flat_args[num_mutated_inps + self.compiled_fw.fw_metadata.num_outputs:],
                )
                # input_info contains info on *every* input,
                # But in the backward(), we are only given grad outputs for every mutated input.
                # We then need to filter out the grad outputs that correspond to metadata-only mutations.
                mutated_inp_indices = self.compiled_fw.fw_metadata.mutated_inp_indices
                input_info = self.compiled_fw.fw_metadata.input_info
                assert len(inp_tangents) == len(mutated_inp_indices)
                inp_tangents_filtered = [
                    x
                    for x, info_idx in zip(inp_tangents, mutated_inp_indices)
                    if input_info[info_idx].mutates_data
                ]
                # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
                out_tangents_filtered = [
                    x
                    for x, info in zip(out_tangents, out_info)
                    if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]
                    and issubclass(info.raw_type, torch.Tensor)
                ]
                # intermediate bases always require gradients, and always participate in the backward graph.
                flat_bw_args = itertools.chain(inp_tangents_filtered, out_tangents_filtered, intermediate_base_tangents)
            else:
                # filter out non-tensor grad_outputs (aka due to ints being returned as outputs in the forward)
                num_mutated_inps = self.compiled_fw.fw_metadata.num_mutated_inputs
                mutated_inp_args = flat_args[:num_mutated_inps] if num_mutated_inps > 0 else []
                user_tangents = flat_args[num_mutated_inps:]
                assert len(user_tangents) == len(out_info)
                filtered_user_tangents = [x for x, info in zip(user_tangents, out_info) if issubclass(info.raw_type, torch.Tensor)]
                flat_bw_args = tuple(mutated_inp_args) + tuple(filtered_user_tangents)

            contiguous_args = [
                t.contiguous() if torch.is_tensor(t) else t for t in flat_bw_args
            ]

            rng_args = []
            if self.compiled_fw.fw_metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

            all_args = [
                *self.compiled_fw.symints,
                *self.compiled_fw.saved_tensors,
                *contiguous_args,
                *rng_args
            ]
            del contiguous_args

            def call_compiled_backward():
                out = call_func_with_args(
                    self.bw_gm.forward,
                    all_args,
                    steal_args=True,
                    disable_amp=torch._C._is_any_autocast_enabled(),
                )

                out = functionalized_rng_runtime_epilogue(self.compiled_fw.fw_metadata, out)
                return tuple(out)

            out = call_compiled_backward()
            return out

    compiled_bw = CompiledBackward()

    return compiled_fw, compiled_bw


class CompileInterpreter(torch.fx.Interpreter):

    def __init__(self, module, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.compiled = []

    def run(self, *args, initial_env=None):
        self.compiled.clear()
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):
        assert 'submod' in target

        submod = self.fetch_attr(target)
        compiled_fw, compiled_bw = compile_submod(submod, args, kwargs)
        self.compiled.append([compiled_fw, compiled_bw])

        return super().call_module(target, args, kwargs)

def split(
    mod: torch.nn.Module,
    multi_use_param_spec: Optional[MultiUseParamSpec] = None,
    tracer=None,
    output_loss_value_spec=None,
    deep_copy_module=False,
    split_policy: Optional[Callable[[torch.fx.GraphModule], torch.fx.GraphModule]] = None,
    return_to_0: bool = True,
    **kwargs,
):
    # TODO: abstract partitioning policy
    old_pipeline_tracer = get_pipeline_tracer()
    tracer = tracer or torch.fx.Tracer()
    set_pipeline_tracer(tracer)
    try:
        # TODO: tracing policy
        if deep_copy_module:
            # because further pipe building activities can modify mod
            mod = copy.deepcopy(mod)
        graph_torch_forward = tracer.trace(mod, **kwargs)
        gm_torch_forward = torch.fx.GraphModule(mod, graph_torch_forward)
    finally:
        set_pipeline_tracer(old_pipeline_tracer)

    if split_policy is not None:
        gm_torch_forward = split_policy(gm_torch_forward)

    gm_split = _from_traced(mod, gm_torch_forward, multi_use_param_spec, output_loss_value_spec,
                            return_to_0)

    return gm_split


def compile_splited(gm_split: fx.GraphModule, *args):

    # run through gm_split and compile each stage
    executor = CompileInterpreter(gm_split)
    executor.run(*args)

    for i, (compiled_fw, _) in enumerate(executor.compiled):
        delattr(gm_split, f"submod_{i}")
        setattr(gm_split, f"submod_{i}", compiled_fw)

    for i, (_, compiled_bw) in enumerate(reversed(executor.compiled)):
        setattr(gm_split, f"submod_bw_{i}", compiled_bw)
    return gm_split


class DetachExecutor(torch.fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """

    def __init__(self, module, garbage_collect_values=True):
        garbage_collect_values = False
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}

    def run(self, *args, initial_env=None):
        self.value_remap = {}
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):

        def detach_tensors(a):
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a

        if BackStage.name in target:
            assert len(args) == 0
            kwargs = dict(kwargs)  # tree node fetch kwargs as immutable_dict
            kwargs["input_values"] = [self.value_remap.get(v, v) for v in kwargs["input_values"]]
        else:
            args = torch.fx.node.map_aggregate(args, detach_tensors)
            kwargs = torch.fx.node.map_aggregate(kwargs, detach_tensors)

        return super().call_module(target, args, kwargs)
