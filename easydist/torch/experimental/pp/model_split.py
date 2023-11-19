'''
Adapted from https://github.com/pytorch/PiPPy/blob/83a2308f4a53ae36eba2f0c1b2b262d5d697d37b/pippy/IR.py#L280
'''
import copy
import logging
import operator
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
# from easydist.torch.experimental.pp.my_proxy_tensor import my_make_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module
from torch.nn.utils import stateless

from easydist.torch.compiler import preprocess_traced_graph
from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from easydist.torch.experimental.pp import (get_pipeline_tracer, set_pipeline_tracer)
from easydist.torch.experimental.pp.backward import (BackStage, _insert_stage_symbolic_backward,
                                                     stage_backward)
from easydist.torch.experimental.pp.loss_wrapper import TrivialLossWrapper
from easydist.torch.init_helper import SetParaInitHelper
from easydist.torch.utils import _enable_compile, _rematerialize_optimizer
from easydist.utils import rgetattr, rsetattr


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


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

class EDCompiledModule(torch.nn.Module):
    stateless_func: Callable
    def __init__(self, stateless_func: Callable, params=None, buffers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stateless_func = stateless_func
        self.name_mapping = {}
        if params is not None:
            self.update_params(params)
        if buffers is not None:
            self.update_buffers(buffers)

    def update_params(self, params):
        self.ed_params = params
        for pname, param in params.items():
            mapped_name = pname.replace('.', '_')
            self.name_mapping[mapped_name] = pname
            self.register_parameter(mapped_name, param)

    def update_buffers(self, buffers):
        self.ed_buffers = buffers
        for bname, buffer in buffers.items():
            mapped_name = bname.replace('.', '_')
            self.name_mapping[mapped_name] = bname
            self.register_buffer(mapped_name, buffer)

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        iter_unmapped = super().named_parameters(prefix, recurse, remove_duplicate)
        return map(lambda pair: (self.name_mapping[pair[0]], pair[1]), iter_unmapped)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        iter_unmapped = super().named_buffers(prefix, recurse)
        return map(lambda pair: (self.name_mapping[pair[0]], pair[1]), iter_unmapped)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        raise NotImplementedError("state_dict not implemented")

class EDCompiledBackward:
    def __init__(self, forward_mod, stateless_backward):
        self.forward_mod = forward_mod
        self.stateless_backward = stateless_backward

    def __call__(
        self,
        stage_output,
        output_grads,
        input_values,
        outputs_with_grads_idxs: List[int],
        stage_info: str,
    ):
        params = dict(self.forward_mod.named_parameters())
        grad_inputs, barrier_token, grads, params = self.stateless_backward(params, stage_output, output_grads, input_values, outputs_with_grads_idxs, stage_info)
        
        for pname in params:
            params[pname].grad = grads[pname]

        self.forward_mod.update_params(params)

        return grad_inputs, barrier_token



def trace_backward(tracing_mode, forward_mod, backward_mod, args, kwargs):
    executor_ret = None

    def stateless_backward(params, stage_output, output_grads, input_values, outputs_with_grads_idxs, stage_info):
        nonlocal executor_ret
        grad_inputs, barrier_token = backward_mod(stage_output, output_grads, input_values, outputs_with_grads_idxs, stage_info)
        executor_ret = (grad_inputs, barrier_token)
        grads = {k: v.grad for k, v in params.items()}
        return grad_inputs, barrier_token, grads, params

    params = dict(forward_mod.named_parameters())
    with _enable_compile():
        gm = make_fx(stateless_backward,
                        tracing_mode=tracing_mode,
                        decomposition_table=EASYDIST_DECOMP_TABLE,
                        _allow_non_fake_inputs=False)(params, *args, **kwargs)

    gm.graph.eliminate_dead_code()
    gm = preprocess_traced_graph(gm)
    gm.recompile()

    return executor_ret, EDCompiledBackward(forward_mod, gm.forward)


class EDCompiledForward(EDCompiledModule):

    def __init__(self, stateless_forward, params, buffers) -> None:
        super().__init__(stateless_forward, params, buffers)

    def forward(self, *args, **kwargs):
        params = self.ed_params
        buffers = self.ed_buffers

        output, params, buffers, grads = self.stateless_func(
            params, buffers, args, kwargs)

        for pname in params:
            params[pname].grad = grads[pname]

        self.update_params(params)
        self.update_buffers(buffers)

        return output

def trace_forward(tracing_mode, module, args, kwargs):
    executor_ret = None

    def stateless_forward(params, buffers, args, kwargs):
        nonlocal executor_ret
        with stateless._reparametrize_module(
                cast(torch.nn.Module, module), {
                    **params,
                    **buffers
                }, tie_weights=True) if module else nullcontext():
            executor_ret = output = module(*args, **kwargs)

        params = dict(module.named_parameters())
        grads = {k: v.grad for k, v in params.items()}
        buffers = dict(module.named_buffers())

        return output, params, buffers, grads

    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    with _enable_compile():
        gm = make_fx(stateless_forward,
                               tracing_mode=tracing_mode,
                               decomposition_table=EASYDIST_DECOMP_TABLE,
                               _allow_non_fake_inputs=False)(params, buffers, args, kwargs)

    gm.graph.eliminate_dead_code()
    gm = preprocess_traced_graph(gm)
    gm.recompile()

    return executor_ret, EDCompiledForward(gm.forward, params, buffers)


def _from_tracing(
    mod: torch.nn.Module,
    *args,
    multi_use_param_spec: Optional[MultiUseParamSpec] = None,
    tracer=None,
    output_loss_value_spec=None,
    deep_copy_module=False,
    split_policy: Optional[Callable[[torch.fx.GraphModule], torch.fx.GraphModule]] = None,
    return_to_0: bool = True,
    **kwargs,
):
    # TODO: abstract partitioning policy
    old__pipeline_tracer = get_pipeline_tracer()
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
        set_pipeline_tracer(old__pipeline_tracer)

    if split_policy is not None:
        gm_torch_forward = split_policy(gm_torch_forward)

    gm_split = _from_traced(mod, gm_torch_forward, multi_use_param_spec, output_loss_value_spec,
                            return_to_0)

    executor = CompileInterpreter(gm_split)
    executor.run(*args)

    for name, (ret_val, compiled_mod) in executor.compiled.items():
        delattr(gm_split, name)
        setattr(gm_split, name, compiled_mod)

    # TODO @botbw debug
    for name, param in gm_split.named_parameters():
        setattr(gm_split, name, None)

    return gm_split


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

    num_stages = _number_and_count_forward_stages(split)

    has_loss_and_backward = False
    generated_loss_spec = output_loss_value_spec

    if mod.training or output_loss_value_spec is not None:
        loss_node, output_node, generated_loss_spec = _find_loss_output(
            mod, split.graph, output_loss_value_spec)
        if loss_node is not None:
            _insert_stage_symbolic_backward(split.graph, loss_node, output_node, return_to_0,
                                            num_stages)
            split.recompile()
            has_loss_and_backward = True
            logging.info("Pipeline is in training mode, backward pass generated")
        else:
            logging.warning(
                "Did not find any loss value from model output, your pipeline will be in inference mode. "
                "If you want your pipeline to be in training mode, please specify a loss value via "
                "`output_loss_value_spec`.")
    else:
        logging.info("Pipeline is in evaluation mode, backward pass not generated")

    return split


def run_local_split_gm(split_gm, *args):
    executor = DetachExecutor(split_gm)
    return executor.run(*args)


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
            args = list(args)
            args[2] = [self.value_remap.get(v, v) for v in args[2]]
            kwargs = {}
        else:
            args = torch.fx.node.map_aggregate(args, detach_tensors)
            kwargs = torch.fx.node.map_aggregate(kwargs, detach_tensors)

        return super().call_module(target, args, kwargs)


class CompileInterpreter(torch.fx.Interpreter):

    def __init__(self, module):
        super().__init__(module, False)
        self.value_remap = {}
        self.compiled = {}
        self.stage_cnt = 0

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

        submod = self.fetch_attr(target)

        if isinstance(submod, BackStage):
            args = list(args)
            args[2] = [self.value_remap.get(v, v) for v in args[2]]
            kwargs = {}
            # TODO: @botbw
            forward_mod = self.compiled[f"submod_{target[-1]}"][1]
            ret = trace_backward("fake", forward_mod, submod, args, kwargs)

        else:
            args = torch.fx.node.map_aggregate(args, detach_tensors)
            kwargs = dict(torch.fx.node.map_aggregate(kwargs, detach_tensors))
            ret = trace_forward("fake", submod, args, kwargs)

        self.compiled[target] = ret

        return ret[0]
