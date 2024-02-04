import imp
import logging
import operator
from enum import Enum
from typing import (Callable, Dict, List, Sequence, Tuple, Any, Union)
from abc import ABC, abstractmethod
from typing import (Callable, Dict, List, Tuple)
from collections import defaultdict

import torch
import torch.fx as fx
from torch.fx._symbolic_trace import _Patcher
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode

from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.utils import rgetattr, rsetattr
from easydist.torch.experimental.pp.ed_split_module import ed_split_module
# ================================= section start ========================================
# The idea of functions in this section are borrowed from
# https://github.com/pytorch/PiPPy/blob/e9e2d5f0164a2e5d952a1424a3926da543365801/pippy/IR.py#L346
__tracer_global = None


def get_tracer_global():
    global __tracer_global
    return __tracer_global


def set_tracer_global(tracer):
    global __tracer_global
    __tracer_global = tracer


@fx.has_side_effect
def fw_bw_split_point():
    tracer_current = get_tracer_global()
    if tracer_current is not None and hasattr(tracer_current, "graph"):
        tracer_current.graph.call_function(fw_bw_split_point, (), {})


@fx.has_side_effect
def step_split_point():
    tracer_current = get_tracer_global()
    if tracer_current is not None and hasattr(tracer_current, "graph"):
        tracer_current.graph.call_function(step_split_point, (), {})

_before_split: Dict[type, Callable[[Any], Tuple[torch.Tensor]]] = {}
_after_split: Dict[type, Callable[[Tuple[torch.Tensor]], Any]] = {}

def before_split_register(*classes):
    def _register(func: Callable):
        for cls in classes:
            assert cls not in _before_split, f"split function for {cls} already registered"
            _before_split[cls] = func
        return func
    return _register

def after_split_register(*classes):
    def _register(func: Callable):
        for cls in classes:
            assert cls not in _after_split, f"split function for {cls} already registered"
            _after_split[cls] = func
        return func
    return _register

def get_registered_by_mro(registered, obj: Any) -> type:
    for cls in obj.__class__.mro():
        if cls in registered:
            return registered[cls]
    raise RuntimeError(f"no registered split function for {obj.__class__}")

# ================================= section end ========================================


def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )

from easydist.torch.experimental.pp.split_op import fw_bw_split_func, before_bw_split_func, fw_bw_param_type, fw_bw_ret_type

# ================================= section start ========================================
# The idea of functions in this section are borrowed from
# https://github.com/pytorch/PiPPy/blob/e9e2d5f0164a2e5d952a1424a3926da543365801/pippy/IR.py#L1206
class PipeSplitWrapper(torch.nn.Module):

    def __init__(self, mod: torch.nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, *args, **kwargs):
        ret = self.mod(*args, **kwargs)
        ctx = {}
        tensor_tuple: Tuple[torch.Tensor] = get_registered_by_mro(_before_split,ret)(ctx, ret)
        tensor_tuple: Tuple[torch.Tensor] = fw_bw_split_func(tensor_tuple)
        ret = get_registered_by_mro(_after_split,ret)(ctx, tensor_tuple)
        return ret

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if hasattr(self.mod, name):
            return getattr(self.mod, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def annotate_split_points(mod: torch.nn.Module, spec: set[str]):
    # TODO: make this implementation out-of-place?
    for qualname in iter(spec):
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
        wrapped_mod = PipeSplitWrapper(mod_to_wrap)
        setattr(predecessor_module, atoms[-1], wrapped_mod)


def split_into_equal_size(nstages: int = 1, ) -> Callable[[torch.nn.Module], torch.fx.GraphModule]:

    def _split_into_nstages_equal_size(mod: torch.nn.Module) -> torch.fx.GraphModule:
        tracer = torch.fx.Tracer()
        g = tracer.trace(mod)
        gm = torch.fx.GraphModule(mod, g)
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
        return gm

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
            ret = fw_bw_split_func(*ret)
            return ret[0] if len(ret) == 1 else ret

        return wrapped_func

    def gen_module_wrapper(target_module):
        return PipeSplitWrapper(target_module)

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


class SplitPatcher(_Patcher):

    def __init__(self, mod: torch.nn.Module, opt: torch.optim.Optimizer):
        super().__init__()
        self.mod = mod
        self.opt = opt

    def __enter__(self):
        patcher = super().__enter__()

        if self.mod:
            mod_cls = type(self.mod)
            orig_forward = mod_cls.forward

            # TODO @botbw: try register_module_full_forward_pre_hook
            def forward_wrapper(mod, *args, **kwargs):
                ret = orig_forward(mod, *args, **kwargs)
                return ret

            patcher.patch_method(mod_cls, 'forward', forward_wrapper, deduplicate=False)

        if self.opt:
            opt_cls = type(self.opt)
            orig_step = opt_cls.step

            def step_wrapper(opt, *args, **kwargs):
                step_split_point()
                orig_step(opt, *args, **kwargs)

            patcher.patch_method(opt_cls, 'step', step_wrapper, deduplicate=False)

        orig_backward = torch.Tensor.backward

        # TODO @botbw: register_module_full_backward_pre_hook
        def backward_wrapper(tensor, *args, **kwargs):
            tensor = before_bw_split_func(tensor)
            orig_backward(tensor, *args, **kwargs)

        patcher.patch_method(torch.Tensor, 'backward', backward_wrapper, deduplicate=False)

        return patcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)


def split_by(mod: torch.nn.Module, traced: torch.fx.GraphModule, split_point: Callable):
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

    # Ask split_module to return mapping from new qualname to old qualname
    qualname_map: Dict[str, str] = {}
    # TODO: what does split do with module invocations? does it move the modules
    # into the submodules?
    split = ed_split_module(traced, mod, split_callback, qualname_map)

    # peephole to remove pipe_split
    for submodule in split.modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ("call_function", split_point):
                    try:
                        submodule.graph.erase_node(node) # nodes that can be removed directly (i.e. not connected in graph)
                    except Exception as e: # nodes which are in graph
                        assert len(node.args) == 1, "split_point should have only one argument (list) or None"
                        args_prev = node.args[0]
                        to_erase = []
                        for gi in node.users:
                            assert gi.op == "call_function" and gi.target == operator.getitem
                            gi_index = gi.args[1]
                            gi.replace_all_uses_with(args_prev[gi_index])
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


def _extract_step_subgraph_from_args(gm: torch.fx.GraphModule, inputs_spec: List[str]):
    new_graph = fx.Graph()
    env = {}
    outputs = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if node.name in inputs_spec:  # only those in inputs
                env[node.name] = new_graph.placeholder(node.name)
        elif node.op == 'call_function':
            if node.target == operator.getitem:
                continue  # getitem is handled in foreach operators
            elif '_foreach_' == node.name[:9]:  # handle foreach operators
                list_args_mask = []
                args = []
                for arg in node.args:
                    # foreach operators in torch/_C/_VariableFunctions.pyi.in
                    if isinstance(arg, (list, tuple)):  # list of Tensors
                        args.append([env[x.name] for x in arg if x.name in env])
                        list_args_mask.append(tuple(x.name in env for x in arg))
                    elif isinstance(arg, torch.fx.Node):  # Tensors
                        args.append(env[arg.name])
                    else:  # Number or _complex
                        args.append(arg)
                kwargs = {}
                for kwarg_name, kwarg in node.kwargs.items():
                    if isinstance(kwarg, (list, tuple)):
                        kwargs[kwarg_name] = [env[x.name] for x in kwarg if x.name in env]
                        list_args_mask.append(tuple(x.name in env for x in kwarg))

                    elif isinstance(kwarg, torch.fx.Node):
                        kwargs[kwarg_name] = env[kwarg.name]
                    else:
                        kwargs[kwarg_name] = kwarg

                assert len(set(list_args_mask)
                           ) == 1, "input list of foreach operators should have the same mask"

                env[node.name] = new_graph.create_node(op='call_function',
                                                       name=node.name,
                                                       target=node.target,
                                                       args=tuple(args),
                                                       kwargs=kwargs)

                output_mask = list_args_mask[0]
                getitem_cnt = 0
                for getitem_user, kept in zip(node.users, output_mask):
                    assert getitem_user.op == 'call_function' and getitem_user.target == operator.getitem
                    if kept:
                        env[getitem_user.name] = new_graph.create_node(op='call_function',
                                                                       name=getitem_user.name,
                                                                       target=operator.getitem,
                                                                       args=(env[node.name],
                                                                             getitem_cnt))
                        getitem_cnt += 1
            else:
                args = []
                for arg in node.args:
                    if isinstance(arg, (list, tuple)):  # list of Tensors
                        args.append([env[x.name] for x in arg if x.name in env])
                    elif isinstance(arg, torch.fx.Node):  # Tensor
                        if arg.name in env:
                            args.append(env[arg.name])
                    else:  # Number or _complex
                        args.append(arg)
                if len(args) != len(node.args):
                    assert len(
                        args) == 0, "This node shall be completed removed since it has no args"
                    continue
                env[node.name] = new_graph.create_node(op='call_function',
                                                       name=node.name,
                                                       target=node.target,
                                                       args=tuple(args))
        elif node.op == 'output':
            for output in node.args[0]:
                if output.name in env:
                    outputs.append(env[output.name])
        else:
            raise RuntimeError(f"op {node.op} not supported")

    new_graph.output(outputs)

    new_graph.eliminate_dead_code()
    new_graph.lint()

    new_gm = fx.GraphModule(gm, new_graph)

    injected_states = {}
    to_pop = []
    for name, val in gm.injected_states.items():
        if name in inputs_spec:
            injected_states[name] = val
            to_pop.append(name)
    for name in to_pop:
        gm.injected_states.pop(name)

    new_gm.inputs_spec = inputs_spec
    new_gm.injected_states = injected_states  # TODO @botbw: move this to meta
    new_gm.outputs_spec = [node.name for node in outputs]

    return new_gm

class CompiledStageBase(ABC):
    @abstractmethod
    def forward(self, saved_for_bw, outputs, **kwargs):...

    @abstractmethod
    def backward(self, saved_for_bw, outputs, **kwargs):...

    @abstractmethod
    def step(self, saved_for_bw, outputs):...

    @abstractmethod
    def has_step(self):...


# TODO @botbw: simplify
def compile_stateful_stages(model, traced_gm, args_flatten, args_spec):
    global_phs_spec = [ph.name for ph in traced_gm.graph.nodes if ph.op == 'placeholder']
    global_outputs_spec = [
        node.name if node else None for node in list(traced_gm.graph.nodes)[-1].args[0]
    ]
    phs_spec_unflatten = pytree.tree_unflatten(global_phs_spec, args_spec)
    states_spec_flatten, _ = pytree.tree_flatten(
        phs_spec_unflatten[:3])  # flatten (params, buffers, named_states)
    inv_params = {arg: param_name for param_name, arg in phs_spec_unflatten[0].items()}
    node_metas = {node.name: node.meta for node in traced_gm.graph.nodes}
    
    splited_global, part_cnt = split_by(model, traced_gm, step_split_point)
    assert part_cnt <= 2, f"part_cnt should be 1 (fw_bw) or 2 (fw_bw + step), but found {part_cnt}"

    class CompiledStage(CompiledStageBase):

        def __init__(self, fw_gm, bw_gm, full_step_gm):
            self.fw_gm = fw_gm
            self.bw_gm = bw_gm

            self.fw_gm_injected_states = set(fw_gm.injected_states.keys())  # injected states

            self.fw_func_args = set(fw_gm.inputs_spec) - set(
                self.fw_gm_injected_states)  # args for self.forward

            self.fw_gm_args_saved_for_bw = set(fw_gm.inputs_spec) & set(
                bw_gm.inputs_spec)  # saved args for self.bw_gm
            self.fw_gm_outputs_saved_for_bw = set(fw_gm.outputs_spec) & set(
                bw_gm.inputs_spec)  # saved self.forward returns for self.bw_gm

            # TODO @botbw: better way of doing this
            self.fw_func_returns = set({
                output
                for output, users in self.fw_gm.call_module_users.items()
                if not (len(users) == 1 and next(iter(users)) == bw_gm.name)
            })  # not only used by bw

            self.bw_func_args = set(bw_gm.inputs_spec) - set(self.fw_gm_args_saved_for_bw) - set(
                self.fw_gm_outputs_saved_for_bw)  # args for self.backward

            self.global_outputs_spec = global_outputs_spec

            self.outputs_to_torch_name = inv_params

            if full_step_gm is not None:  # TODO @botbw: simplify this
                params = list(self.fw_gm_injected_states & set(full_step_gm.inputs_spec))
                param_names = set(inv_params[name] for name in params)
                grad_inputs = list(set(bw_gm.outputs_spec) & set(full_step_gm.inputs_spec))
                input_optim_states, _ = pytree.tree_flatten([
                    phs_spec_unflatten[2][param_name] for param_name in param_names
                    if param_name in phs_spec_unflatten[2]
                ])
                self.step_gm_args = params + grad_inputs + input_optim_states
                self.step_gm = _extract_step_subgraph_from_args(full_step_gm, self.step_gm_args)
                self.step_outputs_spec_to_states_spec = {
                    output: state
                    for output, state in zip(global_outputs_spec, states_spec_flatten)
                }
                for updated_state, ori_state in self.step_outputs_spec_to_states_spec.items():
                    if ori_state in self.outputs_to_torch_name:
                        self.outputs_to_torch_name[updated_state] = self.outputs_to_torch_name[ori_state]

        def forward(self, saved_for_bw=None, outputs=None, **kwargs):
            '''
            inputs:
                kwargs: activations from previous stages
                
                kwargs4gm: arguments for self.fw_gm
                    a. prepare arguments for self.fw_gm which is neighter
                        1. activations from previous stages (i.e. kwargs)
                        2. params and buffers of this stage (i.e. self.fw_gm.injected_states)
                    
                    b. save arguments for self.bw_gm and output
                        1. save kwargs4gm for backward if it appears in self.bw_gm.inputs_spec (activations, params for example)
                        2. save kwargs4gm for output if it appears in global_outputs_spec (buffers for example)
                
            outputs:
                output_from_gm: outputs of self.fw_gm
                    a. filter outputs that are required in following stages TODO @botbw
                    b. save arguments for self.bw_gm and output
                        1. save for backward if it appears in self.bw_gm.inputs_spec (intermediate values for example)
                        2. save for output if it appears in global_outputs_spec (original model outputs for example)
            '''
            if saved_for_bw is None or outputs is None:
                if not hasattr(self, 'saved_for_bw'):
                    self.saved_for_bw = {}
                if not hasattr(self, 'outputs'):
                    self.outputs = {}
                saved_for_bw = self.saved_for_bw
                outputs = self.outputs

            assert set(kwargs.keys(
            )) == self.fw_func_args, f"known kwargs {kwargs}, {self.fw_func_args} are required"
            kwargs4gm = {}
            for arg_name in self.fw_gm.inputs_spec:
                if arg_name in kwargs:
                    kwargs4gm[arg_name] = kwargs[arg_name]
                else:
                    kwargs4gm[arg_name] = self.fw_gm.injected_states[arg_name]
                    if arg_name in global_outputs_spec:  # params need to be returned directly if there is no opt_gm TODO botbw: seperate params and buffers?
                        outputs[arg_name] = kwargs4gm[arg_name]

                if arg_name in self.fw_gm_args_saved_for_bw:
                    saved_for_bw[arg_name] = kwargs4gm[arg_name]

            output_from_gm = self.fw_gm(**kwargs4gm)

            ret = {}
            for output_name, output in zip(self.fw_gm.outputs_spec, output_from_gm):
                if output_name in self.fw_func_returns:
                    ret[output_name] = output

                if output_name in self.fw_gm_outputs_saved_for_bw:
                    saved_for_bw[output_name] = output

                if output_name in global_outputs_spec:
                    outputs[output_name] = output

            return ret

        def backward(self, saved_for_bw=None, outputs=None, **kwargs):
            if saved_for_bw is None or outputs is None:
                saved_for_bw = self.saved_for_bw
                outputs = self.outputs

            assert set(kwargs.keys()) == self.bw_func_args, "backward args should be saved for fw"
            kwargs4gm = {}
            for arg_name in self.bw_gm.inputs_spec:
                if arg_name in kwargs:
                    kwargs4gm[arg_name] = kwargs[arg_name]
                else:
                    kwargs4gm[arg_name] = saved_for_bw[arg_name]
                    saved_for_bw.pop(arg_name)

                # if arg_name in global_outputs_spec:
                #     outputs[arg_name] = kwargs4gm[arg_name]

            assert len(saved_for_bw) == 0, "all backward args should be used"
            output_from_gm = self.bw_gm(**kwargs4gm)

            ret = {}
            for output_name, output in zip(self.bw_gm.outputs_spec, output_from_gm):
                if output_name in global_outputs_spec:
                    outputs[output_name] = output
                else:
                    assert output_name in self.bw_gm.call_module_users, "for bw_gm, output should be neither output or returned"
                    ret[output_name] = output
            return ret

        def step(self, saved_for_bw=None, outputs=None):
            if not (hasattr(self, 'step_gm_args') and hasattr(self, 'step_gm')):
                raise NotImplementedError("This compiled stage doesn't contain step_gm")
            if saved_for_bw is None or outputs is None:
                saved_for_bw = self.saved_for_bw
                outputs = self.outputs
            kwargs = {}
            for arg_name in self.step_gm_args:
                if arg_name in self.step_gm.injected_states:
                    kwargs[arg_name] = self.step_gm.injected_states[arg_name]
                elif arg_name in self.fw_gm.inputs_spec:
                    kwargs[arg_name] = self.fw_gm.injected_states[arg_name]
                elif arg_name in saved_for_bw:
                    kwargs[arg_name] = saved_for_bw[arg_name]
                elif arg_name in outputs:
                    kwargs[arg_name] = outputs[arg_name]
                else:
                    raise RuntimeError(f"arg {arg_name} not found")
            rets = self.step_gm(**kwargs)

            for output, ret in zip(self.step_gm.outputs_spec, rets):
                if output in global_outputs_spec:
                    outputs[output] = ret
                if output in self.step_outputs_spec_to_states_spec:
                    state_name = self.step_outputs_spec_to_states_spec[output]
                    self.fw_gm.injected_states[state_name] = ret

            return None  # step should always return None

        def has_step(self):
            return hasattr(self, 'step_gm_args') and hasattr(self, 'step_gm')

    name2state = {name: state for name, state in zip(states_spec_flatten, args_flatten)}

    def gen_stateful_submod(node, submod, is_bw=False):
        # process input
        inputs_spec = []
        inputs_users = []
        injected_states = {}
        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            if (not is_bw) and arg.name in states_spec_flatten:  # inject states to the first submod 
                # assert arg.name in name2state, f"{arg.name} not found in name2state or has been injected"
                if arg.name in name2state: # TODO @botbw: seperate params, buffers, and named_states in order to know whether there is params sharing
                    injected_states[arg.name] = name2state.pop(arg.name)

        # process output
        outputs_spec = []
        call_module_users = defaultdict(set)
        getitem_users = [
            user.op == 'call_function' and user.target == operator.getitem for user in node.users
        ]
        if any(getitem_users):  # output is tuple
            assert all(getitem_users), "determined by ed_split_module"
            for output in node.users:
                outputs_spec.append(output.name)
                for user in output.users:
                    if user.op == 'call_module':
                        call_module_users[output.name].add(user.name)
        else:  # output is value
            assert not any(getitem_users)
            outputs_spec.append(node.name)
            for user in node.users:
                if user.op == 'call_module':
                    call_module_users[node.name].add(user.name)

        # TODO @botbw: move this to meta
        assert not (hasattr(submod, 'inputs_spec') or hasattr(submod, 'injected_states')
                    or hasattr(submod, 'outputs_spec') or hasattr(submod, 'call_module_users')
                    or hasattr(submod, 'name'))
        submod.inputs_spec = inputs_spec
        submod.injected_states = injected_states
        submod.outputs_spec = outputs_spec
        submod.call_module_users = call_module_users
        submod.name = node.target

        return submod

    current_stateful_fw_bw = None
    compiled_stages = []
    submod_cnt = 0
    for node in splited_global.graph.nodes:
        if node.op == 'call_module':
            submod_cnt += 1
            submod_global_name = node.target
            submod_global = getattr(splited_global, submod_global_name)
            if submod_cnt % 2 :  # fw bw gm
                splited_fw_bw_gm, part_cnt = split_by(submod_global, submod_global, torch.ops.easydist.fw_bw_split.default)

                stateful_fw_bw = []
                for n in splited_fw_bw_gm.graph.nodes:
                    if n.op == 'call_module':  # extrace submods
                        assert 'submod_' == n.target[:7] and len(
                            n.kwargs) == 0, "splited_model should have no kwargs"
                        idx = int(n.target[7:])
                        submod = getattr(splited_fw_bw_gm, n.target)
                        stateful_fw_bw.append(gen_stateful_submod(n, submod, is_bw = idx >= part_cnt // 2))

                assert len(stateful_fw_bw) % 2 == 0, "each fw_gm should have a corresponding bw_gm"
                num_stage = len(stateful_fw_bw) // 2

                assert current_stateful_fw_bw is None, "There should be no consecutive compiled_fw_bw"
                current_stateful_fw_bw = stateful_fw_bw
            else:  # optimizer gm
                step_gm_global = gen_stateful_submod(node, submod_global)

                assert current_stateful_fw_bw is not None, "There should be a stateful_bw_fw before optimizer step"
                num_stage = len(current_stateful_fw_bw) // 2
                for fw_gm, bw_gm in zip(current_stateful_fw_bw[:num_stage],
                                        reversed(current_stateful_fw_bw[num_stage:])):
                    compiled_stage = CompiledStage(fw_gm, bw_gm, step_gm_global)
                    compiled_stages.append(compiled_stage)

                assert len(
                    step_gm_global.injected_states
                ) == 0, "All states of step_gm_global should have been injected to step_gm"
                current_stateful_fw_bw = None

    assert len(name2state) == 0, "All states should have been injected"

    if current_stateful_fw_bw is not None:  # forward and backward followed by no step
        num_stage = len(current_stateful_fw_bw) // 2
        for fw_gm, bw_gm in zip(current_stateful_fw_bw[:num_stage],
                                reversed(current_stateful_fw_bw[num_stage:])):
            compiled_stage = CompiledStage(fw_gm, bw_gm, None)
            compiled_stages.append(compiled_stage)
        current_stateful_fw_bw = None

    g = fx.Graph()
    env = {}
    submod_idx = 0
    node_to_stage = {}

    for node in splited_fw_bw_gm.graph.nodes:
        if node.op == 'placeholder':
            if node.name not in states_spec_flatten and len(node.users) > 0:
                env[node.name] = g.placeholder(node.name)
        elif node.op == 'call_module':
            if submod_idx < num_stage:  # forward
                stage_idx = submod_idx
                stage = compiled_stages[submod_idx]
                node_name = f'stage_{submod_idx}_fw'
                out_maybe_tuple = g.create_node(
                    name=node_name,
                    op='call_function',
                    target=stage.forward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.fw_func_args})
                node_to_stage[node_name] = stage_idx
                for output in stage.fw_func_returns:
                    env[output] = g.create_node(
                        name=output,
                        op='call_function',
                        target=operator.getitem,
                        args=(out_maybe_tuple, output)
                    )
                    node_to_stage[output] = stage_idx
            else:  # backward
                stage_idx = 2 * num_stage - submod_idx - 1
                stage = compiled_stages[stage_idx]
                node_name = f'stage_{stage_idx}_bw'
                out_maybe_tuple = g.create_node(
                    name=node_name,
                    op='call_function',
                    target=stage.backward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.bw_func_args})
                node_to_stage[node_name] = stage_idx
                for output in stage.bw_gm.outputs_spec:
                    if not output in global_outputs_spec:
                        env[output] = g.create_node(
                            name=output,
                            op='call_function',
                            target=operator.getitem, 
                            args=(out_maybe_tuple, output))
                        node_to_stage[output] = stage_idx
                if hasattr(stage, 'step_gm'):
                    g.create_node(
                        name=f'stage_{stage_idx}_step',
                        op='call_function',
                        target=stage.step
                        )
                    node_to_stage[f'stage_{stage_idx}_step'] = stage_idx
            submod_idx += 1

    def eliminate_dead_node():
        raise RuntimeError("This method should be called since the graph doesn't have output node")
    setattr(g, 'eliminate_dead_node', eliminate_dead_node)
    gm = fx.GraphModule({}, g)

    save_graphviz_dot(gm, 'gm')
    out2idx = {name: i for i, name in enumerate(global_outputs_spec)}
    return global_phs_spec, out2idx, compiled_stages, gm, node_metas, node_to_stage

@before_split_register(torch.Tensor)
def tensor_before_split(ctx: dict, input: torch.Tensor) -> Tuple[torch.Tensor]:
    return tuple([input])

@after_split_register(torch.Tensor)
def tensor_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> torch.Tensor:
    return output[0]

@before_split_register(list)
def list_before_split(ctx: dict, input: List[Union[torch.Tensor, Any]]) -> Tuple[torch.Tensor]:
    ctx['is_tensor'] = []
    ctx['non_tensor_vals'] = []
    tup = []
    for x in input:
        ctx['is_tensor'].append(isinstance(x, torch.Tensor))
        if ctx['is_tensor'][-1]:
            tup.append(x)
        else:
            ctx['non_tensor_vals'].append(x)

    return tuple(tup)

@after_split_register(list)
def list_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> List[Union[torch.Tensor, Any]]:
    ret = []
    output = list(output)
    for is_tensor in ctx['is_tensor']:
        if is_tensor:
            ret.append(output.pop(0))
        else:
            ret.append(ctx['non_tensor_vals'].pop(0))
    return ret

@before_split_register(tuple)
def tuple_before_split(ctx: dict, input: Tuple[Union[torch.Tensor, Any]]) -> Tuple[torch.Tensor]:
    return list_before_split(ctx, list(input))

@after_split_register(tuple)
def tuple_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> Tuple[Union[torch.Tensor, Any]]:
    return tuple(list_after_split(ctx, output))

# TODO @botbw: how to deal with dict?
