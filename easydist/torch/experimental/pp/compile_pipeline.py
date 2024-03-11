from dataclasses import dataclass
import logging
import operator
import textwrap
from typing import (Callable, Dict, List, Optional, Set, Tuple, Any, Union)
from typing import (Callable, Dict, List, Tuple)
from collections import defaultdict
from enum import Enum

import torch
import torch.fx as fx
from torch.fx._symbolic_trace import _Patcher
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode

from easydist.torch.experimental.pp.utils import save_graphviz_dot
from easydist.utils import rgetattr, rsetattr
from easydist.torch.experimental.pp.ed_split_module import ed_split_module


class StateType(Enum):
    PARAMS = "params"
    BUFFERS = "buffers"
    OPTIMSTATES = "optimstates"


class SubmodType(Enum):
    FW = "fw"
    BW = "bw"
    STEP = "step"


@dataclass
class CompiledMeta:
    nstages: int

    # input meta
    params_names_unflatten: Dict[str, str]
    buffers_names_unflatten: Dict[str, str]
    optimstates_names_unflatten: Dict[str, Dict[str, str]]
    args_names_unflatten: Tuple[str, ...]
    kwargs_names_unflatten: Dict[str, str]

    # node name to torch name mapping
    inv_params: Dict[str, str]
    inv_buffers: Dict[str, str]
    inv_optimstates: Dict[str, str]
    inv_optimstates_type: Dict[str, str]

    # output meta
    output_params_names_unflatten: Dict[str, str]
    output_buffers_names_unflatten: Dict[str, str]
    output_optimstates_names_unflatten: Dict[str, Dict[str, str]]
    output_optimstates_names_flatten: List[str]
    nones_or_grads_names_unflatten: Dict[str, str]
    returns_names_unflatten: Union[Tuple[str, ...]]

    # output node to input node mapping
    output2input_params: Dict[str, str]
    output2input_buffers: Dict[str, str]
    output2input_optimstates: Dict[str, str]

    # node name to stage idx mapping
    node_to_stage_idx: Dict[str, int]


@dataclass
class EDGraphModule:
    gm: torch.fx.GraphModule
    submod_type: SubmodType
    inputs_spec: set[str]
    injected_states: Dict[StateType, Dict[str, torch.Tensor]]
    outputs_spec: List[str]
    call_module_users: Dict[str, Set[str]]
    name: str

    def __call__(self, *args, **kwargs):
        return self.gm(*args, **kwargs)


class CompiledStage:  # TODO @botbw: make this picklable

    def __init__(self, compiled_meta: CompiledMeta, fw_gm: EDGraphModule,
                 bw_gm: Optional[EDGraphModule], full_step_gm: Optional[EDGraphModule]):
        self.compiled_meta = compiled_meta
        self.fw_gm = fw_gm
        params_and_buffers = (set(fw_gm.injected_states[StateType.PARAMS].keys())
                              | set(fw_gm.injected_states[StateType.BUFFERS].keys()))
        self.fw_func_args = set(fw_gm.inputs_spec) - params_and_buffers  # args for self.forward
        self.fw_func_returns = set(  # TODO @botbw: better way of doing this
            output for output, users in self.fw_gm.call_module_users.items()
            if not (len(users) == 1 and (bw_gm is not None and next(iter(users)) == bw_gm.name))
        )  # not just used by bw, need to pass to next stage

        if bw_gm is not None:
            self.bw_gm = bw_gm
            args_activations = set(fw_gm.inputs_spec) & set(bw_gm.inputs_spec) - params_and_buffers
            outputs_activations = set(fw_gm.outputs_spec) & set(bw_gm.inputs_spec)
            self.activation_nodes = args_activations | outputs_activations
            self.bw_func_args = set(bw_gm.inputs_spec) - set(
                self.activation_nodes) - params_and_buffers  # args for self.backward

        if full_step_gm is not None:  # TODO @botbw: simplify this
            stage_params_inputs = set(self.fw_gm.injected_states[StateType.PARAMS]) & set(
                full_step_gm.inputs_spec)
            inv_stage_params_inputs = set(self.compiled_meta.inv_params[name]
                                          for name in stage_params_inputs)
            stage_grad_inputs = set({
                self.compiled_meta.nones_or_grads_names_unflatten[k]
                for k in inv_stage_params_inputs
            })
            stage_optimstates_inputs, _ = pytree.tree_flatten([
                self.compiled_meta.optimstates_names_unflatten[k] for k in inv_stage_params_inputs
                if k in self.compiled_meta.optimstates_names_unflatten
            ])
            stage_optimstates_inputs = set(stage_optimstates_inputs)
            self.step_gm_args = stage_params_inputs | stage_grad_inputs | stage_optimstates_inputs
            self.step_gm = _extract_step_subgraph_from_args(full_step_gm, self.step_gm_args)

    @torch.no_grad
    def forward(self, activations_chunk=None, outputs_chunk=None, **kwargs):
        assert set(kwargs.keys(
        )) == self.fw_func_args, f"known kwargs {kwargs}, {self.fw_func_args} are required"

        if activations_chunk is None or outputs_chunk is None:  # for local run
            if not hasattr(self, 'activations'):
                self.activations = {}
            if not hasattr(self, 'outputs'):
                self.outputs = {}
            activations_chunk = self.activations
            outputs_chunk = self.outputs

        kwargs4gm = {}
        for arg_name in self.fw_gm.inputs_spec:
            if arg_name in kwargs:
                kwargs4gm[arg_name] = kwargs[arg_name]
            elif arg_name in self.fw_gm.injected_states[StateType.PARAMS]:  # params
                kwargs4gm[arg_name] = self.fw_gm.injected_states[StateType.PARAMS][arg_name]
                if arg_name in self.compiled_meta.output2input_params:  # param in returns
                    outputs_chunk[arg_name] = kwargs4gm[arg_name]
            elif arg_name in self.fw_gm.injected_states[StateType.BUFFERS]:  # buffers
                kwargs4gm[arg_name] = self.fw_gm.injected_states[StateType.BUFFERS][arg_name]
                if arg_name in self.compiled_meta.output2input_buffers:  # buffer in returned value
                    outputs_chunk[arg_name] = kwargs4gm[arg_name]
            else:
                raise RuntimeError(f"arg {arg_name} not found")

            if self.has_bw() and arg_name in self.activation_nodes:
                activations_chunk[arg_name] = kwargs4gm[arg_name]

        output_from_gm = _to_tuple(self.fw_gm(**kwargs4gm))

        ret = {}
        assert len(output_from_gm) == len(
            self.fw_gm.outputs_spec
        ), "output_from_gm should have the same length as self.fw_gm.outputs_spec"
        for output_name, output in zip(self.fw_gm.outputs_spec, output_from_gm):
            if (output_name in self.fw_func_returns) \
                or (self.has_bw() and output_name in self.activation_nodes) \
                    or (output_name in self.compiled_meta.returns_names_unflatten) \
                        or (output_name in self.compiled_meta.output2input_buffers): # TODO @botbw: simplify this
                if output_name in self.fw_func_returns:  # output in next stages
                    ret[output_name] = output
                if self.has_bw() and output_name in self.activation_nodes:  # output in activations
                    activations_chunk[output_name] = output
                if output_name in self.compiled_meta.returns_names_unflatten:  # output in returns
                    outputs_chunk[output_name] = output
                if (output_name
                        in self.compiled_meta.output2input_buffers):  # output is updated buffer
                    outputs_chunk[output_name] = output
                    input_node_name = self.compiled_meta.output2input_buffers[output_name]
                    self.fw_gm.injected_states[StateType.BUFFERS][
                        input_node_name] = output  # update buffer in case it's not updated in place.
            else:
                raise RuntimeError(f"output {output_name} not sure where to go")

        return ret

    @torch.no_grad
    def backward(self, activations_chunk=None, outputs_chunk=None, **kwargs):
        if not self.has_bw():
            raise NotImplementedError("This compiled stage doesn't contain bw_gm")

        assert set(kwargs.keys()) == self.bw_func_args, "backward args should be saved for fw"

        if activations_chunk is None or outputs_chunk is None:  # for local run
            activations_chunk = self.activations
            outputs_chunk = self.outputs

        kwargs4gm = {}
        for arg_name in self.bw_gm.inputs_spec:
            if arg_name in kwargs:
                kwargs4gm[arg_name] = kwargs[arg_name]
            elif arg_name in activations_chunk:
                kwargs4gm[arg_name] = activations_chunk[arg_name]
                activations_chunk.pop(arg_name)
            elif arg_name in self.fw_gm.injected_states[StateType.PARAMS]:  # param
                kwargs4gm[arg_name] = self.fw_gm.injected_states[StateType.PARAMS][arg_name]
            elif arg_name in self.fw_gm.injected_states[StateType.BUFFERS]:  # buffer
                kwargs4gm[arg_name] = self.fw_gm.injected_states[StateType.BUFFERS][arg_name]
            else:
                raise RuntimeError(f"arg {arg_name} not found")

        assert len(activations_chunk) == 0, "all backward args should be used"
        output_from_gm = _to_tuple(self.bw_gm(**kwargs4gm))

        ret = {}
        assert len(output_from_gm) == len(
            self.bw_gm.outputs_spec
        ), "output_from_gm should have the same length as self.bw_gm.outputs_spec"

        for output_name, output in zip(self.bw_gm.outputs_spec, output_from_gm):
            if output_name in self.compiled_meta.nones_or_grads_names_unflatten.values():
                outputs_chunk[output_name] = output
            elif output_name in self.bw_gm.call_module_users:
                ret[output_name] = output
            else:
                raise RuntimeError(f"output {output_name} not sure where to go")
        return ret

    @torch.no_grad
    def step(self, outputs_batch=None):  # params_and_grads is the args
        if not self.has_step():
            raise NotImplementedError("This compiled stage doesn't contain step_gm")

        if outputs_batch is None:
            outputs_batch = self.outputs

        kwargs = {}
        for arg_name in self.step_gm_args:
            if arg_name in self.step_gm.injected_states[StateType.OPTIMSTATES]:  # optimstates
                kwargs[arg_name] = self.step_gm.injected_states[StateType.OPTIMSTATES][arg_name]
            elif arg_name in self.fw_gm.injected_states[StateType.PARAMS]:  # params
                kwargs[arg_name] = self.fw_gm.injected_states[StateType.PARAMS][arg_name]
            elif arg_name in self.compiled_meta.nones_or_grads_names_unflatten.values(
            ):  # grads already in outputs
                kwargs[arg_name] = outputs_batch[arg_name]
            else:
                raise RuntimeError(f"arg {arg_name} not sure where to go")

        rets = _to_tuple(self.step_gm(**kwargs))

        for output, ret in zip(self.step_gm.outputs_spec, rets):
            if output in self.compiled_meta.output2input_params:  # params
                outputs_batch[output] = ret
                input_node_name = self.compiled_meta.output2input_params[output]
                self.fw_gm.injected_states[StateType.PARAMS][
                    input_node_name] = ret  # update params in case it's not updated in place.
            elif output in self.compiled_meta.output_optimstates_names_flatten:  # optimstates
                outputs_batch[output] = ret
                input_node_name = self.compiled_meta.output2input_optimstates[output]
                self.step_gm.injected_states[StateType.OPTIMSTATES][
                    input_node_name] = ret  # update optimstates in case it's not updated in place.
            else:
                raise RuntimeError(f"output {output} not sure where to go")

        return None  # step should always return None

    def has_step(self):
        return hasattr(self, 'step_gm')

    def has_bw(self):
        return hasattr(self, 'bw_gm')

    def state_dict(self):
        state_dict = {}
        state_dict.update(self.named_parameters())
        state_dict.update(self.named_buffers())
        return state_dict

    def load_state_dict(self, state_dict):
        for torch_name, tensor in state_dict.items():
            if torch_name in self.compiled_meta.inv_params:
                self.fw_gm.injected_states[StateType.PARAMS][torch_name] = tensor
            elif torch_name in self.compiled_meta.inv_buffers:
                self.fw_gm.injected_states[StateType.BUFFERS][torch_name] = tensor
            else:
                raise RuntimeError(f"state_dict key {torch_name} not found")

    def load_optimizer_state_dict(self, state_dict):
        for torch_name, state in state_dict.items():
            if torch_name in self.compiled_meta.inv_optimstates:
                self.step_gm.injected_states[StateType.OPTIMSTATES][torch_name] = state
            else:
                raise RuntimeError(f"state_dict key {torch_name} not found")

    def named_parameters(self):
        return {
            self.compiled_meta.inv_params[name]: tensor
            for name, tensor in self.fw_gm.injected_states[StateType.PARAMS].items()
        }

    def named_buffers(self):
        return {
            self.compiled_meta.inv_buffers[name]: tensor
            for name, tensor in self.fw_gm.injected_states[StateType.BUFFERS].items()
        }

    def optimizer_state_dict(self):
        optim_state = defaultdict(dict)
        for name, tensor in self.step_gm.injected_states[StateType.OPTIMSTATES].items():
            optim_state[self.compiled_meta.inv_optimstates[name]][
                self.compiled_meta.inv_optimstates_type[name]] = tensor
        return dict(optim_state)


# ================================= section start ========================================
# The idea of functions in this section are borrowed from
# https://github.com/pytorch/PiPPy/blob/e9e2d5f0164a2e5d952a1424a3926da543365801/pippy/IR.py#L346
__tracer_global = None
__backward_flag = False


def get_backward_flag():
    global __backward_flag
    return __backward_flag


def set_backward_flag(flag):
    global __backward_flag
    __backward_flag = flag


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


from easydist.torch.experimental.pp.split_op import fw_bw_split_func, before_bw_split_func


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
        tensor_tuple: Tuple[torch.Tensor] = get_registered_by_mro(_before_split, ret)(ctx, ret)
        tensor_tuple: Tuple[torch.Tensor] = fw_bw_split_func(tensor_tuple)
        ret = get_registered_by_mro(_after_split, ret)(ctx, tensor_tuple)
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
        logging.debug(f"Total model size: {total_size}, " f"per stage size: {per_stage_size}")

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

        if (accumulate_size + new_size <=
                threshold):  # can accommodate this node in current bucket
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
            set_backward_flag(True)

        patcher.patch_method(torch.Tensor, 'backward', backward_wrapper, deduplicate=False)

        return patcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)


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

    # Ask split_module to return mapping from new qualname to old qualname
    qualname_map: Dict[str, str] = {}
    # TODO: what does split do with module invocations? does it move the modules
    # into the submodules?
    split = ed_split_module(traced, None, split_callback, qualname_map)

    # peephole to remove pipe_split
    for submodule in split.modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ("call_function", split_point):
                    try:
                        submodule.graph.erase_node(
                            node
                        )  # nodes that can be removed directly (i.e. not connected in graph)
                    except Exception as e:  # nodes which are in graph
                        assert len(
                            node.args
                        ) == 1, "split_point should have only one argument (list) or None"
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


def _extract_step_subgraph_from_args(ed_gm: EDGraphModule, inputs_spec: set[str]):
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
                    assert not any(
                        isinstance(arg, torch.fx.Node) for arg in
                        args), "This node shall be completed removed since it has no tensor args"
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

    new_graph.output(tuple(outputs))

    new_graph.eliminate_dead_code()
    new_graph.lint()

    new_gm = fx.GraphModule(gm, new_graph)

    injected_states = {
        StateType.OPTIMSTATES: {},
    }
    to_pop = []
    for name, val in ed_gm.injected_states[StateType.OPTIMSTATES].items():
        if name in inputs_spec:
            injected_states[StateType.OPTIMSTATES][name] = val
            to_pop.append(name)
    for name in to_pop:
        ed_gm.injected_states[StateType.OPTIMSTATES].pop(name)

    output_spec = [node.name for node in outputs]

    return EDGraphModule(new_gm, ed_gm.submod_type, inputs_spec, injected_states, output_spec,
                         ed_gm.call_module_users, ed_gm.name)


def func_inputs_to_graph_inputs_by_stages(compiled_meta: CompiledMeta,
                                          *args,
                                          move_to_device=False,
                                          **kwargs):
    nstages = compiled_meta.nstages
    args_names_unflatten = compiled_meta.args_names_unflatten
    kwargs_names_unflatten = compiled_meta.kwargs_names_unflatten
    name_to_stage_idx = compiled_meta.node_to_stage_idx

    stage_kwargs = [{} for _ in range(nstages)]
    assert len(args) == len(args_names_unflatten), "args should have the same length as args_names"
    for arg_name, arg_val in zip(args_names_unflatten, args):
        if not arg_name in name_to_stage_idx:
            continue
        stage_idx = name_to_stage_idx[arg_name]
        if move_to_device:
            arg_val = arg_val.to(f'cuda:{stage_idx}')  # TODO @botbw: improve this
        stage_kwargs[stage_idx][arg_name] = arg_val

    assert set(kwargs.keys()) == set(
        kwargs_names_unflatten.keys()), "kwargs should have the same keys as kwargs_names"
    for k, v in kwargs.items():
        if not k in name_to_stage_idx:
            continue
        stage_idx = name_to_stage_idx[k]
        stage_kwargs[stage_idx][kwargs_names_unflatten[k]] = v
    return stage_kwargs


def graph_outputs_to_func_outputs(compiled_meta: CompiledMeta, node_outputs: Dict[str,
                                                                                  torch.Tensor]):
    maybe_updated_params_names_unflatten = compiled_meta.output_params_names_unflatten
    maybe_updated_buffers_names_unflatten = compiled_meta.output_buffers_names_unflatten
    updated_optimstates_names_unflatten = compiled_meta.output_optimstates_names_unflatten
    nones_or_grads_names_unflatten = compiled_meta.nones_or_grads_names_unflatten
    returns_names_unflatten = compiled_meta.returns_names_unflatten

    ret = []
    ret.append({
        torch_name: node_outputs[node_name]
        for torch_name, node_name in maybe_updated_params_names_unflatten.items()
    })
    ret.append({
        torch_name: node_outputs[node_name]
        for torch_name, node_name in maybe_updated_buffers_names_unflatten.items()
    })
    optimstates = defaultdict(dict)
    for torch_name, state_dict in updated_optimstates_names_unflatten.items():
        for typ, ph_name in state_dict.items():
            optimstates[torch_name][typ] = node_outputs[ph_name]

    ret.append(dict(optimstates))
    ret.append({
        torch_name: node_outputs[node_name]
        for torch_name, node_name in nones_or_grads_names_unflatten.items()
    })
    ret.append(tuple(node_outputs[node_name] for node_name in returns_names_unflatten))
    return ret


def graph_outputs_to_func_outputs_non_strict(compiled_meta: CompiledMeta, outputs_dict):
    maybe_updated_params_names_unflatten = compiled_meta.output_params_names_unflatten
    maybe_updated_buffers_names_unflatten = compiled_meta.output_buffers_names_unflatten
    updated_optimstates_names_unflatten = compiled_meta.output_optimstates_names_unflatten
    nones_or_grads_names_unflatten = compiled_meta.nones_or_grads_names_unflatten
    returns_names_unflatten = compiled_meta.returns_names_unflatten

    ret = []
    ret.append({
        torch_name: outputs_dict[node_name]
        for torch_name, node_name in maybe_updated_params_names_unflatten.items()
        if node_name in outputs_dict
    })
    ret.append({
        torch_name: outputs_dict[node_name]
        for torch_name, node_name in maybe_updated_buffers_names_unflatten.items()
        if node_name in outputs_dict
    })
    optimstates = defaultdict(dict)
    for torch_name, state_dict in updated_optimstates_names_unflatten.items():
        for typ, ph_name in state_dict.items():
            if ph_name in outputs_dict:
                optimstates[torch_name][typ] = outputs_dict[ph_name]

    ret.append(dict(optimstates))
    ret.append({
        torch_name: outputs_dict[node_name]
        for torch_name, node_name in nones_or_grads_names_unflatten.items()
        if node_name in outputs_dict
    })
    ret.append(
        tuple(outputs_dict[node_name] if node_name in outputs_dict else None
              for node_name in returns_names_unflatten))
    return ret


# TODO @botbw: simplify
def compile_pipeline(traced_stateless_func: fx.GraphModule,
                     nstages: int,
                     stateless_func_args,
                     delayed_init=False,
                     strict=True):
    is_backward_called = get_backward_flag()

    phs_names_flatten = [
        ph.name for ph in traced_stateless_func.graph.nodes if ph.op == 'placeholder'
    ]
    phs_names_unflatten = pytree.tree_unflatten(phs_names_flatten, traced_stateless_func._in_spec)

    outputs_names_flatten = [
        node.name if node else None for node in list(traced_stateless_func.graph.nodes)[-1].args[0]
    ]
    outputs_names_unflatten = pytree.tree_unflatten(outputs_names_flatten,
                                                    traced_stateless_func._out_spec)

    # defined in stateless_func
    params, buffers, optimstates, args, kwargs = stateless_func_args

    if delayed_init:
        fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True, allow_non_fake_inputs=False)
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

        params, buffers, optimstates = pytree.tree_map(wrap_fake, [params, buffers, optimstates])

    params_names_unflatten, buffers_names_unflatten, optimstates_names_unflatten, args_names_unflatten, kwargs_names_unflatten = phs_names_unflatten
    output_params_names_unflatten, output_buffers_names_unflatten, output_optimstates_names_unflatten, nones_or_grads_names_unflatten, returns_names_unflatten = outputs_names_unflatten
    output_optimstates_names_flatten, _ = pytree.tree_flatten(output_optimstates_names_unflatten)
    returns_names_unflatten = _to_tuple(returns_names_unflatten)  # returns might be value or tuple

    inv_params = {arg: param_name for param_name, arg in params_names_unflatten.items()}
    inv_buffers = {arg: buffer_name for buffer_name, arg in buffers_names_unflatten.items()}
    inv_optimstates = {}
    inv_optimstates_type = {}
    for param_key, state in optimstates_names_unflatten.items():
        for typ, ph_name in state.items():
            inv_optimstates[ph_name] = param_key
            inv_optimstates_type[ph_name] = typ

    output2input_params = {
        output_name: input_name
        for output_name, input_name in zip(
            pytree.tree_flatten(output_params_names_unflatten)[0],
            pytree.tree_flatten(params_names_unflatten)[0])
    }

    output2input_buffers = {
        output_name: input_name
        for output_name, input_name in zip(
            pytree.tree_flatten(output_buffers_names_unflatten)[0],
            pytree.tree_flatten(buffers_names_unflatten)[0])
    }

    output2input_optimstates = {
        output_name: input_name
        for output_name, input_name in zip(output_optimstates_names_flatten,
                                           pytree.tree_flatten(optimstates_names_unflatten)[0])
    }

    splited_global, part_cnt = split_by(traced_stateless_func, step_split_point)
    assert part_cnt <= 2, f"part_cnt should be 1 (fw or fw_bw) or 2 (fw_bw + step), but found {part_cnt}"

    states_used_by = defaultdict(list)

    def extract_output(node):
        # process output
        outputs_spec = []
        call_module_users = defaultdict(set)
        getitem_users = [
            user.op == 'call_function' and user.target == operator.getitem for user in node.users
        ]
        if any(getitem_users):  # output is tuple
            assert all(getitem_users), "determined by ed_split_module"
            for gi in node.users:
                outputs_spec.append(gi.name)
                for gi_user in gi.users:
                    if gi_user.op == 'call_module':
                        call_module_users[gi.name].add(gi_user.name)
        else:  # output is value
            assert not any(getitem_users)
            outputs_spec.append(node.name)
            for user in node.users:
                if user.op == 'call_module':
                    call_module_users[node.name].add(user.name)
        return outputs_spec, call_module_users

    def extract_fw_submod(node, submod):
        # process input
        inputs_spec = []
        inputs_users = []
        injected_states = {
            StateType.PARAMS: {},
            StateType.BUFFERS: {},
        }

        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            states_used_by[arg.name].append(node.name)
            try:
                if arg.name in inv_params:
                    injected_states[StateType.PARAMS][arg.name] = params.pop(inv_params[arg.name])
                elif arg.name in inv_buffers:  # inject states to the first submod
                    injected_states[StateType.BUFFERS][arg.name] = buffers.pop(
                        inv_buffers[arg.name])
            except KeyError:
                name = inv_params[arg.name] if arg.name in inv_params else inv_buffers[arg.name]
                typ = StateType.PARAMS if arg.name in inv_params else StateType.BUFFERS
                raise RuntimeError(
                    f"{typ}:{name}({arg.name}) is found used by multiple forward submods {states_used_by[arg.name]}"
                )

        # process output
        outputs_spec, call_module_users = extract_output(node)

        return EDGraphModule(submod, SubmodType.FW, inputs_spec, injected_states, outputs_spec,
                             call_module_users, node.target)

    def extract_bw_submod(node, submod):
        # process input
        inputs_spec = []
        inputs_users = []
        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            states_used_by[arg.name].append(node.name)

        # process output
        outputs_spec, call_module_users = extract_output(node)

        return EDGraphModule(submod, SubmodType.BW, inputs_spec, {}, outputs_spec,
                             call_module_users, node.target)

    def extract_step_submod(node, submod):
        # process input
        inputs_spec = []
        inputs_users = []

        injected_states = {
            StateType.OPTIMSTATES: {},
        }
        for arg in node.args:
            inputs_spec.append(arg.name)
            inputs_users.append({user.name for user in arg.users})
            states_used_by[arg.name].append(node.name)
            if arg.name in inv_optimstates:  # inject states to the first submod (might be fw_gm or step_gm but not bw_gm)
                try:
                    torch_state_name = inv_optimstates[arg.name]
                    injected_states[StateType.OPTIMSTATES][
                        arg.name] = optimstates[torch_state_name].pop(
                            inv_optimstates_type[arg.name])
                except KeyError:
                    name = inv_optimstates[arg.name]
                    typ = StateType.OPTIMSTATES
                    assert False, f"Please report this, {typ}:{name}({arg.name}) is found used by multiple step submods {states_used_by[arg.name]}"

        # process output
            outputs_spec, call_module_users = extract_output(node)

        return EDGraphModule(submod, SubmodType.STEP, inputs_spec, injected_states, outputs_spec,
                             call_module_users, node.target)

    compiled_meta = CompiledMeta(
        nstages=nstages,
        params_names_unflatten=params_names_unflatten,
        buffers_names_unflatten=buffers_names_unflatten,
        optimstates_names_unflatten=optimstates_names_unflatten,
        args_names_unflatten=args_names_unflatten,
        kwargs_names_unflatten=kwargs_names_unflatten,
        inv_params=inv_params,
        inv_buffers=inv_buffers,
        inv_optimstates=inv_optimstates,
        inv_optimstates_type=inv_optimstates_type,
        output_params_names_unflatten=output_params_names_unflatten,
        output_buffers_names_unflatten=output_buffers_names_unflatten,
        output_optimstates_names_unflatten=output_optimstates_names_unflatten,
        output_optimstates_names_flatten=output_optimstates_names_flatten,
        nones_or_grads_names_unflatten=nones_or_grads_names_unflatten,
        returns_names_unflatten=returns_names_unflatten,
        output2input_params=output2input_params,
        output2input_buffers=output2input_buffers,
        output2input_optimstates=output2input_optimstates,
        node_to_stage_idx=None)

    current_stateful_fw_bw = None
    compiled_stages: List[CompiledStage] = []
    submod_idx = 0
    for node in splited_global.graph.nodes:
        if node.op == 'call_module':
            assert len(node.kwargs) == 0, "splited_model should have no kwargs"
            submod = getattr(splited_global, node.target)
            if submod_idx % 2 == 0:  # fw or fw_bw gm
                splited_fw_bw_gm, part_cnt = split_by(submod,
                                                      torch.ops.easydist.fw_bw_split.default)
                save_graphviz_dot(splited_fw_bw_gm, f"fw_bw.dot")
                assert part_cnt == nstages * 2 if is_backward_called else nstages, "part_cnt should be nstages * 2 if backward is called"
                stateful_fw_bw = []
                fw_bw_submod_idx = 0
                for n in splited_fw_bw_gm.graph.nodes:
                    if n.op == 'call_module':  # extract submods
                        assert len(n.kwargs) == 0, "splited_model should have no kwargs"
                        fw_or_bw_submod = getattr(splited_fw_bw_gm, n.target)
                        is_fw = (not is_backward_called or fw_bw_submod_idx < part_cnt // 2)
                        stateful_fw_bw.append(
                            extract_fw_submod(n, fw_or_bw_submod) if is_fw else extract_bw_submod(
                                n, fw_or_bw_submod))
                        fw_bw_submod_idx += 1
                assert current_stateful_fw_bw is None, "There should be no consecutive compiled_fw_bw"
                current_stateful_fw_bw = stateful_fw_bw
            else:  # optimizer gm
                assert is_backward_called and current_stateful_fw_bw is not None, "There should be a stateful_bw_fw before optimizer step"
                step_gm_global = extract_step_submod(node, submod)
                for fw_gm, bw_gm in zip(current_stateful_fw_bw[:nstages],
                                        reversed(current_stateful_fw_bw[nstages:])):
                    compiled_stage = CompiledStage(compiled_meta, fw_gm, bw_gm, step_gm_global)
                    compiled_stages.append(compiled_stage)

                assert len(
                    step_gm_global.injected_states[StateType.OPTIMSTATES]
                ) == 0, "All states of step_gm_global should have been injected to step_gm"
                current_stateful_fw_bw = None
            submod_idx += 1

    if params or buffers or any(optimstates.values()):
        debugging_info = textwrap.dedent(f"""
            Some states were erased, if this is as intended, set strict=False
            Erased: 
                Params: 
                    {' '.join(params)}
                Buffers: 
                    {' '.join(buffers)}
                Optimstates: 
                    {' '.join(optimstates)}
            """)
        if strict:
            raise RuntimeError(debugging_info)
        else:
            logging.warning(debugging_info)

    if current_stateful_fw_bw is not None:  # fw or fw_bw
        if is_backward_called:
            for fw_gm, bw_gm in zip(current_stateful_fw_bw[:nstages],
                                    reversed(current_stateful_fw_bw[nstages:])):
                compiled_stage = CompiledStage(compiled_meta, fw_gm, bw_gm, None)
                compiled_stages.append(compiled_stage)
        else:
            for fw_gm in current_stateful_fw_bw:
                compiled_stage = CompiledStage(compiled_meta, fw_gm, None, None)
                compiled_stages.append(compiled_stage)
        current_stateful_fw_bw = None

    g = fx.Graph()
    env = {}
    submod_idx = 0
    name_to_stage_idx = {}
    all_states_names_flatten = set(
        pytree.tree_flatten(
            [params_names_unflatten, buffers_names_unflatten, optimstates_names_unflatten])[0])

    for node in splited_fw_bw_gm.graph.nodes:
        if node.op == 'placeholder':
            if len(node.users) > 0 and node.name not in all_states_names_flatten:
                env[node.name] = g.placeholder(node.name)
        elif node.op == 'call_module':
            if is_backward_called:  # fw_bw
                if submod_idx < nstages:  # forward
                    stage_idx = submod_idx
                    stage = compiled_stages[submod_idx]
                    node_name = f'stage_{submod_idx}_fw'
                    out_maybe_tuple = g.create_node(
                        name=node_name,
                        op='call_function',
                        target=stage.forward,
                        kwargs={arg_name: env[arg_name]
                                for arg_name in stage.fw_func_args})
                    name_to_stage_idx[node_name] = stage_idx
                    for arg_name in stage.fw_func_args:
                        if env[arg_name].op == 'placeholder':
                            name_to_stage_idx[arg_name] = stage_idx
                    for output in stage.fw_func_returns:
                        env[output] = g.create_node(name=output,
                                                    op='call_function',
                                                    target=operator.getitem,
                                                    args=(out_maybe_tuple, output))
                        name_to_stage_idx[output] = stage_idx
                else:  # backward
                    stage_idx = 2 * nstages - submod_idx - 1
                    stage = compiled_stages[stage_idx]
                    node_name = f'stage_{stage_idx}_bw'
                    out_maybe_tuple = g.create_node(
                        name=node_name,
                        op='call_function',
                        target=stage.backward,
                        kwargs={arg_name: env[arg_name]
                                for arg_name in stage.bw_func_args})
                    name_to_stage_idx[node_name] = stage_idx
                    for output in stage.bw_gm.outputs_spec:
                        if not output in outputs_names_flatten:
                            env[output] = g.create_node(name=output,
                                                        op='call_function',
                                                        target=operator.getitem,
                                                        args=(out_maybe_tuple, output))
                            name_to_stage_idx[output] = stage_idx
                    if hasattr(stage, 'step_gm'):
                        g.create_node(name=f'stage_{stage_idx}_step',
                                      op='call_function',
                                      target=stage.step)
                        name_to_stage_idx[f'stage_{stage_idx}_step'] = stage_idx
            else:  # fw
                stage_idx = submod_idx
                stage = compiled_stages[submod_idx]
                node_name = f'stage_{submod_idx}_fw'
                out_maybe_tuple = g.create_node(
                    name=node_name,
                    op='call_function',
                    target=stage.forward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.fw_func_args})
                name_to_stage_idx[node_name] = stage_idx
                for arg_name in stage.fw_func_args:
                    if env[arg_name].op == 'placeholder':
                        name_to_stage_idx[arg_name] = stage_idx
                for output in stage.fw_func_returns:
                    env[output] = g.create_node(name=output,
                                                op='call_function',
                                                target=operator.getitem,
                                                args=(out_maybe_tuple, output))
                    name_to_stage_idx[output] = stage_idx
            submod_idx += 1

    compiled_meta.node_to_stage_idx = name_to_stage_idx  # TODO @botbw move this to constructor?
    # def eliminate_dead_node():
    #     raise RuntimeError("This method should be called since the graph doesn't have output node")
    # setattr(g, 'eliminate_dead_node', eliminate_dead_node)
    local_gm = fx.GraphModule({}, g)

    return compiled_meta, compiled_stages, local_gm


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
