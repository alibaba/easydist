'''
Adapted from https://github.com/pytorch/PiPPy/blob/83a2308f4a53ae36eba2f0c1b2b262d5d697d37b/pippy/IR.py#L280
'''
import logging
import operator
from collections import defaultdict
from enum import Enum
from typing import (Callable, Dict, List, Tuple)

import torch
import torch.fx as fx
from torch.fx._symbolic_trace import _Patcher
from torch.fx.passes.split_module import split_module

from easydist.utils import rgetattr, rsetattr

__tracer_global = None

__optim_split_point_cnt = 0


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

def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )

# https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class _FWBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):  # TODO @botbw: support kwargs: https://github.com/pytorch/pytorch/issues/96337
        fw_bw_split_point()
        need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
        args = tuple(arg.clone() if need_clone(arg) else arg
                     for arg in args)  # TODO @botbw: have to clone? (in case the following op is in-place)
        return args

    @staticmethod
    def backward(ctx, *grad_output):
        fw_bw_split_point()
        return grad_output


def fw_bw_split_func(*args, **kwargs):
    if len(kwargs):
        raise TypeError(
            "fw_bw_split_func() got an unexpected keyword argument '%s', autograd.Function haven't support kwargs yet, try SplitPoint.END to solve this" % list(kwargs.keys()))
    return _FWBWSplitFunc.apply(*args)

@fx.has_side_effect
def optim_split_point():
    global __optim_split_point_cnt  # TODO @botbw: currently only one optim stage
    if __optim_split_point_cnt > 0:
        return
    __optim_split_point_cnt += 1
    tracer_current = get_tracer_global()
    if tracer_current is not None and hasattr(tracer_current, "graph"):
        tracer_current.graph.call_function(optim_split_point, (), {})


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
        ret = None
        try:
            if self.split_point == self.SplitPoint.BEGINNING:
                args = fw_bw_split_func(*args, **kwargs)

            ret = self.mod(*args, **kwargs)
        finally:
            if self.split_point == self.SplitPoint.END:
                ret = _to_tuple(ret)
                ret = fw_bw_split_func(*ret)
                if len(ret) == 1:
                    ret = ret[0]
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


class SplitPatcher(_Patcher):

    def __init__(self, mod: torch.nn.Module, opt: torch.optim.Optimizer):
        super().__init__()
        self.mod = mod
        assert opt is None, "Haven't support optimizers yet"
        self.opt = opt

    def __enter__(self):
        patcher = super().__enter__()

        if self.mod:
            mod_cls = type(self.mod)
            orig_forward = mod_cls.forward

            def forward_wrapper(mod, *args, **kwargs):
                fw_bw_split_point(
                )  # TODO @botbw: better way of doing this, only one split point in forward (or the loss node might be in a seperate stage)
                ret = orig_forward(mod, *args, **kwargs)
                return ret

            patcher.patch_method(mod_cls, 'forward', forward_wrapper, deduplicate=False)

        if self.opt:
            opt_cls = type(self.opt)
            orig_step = opt_cls.step
            orig_zero_grad = opt_cls.zero_grad

            def step_wrapper(opt, *args, **kwargs):
                optim_split_point()
                orig_step(opt, *args, **kwargs)

            def zero_grad_wrapper(opt, *args, **kwargs):
                optim_split_point()
                orig_zero_grad(opt, *args, **kwargs)

            patcher.patch_method(opt_cls, 'step', step_wrapper, deduplicate=False)
            patcher.patch_method(opt_cls, 'zero_grad', zero_grad_wrapper, deduplicate=False)

        orig_backward = torch.Tensor.backward

        def backward_wrapper(tensor, *args, **kwargs):
            fw_bw_split_point()
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
    split = split_module(traced, mod, split_callback, qualname_map)

    # peephole to remove pipe_split
    for submodule in split.modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ("call_function", split_point):
                    submodule.graph.erase_node(node)
            submodule.recompile()

    split.graph.eliminate_dead_code()
    split.delete_all_unused_submodules()
    split.graph.lint()
    split.recompile()

    return split


# TODO @botbw: simplify
def compile_stateful_stages(fw_bw_gm, flat_model_states):
    phs_inject = defaultdict(list)
    name2state = {}
    ph2name = []
    out2idx = {}
    model_state_phs_cnt = 0

    for node in fw_bw_gm.graph.nodes:
        if node.op == 'placeholder':
            if model_state_phs_cnt < len(flat_model_states):
                name2state[node.name] = flat_model_states[model_state_phs_cnt]
            ph2name.append(node.name)
            model_state_phs_cnt += 1
        elif node.op == 'call_module':
            assert 'submod_' in node.target and len(
                node.kwargs) == 0, "splited_model should have no kwargs"
            for arg in node.args:
                if arg.name in name2state:
                    phs_inject[arg.name].append(node.name)
        elif node.op == 'output':
            for i, arg in enumerate(node.args[0]):
                if arg is None: continue
                out2idx[arg.name] = i

    for name, users in phs_inject.items():
        phs_inject[name] = users[0]  # inject to forward stage

    stateful_submods = {}
    for node in fw_bw_gm.graph.nodes:
        if node.op == 'call_module':  # extrace submods
            assert 'submod_' in node.target and len(
                node.kwargs) == 0, "splited_model should have no kwargs"
            submod_name = node.target
            submod = getattr(fw_bw_gm, submod_name)
            root = {submod_name: submod}

            g = fx.Graph()
            # process input
            args = []
            args_spec = []
            args_users = []
            injected_states = {}
            for arg in node.args:
                args.append(g.placeholder(arg.name))
                args_spec.append(arg.name)
                args_users.append({user.name for user in arg.users})
                if arg.name in phs_inject and phs_inject[arg.name] == node.name:
                    injected_states[arg.name] = name2state[arg.name]

            # call submod
            out_maybe_tuple = g.call_module(submod_name, tuple(args))

            # process output
            outputs = []
            output_spec = []
            output_users = []
            getitem_users = [
                user.op == 'call_function' and user.target == operator.getitem
                for user in node.users
            ]
            if all(getitem_users):  # output is tuple
                for output in node.users:
                    outputs.append(
                        g.call_function(operator.getitem, (out_maybe_tuple, output.args[1])))
                    output_spec.append(output.name)
                    output_users.append({user.name for user in output.users})
            else:  # output is value
                assert not any(getitem_users)
                outputs.append(out_maybe_tuple)
                output_spec.append(node.name)
                output_users.append({user.name for user in node.users})
            g.output(outputs)

            # new gm
            gm = fx.GraphModule(root, g)
            gm.graph.lint()
            assert not (hasattr(gm, 'args_spec') or hasattr(gm, 'args_users')
                        or hasattr(gm, 'injected_states') or hasattr(gm, 'output_spec')
                        or hasattr(gm, 'output_users'))
            gm.args_spec = args_spec
            gm.args_users = args_users
            gm.injected_states = injected_states
            gm.output_spec = output_spec
            gm.output_users = output_users
            stateful_submods[node.name] = gm

    class CompiledStage:

        def __init__(self, fw_gm, bw_gm, num_stage):
            self.fw_gm = fw_gm
            self.bw_gm = bw_gm

            self.injected_states = set(fw_gm.injected_states.keys())  # injected states

            self.fw_func_args = set(fw_gm.args_spec) - set(
                self.injected_states)  # args for forward func

            self.fw_gm_args_saved_for_bw = set(fw_gm.args_spec) & set(
                bw_gm.args_spec)  # saved args for bw_gm
            self.fw_gm_outputs_saved_for_bw = set(fw_gm.output_spec) & set(
                bw_gm.args_spec)  # saved outputs for bw_gm

            def isUsedByNext(node_users):  # TODO @botbw: better way to do this
                for user in node_users:
                    if user == 'output':
                        continue
                    elif int(user.split('_')[1]) <= num_stage:
                        return True
                return False

            self.fw_func_returns = {
                output_name: None
                for (output_name, users) in zip(fw_gm.output_spec, fw_gm.output_users)
                if isUsedByNext(users)
            }

            self.bw_func_args = set(bw_gm.args_spec) - set(self.fw_gm_args_saved_for_bw) - set(
                self.fw_gm_outputs_saved_for_bw)

            def isOutputArgs(node_users):
                return len(node_users) == 0 or 'output' in node_users

            fw_outputs_spec = {
                **{
                    arg_name: None
                    for (arg_name, users) in zip(fw_gm.args_spec, fw_gm.args_users) if isOutputArgs(users)
                },
                **{
                    output_name: None
                    for (output_name, users) in zip(fw_gm.output_spec, fw_gm.output_users) if isOutputArgs(users)
                }
            }
            bw_outputs_spec = {
                **{
                    arg_name: None
                    for (arg_name, users) in zip(bw_gm.args_spec, bw_gm.args_users) if isOutputArgs(users)
                },
                **{
                    output_name: None
                    for (output_name, users) in zip(bw_gm.output_spec, bw_gm.output_users) if isOutputArgs(users)
                }
            }
            self.stage_outputs_spec = set(fw_outputs_spec.keys()) | set(bw_outputs_spec.keys())

            self.args_saved_for_bw = {}
            self.outputs = {}

        def forward(self, **kwargs):
            assert set(kwargs.keys()) == self.fw_func_args, "forward args should be saved for bw"
            self.outputs.clear()
            kwargs4gm = {}
            for arg_name in self.fw_gm.args_spec:
                if arg_name in kwargs:
                    kwargs4gm[arg_name] = kwargs[arg_name]
                else:
                    kwargs4gm[arg_name] = self.fw_gm.injected_states[arg_name]

                if arg_name in self.fw_gm_args_saved_for_bw:
                    self.args_saved_for_bw[arg_name] = kwargs4gm[arg_name]

                if arg_name in self.stage_outputs_spec:
                    self.outputs[arg_name] = kwargs4gm[arg_name]

            output_from_gm = self.fw_gm(**kwargs4gm)

            ret = {}
            for output_name, output in zip(self.fw_gm.output_spec, output_from_gm):
                if output_name in self.fw_func_returns:
                    ret[output_name] = output

                if output_name in self.fw_gm_outputs_saved_for_bw:
                    self.args_saved_for_bw[output_name] = output

                if output_name in self.stage_outputs_spec:
                    self.outputs[output_name] = output

            return ret

        def backward(self, **kwargs):
            assert set(kwargs.keys()) == self.bw_func_args, "backward args should be saved for fw"
            kwargs4gm = {}
            for arg_name in self.bw_gm.args_spec:
                if arg_name in kwargs:
                    kwargs4gm[arg_name] = kwargs[arg_name]
                else:
                    kwargs4gm[arg_name] = self.args_saved_for_bw[arg_name]
                    self.args_saved_for_bw.pop(arg_name)

                if arg_name in self.stage_outputs_spec:
                    self.outputs[arg_name] = kwargs4gm[arg_name]

            assert len(self.args_saved_for_bw) == 0, "all backward args should be used"
            output_from_gm = self.bw_gm(**kwargs4gm)

            ret = {}
            for output_name, output in zip(self.bw_gm.output_spec, output_from_gm):
                if output_name in self.stage_outputs_spec:
                    self.outputs[output_name] = output
                else:
                    ret[output_name] = output
            return ret

    num_stage = len(stateful_submods) // 2  # TODO @botbw: correct to do this?
    compiled_stages = []
    for i in range(num_stage):
        fw_gm = stateful_submods[f'submod_{i + 1}']
        bw_gm = stateful_submods[f'submod_{num_stage * 2 - i}']
        compiled_stages.append(CompiledStage(fw_gm, bw_gm, num_stage))

    g = fx.Graph()
    env = {}
    submod_idx = 0

    for node in fw_bw_gm.graph.nodes:
        if node.op == 'placeholder':
            if node.name not in phs_inject and len(node.users) > 0:
                env[node.name] = g.placeholder(node.name)
        elif node.op == 'call_module':
            if submod_idx < num_stage:
                stage = compiled_stages[submod_idx]
                out_maybe_tuple = g.call_function(
                    stage.forward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.fw_func_args})
                for output in stage.fw_gm.output_spec:
                    if output in stage.fw_func_returns:
                        env[output] = g.call_function(operator.getitem, (out_maybe_tuple, output))
            else:
                stage = compiled_stages[2 * num_stage - submod_idx - 1]
                out_maybe_tuple = g.call_function(
                    stage.backward,
                    kwargs={arg_name: env[arg_name]
                            for arg_name in stage.bw_func_args})
                for output in stage.bw_gm.output_spec:
                    if not output in stage.stage_outputs_spec:
                        env[output] = g.call_function(operator.getitem, (out_maybe_tuple, output))
            submod_idx += 1

    def eliminate_dead_node():
        raise RuntimeError("This method should be called since the graph doesn't have output node")

    setattr(g, 'eliminate_dead_node', eliminate_dead_node)
    gm = fx.GraphModule({}, g)
    return ph2name, out2idx, compiled_stages, gm


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
    insert_before_nodes: List[torch.fx.Node] = []

    def new_stage_before(node):
        insert_before_nodes.append(node)

    # Track the parameters we have seen in the current bucket and their total size
    accumulate_size = 0
    accumulate_params: Dict = {}

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
            # For function, the parameter it uses can be shared with other functions seen previously
            for param_name, size in param_sizes.items():
                if param_name not in accumulate_params:  # new parameter
                    new_params.setdefault(param_name)
                    new_size += size
                else:  # repeated parameter; mark down; use later
                    repeated_params.setdefault(param_name)
                    repeated_size += size
        elif node.op == "call_module":
            # For module, we count its paramters as a single whole
            for param_name, size in param_sizes.items():
                new_size += size

        if (accumulate_size + new_size
                <= threshold):  # can accommodate this node in current bucket
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif (accumulate_size == 0 and new_size > threshold):  # this node becomes a stage
            new_stage_before(node.next)
        else:  # cannot accommodate this node
            new_stage_before(node)
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
        return PipeSplitWrapper(target_module, PipeSplitWrapper.SplitPoint.END)

    nstages = 1
    for node in insert_before_nodes:
        prev = node.prev
        if nstages == max_stages:
            break
        if prev.op == "call_function":
            prev.target = gen_func_wrapper(prev.target)
        else:
            assert prev.op == "call_module"
            rsetattr(gm, prev.target, gen_module_wrapper(rgetattr(gm, prev.target)))

        nstages += 1

    # Since we transformed the graph, we need to recompile the module
    gm.recompile()

    return gm, nstages
