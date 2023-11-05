# Modified from PiPPy
from pippy.debug import map_debug_info
from typing import List
import copy
import operator
import logging
from enum import Enum
from typing import Dict, Optional, Union, Callable, Tuple, List, Any

import torch
from torch.fx.passes.split_module import split_module

from easydist.torch.experimental.compiler import _compile
from easydist.torch.experimental.init_helper import SetParaInitHelper
from easydist.torch.utils import get_input_signature


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


MultiUseParamSpec = Union[
    MultiUseParameterConfig, Dict[str, MultiUseParameterConfig]
]

_pipeline_tracer = None


def pipe_split():
    if _pipeline_tracer is not None and hasattr(_pipeline_tracer, "graph"):
        _pipeline_tracer.graph.call_function(pipe_split, (), {})


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


def annotate_split_points(
    mod: torch.nn.Module, spec: Dict[str, PipeSplitWrapper.SplitPoint]
):
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


def from_tracing(
    mod: torch.nn.Module,
    multi_use_param_spec: Optional[MultiUseParamSpec] = None,
    tracer=None,
    output_loss_value_spec=None,
    deep_copy_module=False,
    split_policy: Optional[
        Callable[[torch.fx.GraphModule], torch.fx.GraphModule]
    ] = None,
    return_to_0: bool = True,
    **kwargs,
):
    # TODO: abstract partitioning policy
    global _pipeline_tracer
    old__pipeline_tracer = _pipeline_tracer
    _pipeline_tracer = tracer or torch.fx.Tracer()
    try:
        # TODO: tracing policy
        if deep_copy_module:
            mod = copy.deepcopy(
                mod
            )  # because further pipe building activities can modify mod
        graph = _pipeline_tracer.trace(mod, **kwargs)

        traced = torch.fx.GraphModule(mod, graph)
    finally:
        _pipeline_tracer = old__pipeline_tracer

    if split_policy is not None:
        traced = split_policy(traced)

    return _from_traced(
        mod,
        traced,
        multi_use_param_spec,
        output_loss_value_spec=output_loss_value_spec,
        return_to_0=return_to_0,
    )


def _find_loss_from_output_and_spec(output_val, spec_val):
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, torch.fx.Node):
            raise RuntimeError(
                f"Loss spec must specify a dynamic value but got {output_val}"
            )
        return output_val

    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if len(output_val) != len(spec_val):
            raise RuntimeError(
                f"Output value {output_val} must match length of loss specification "
                f"{spec_val}"
            )
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(
            f"Did not find loss value in specification {spec_val}"
        )

    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(
                f"Output value {output_val} must match keys of loss specification "
                f"{spec_val}"
            )
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(
                output_val[k], spec_val[k]
            )
            if loss_val is not None:
                return loss_val
        raise RuntimeError(
            f"Did not find loss value in specification {spec_val}"
        )

    raise RuntimeError(
        f"Unsupported type {type(spec_val)} in loss specification"
    )


def _find_loss_output(
    mod: torch.nn.Module, g: torch.fx.Graph, output_loss_value_spec
):
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
        loss_node = _find_loss_from_output_and_spec(
            output_val, output_loss_value_spec
        )
        generated_spec = output_loss_value_spec

    return loss_node, output_node, generated_spec


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


def _insert_stage_symbolic_backward(
    g: torch.fx.Graph,
    loss_node: torch.fx.Node,
    output_node: torch.fx.Node,
    return_to_0: bool,
):
    # Collect metadata about tuple output values. TODO: move this to split_module or FX IR
    tuples: Dict[torch.fx.Node, Tuple] = {}
    for node in reversed(g.nodes):
        if node.op == "call_function":
            # In the forward pass, only emit placeholder, module calls, and
            # getitem calls. If we have a target other than getitem in this
            # (forward-only) code, there is a bug.
            assert node.target == operator.getitem, (
                "Found non-getitem call in forward pass. "
                "Please report a bug to PiPPy"
            )
            assert (
                len(node.args) == 2
            ), "Found malformed getitem call. Please report a bug to PiPPy"
            indexed_value, node_idx = tuple(node.args)

            # indexed_value is a collection that we are indexing into. It could
            # exist in the tuples map if we've processed another `getitem`
            # already.
            existing_list_size = (
                len(tuples[indexed_value]) if indexed_value in tuples else -1
            )
            new_list_size = max(node_idx + 1, existing_list_size)

            reconstructed_list = [None for _ in range(new_list_size)]

            # Copy over existing elements if present
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val

            # Populate value represented by this node
            reconstructed_list[node_idx] = node

            tuples[indexed_value] = tuple(reconstructed_list)

    # Keep track of nodes that dominate the loss node.
    # We will only emit backward operations for nodes that can contribute
    # to the specified loss value.
    live_nodes = {loss_node: None}
    val_to_grad: Dict[torch.fx.Node, Optional[torch.fx.Node]] = {
        loss_node: None
    }

    def assign_or_accumulate_grad(forward_node, grad_value):
        if forward_node in val_to_grad and forward_node.op != "placeholder":
            grad_value = g.call_function(
                _null_coalesce_accumulate,
                (val_to_grad[forward_node], grad_value),
            )
        val_to_grad[forward_node] = grad_value

    with g.inserting_before(output_node):
        barrier_tokens = []
        last_grads = None

        for node in reversed(g.nodes):
            if node not in live_nodes:
                continue

            def add_to_live_nodes(n):
                live_nodes.setdefault(n, None)

            torch.fx.node.map_arg(node.args, add_to_live_nodes)
            torch.fx.node.map_arg(node.kwargs, add_to_live_nodes)
            if node.op == "call_module":
                output_grads: Union[
                    Tuple[Optional[torch.fx.Node], ...], Optional[torch.fx.Node]
                ]
                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple(
                        val_to_grad.get(n, None) for n in tuples[node]
                    )
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes
                    ]
                else:
                    stage_output = (node,)
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]

                output_grads = (
                    (output_grads,)
                    if not isinstance(output_grads, tuple)
                    else output_grads
                )

                # TODO @botbw: get stage_backward in easydist
                grad_call = g.call_module(
                    'stage_backward',
                    kwargs={
                        "stage_output": stage_output,
                        "output_grads": output_grads,
                        "input_values": list(node.all_input_nodes),
                        "outputs_with_grads_idxs": outputs_with_grads_idxs,
                    }
                )
                # Insert backward stage debug info
                kwargs_copy = dict(grad_call.kwargs)
                kwargs_copy[
                    "stage_info"
                ] = f"{grad_call} for stage {node.format_node()}"
                grad_call.kwargs = kwargs_copy

                grad_call_proxy = torch.fx.Proxy(grad_call)
                grads, barrier_token = (
                    grad_call_proxy[0].node,
                    grad_call_proxy[1].node,
                )
                barrier_tokens.append(barrier_token)
                last_grads = grads

                input_nodes = list(node.all_input_nodes)
                grads_proxy = torch.fx.Proxy(grads)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)

        # Insert barrier call - reconnect the original pipeline output (output_node.args[0])
        # to go through the `sync_barrier` call, then make the pipeline output the output
        # of the sync_barrier call. When the driver gets the pipeline output, it is
        # guaranteed that all backwards jobs for that micro-batch have been executed.
        # When all micro-batch pipeline outputs are ready, gradients have been fully
        # computed and synchronized and the optimizer step can be applied.
        if return_to_0:
            barrier_call = g.call_function(
                sync_barrier, (output_node.args[0], barrier_tokens, last_grads)
            )
            output_node.args = (barrier_call,)

    return g


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

    def move_param_to_callee(
        root, callee_name, param_val, use_idx, is_buffer
    ):
        assert isinstance(param_val, torch.Tensor), (
            f"Expected '{node.target}' to be {torch.Tensor} but got {type(param_val)}."
            + (
                f" It might happen if module '{node.target}' was passed to some 'leaf function'"
                f"(see https://pytorch.org/docs/stable/fx.html#torch.fx.wrap). Please inspect "
                f"usages of '{node.target}' in the traced graph."
                if isinstance(param_val, torch.nn.Module)
                else ""
            )
        )
        callee = root.get_submodule(callee_name)
        new_param_name = f"moved_{node.target.replace('.', '_')}"
        assert not hasattr(
            callee, new_param_name
        ), f"Module {callee_name} already has a parameter named {new_param_name}"
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

            move_param_to_callee(
                split, user.target, param_val, use_idx, is_buffer
            )

            to_delete.append((mod_itr, atoms))

    # deferral deletion
    for mod_itr, atoms in to_delete:
        delattr(mod_itr, atoms[-1])

    split.graph.lint()
    split.recompile()

    # Handle multi-use parameters based on user's configuration
    # TODO: generalize this to sequential
    multi_use_param_spec = multi_use_param_spec or {}

    multi_use_params_qualnames: Dict[
        str, Optional[MultiUseParameterConfig]
    ] = {}
    for node in split.graph.nodes:
        if node.op == "get_attr" and len(node.users) > 1:
            multi_use_params_qualnames.setdefault(node.target)

    for param in multi_use_params_qualnames:
        if isinstance(multi_use_param_spec, MultiUseParameterConfig):
            multi_use_params_qualnames[param] = multi_use_param_spec
        elif isinstance(multi_use_param_spec, dict):
            multi_use_params_qualnames[param] = multi_use_param_spec.get(
                param, MultiUseParameterConfig.TRANSMIT
            )
        else:
            raise ValueError(
                "multi_use_param_spec must be MultiUseParamSpec enum or dict"
            )

    # TODO: do we maintain the invariant that `Node.users` is topologically ordered? I don't think so
    node_to_first_user: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in split.graph.nodes:
        for input in node.all_input_nodes:
            if input not in node_to_first_user:
                node_to_first_user[input] = node

    for node in split.graph.nodes:
        if (
            node.op == "get_attr"
            and node.target in multi_use_params_qualnames
        ):
            reuse_type = multi_use_params_qualnames[node.target]
            if reuse_type == MultiUseParameterConfig.TRANSMIT:
                first_user = node_to_first_user[node]
                assert first_user.op == "call_module"

                use_idx = delete_user_reference(
                    node, first_user, delete_node=False
                )

                atoms = node.target.split(".")
                mod_itr = split
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                param_val = getattr(mod_itr, atoms[-1])
                is_buffer = atoms[-1] in mod_itr._buffers

                callee_param_def = move_param_to_callee(
                    split, first_user.target, param_val, use_idx, is_buffer
                )

                delattr(mod_itr, atoms[-1])

                # Add extra output to the callee and switch references to the parameter
                # access in the pipeline graph to use this.
                submod = split.get_submodule(first_user.target)
                callee_output_nodes = [
                    n for n in submod.graph.nodes if n.op == "output"
                ]
                assert len(callee_output_nodes) == 1
                callee_output_node = callee_output_nodes[0]

                # TODO: zero outputs?
                if isinstance(callee_output_node.args[0], tuple):
                    new_output_args = callee_output_node.args[0] + (
                        callee_param_def,
                    )
                    callee_output_node.args = (new_output_args,)
                    new_output_idx = len(new_output_args) - 1
                    promoted_to_tuple = False
                else:
                    new_output_args = (
                        callee_output_node.args[0],
                        callee_param_def,
                    )
                    callee_output_node.args = (new_output_args,)
                    new_output_idx = len(new_output_args) - 1
                    promoted_to_tuple = True

                submod.graph.lint()
                submod.recompile()

                with split.graph.inserting_after(first_user):
                    if promoted_to_tuple:
                        # TODO: test this code path
                        orig_output_getitem = split.graph.call_function(
                            operator.getitem, (first_user, 0)
                        )
                        first_user.replace_all_uses_with(
                            orig_output_getitem
                        )
                        # HACK because the above replace_all_uses with ALSO replaced the instance
                        # of first_user within the getitem node we just added
                        orig_output_getitem.args = (
                            first_user,
                        ) + orig_output_getitem.args[1:]

                    transmitted_value_getitem = split.graph.call_function(
                        operator.getitem, (first_user, new_output_idx)
                    )
                    node.replace_all_uses_with(transmitted_value_getitem)
                    split.graph.erase_node(node)
            elif reuse_type == MultiUseParameterConfig.REPLICATE:
                for user in copy.copy(node.users):
                    use_idx = delete_user_reference(
                        node, user, delete_node=False
                    )
                    atoms = node.target.split(".")
                    mod_itr = split
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)
                    param_val = getattr(mod_itr, atoms[-1])
                    is_buffer = atoms[-1] in mod_itr._buffers

                    move_param_to_callee(
                        split, user.target, param_val, use_idx, is_buffer
                    )

                atoms = node.target.split(".")
                mod_itr = split
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)

                delattr(mod_itr, atoms[-1])

                split.graph.erase_node(node)
            else:
                raise ValueError(
                    f"Unknown multi-use config value {reuse_type} specified for {node.target}"
                )

    split.delete_all_unused_submodules()

    split.graph.lint()
    split.recompile()

    num_stages = _number_and_count_forward_stages(split)

    # TODO @botbw: deal with backward pass
    has_loss_and_backward = False
    generated_loss_spec = output_loss_value_spec

    if mod.training or output_loss_value_spec is not None:
        # set up backward module
        setattr(split, "stage_backward", StageBackward())
        loss_node, output_node, generated_loss_spec = _find_loss_output(
            mod, split.graph, output_loss_value_spec
        )
        if loss_node is not None:
            _insert_stage_symbolic_backward(
                split.graph,
                loss_node,
                output_node,
                return_to_0,
            )
            split.recompile()
            has_loss_and_backward = True
            logging.info(
                "Pipeline is in training mode, backward pass generated"
            )
        else:
            logging.warning(
                "Did not find any loss value from model output, your pipeline will be in inference mode. "
                "If you want your pipeline to be in training mode, please specify a loss value via "
                "`output_loss_value_spec`."
            )
    else:
        logging.info(
            "Pipeline is in evaluation mode, backward pass not generated"
        )

    return split

    # return Pipe(
    #     split,
    #     qualname_map,
    #     num_stages,
    #     has_loss_and_backward,
    #     generated_loss_spec,
    # )


def _analyze_node_size(
    gm: torch.fx.GraphModule,
) -> Dict[torch.fx.Node, Dict[str, int]]:
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

        if (
            accumulate_size + new_size <= threshold
        ):  # can accommodate this node in current bucket
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif (
            accumulate_size == 0 and new_size > threshold
        ):  # this node becomes a stage
            new_stage_before(node.next)
        else:  # cannot accommodate this node
            new_stage_before(node)
            accumulate_size = repeated_size + new_size
            accumulate_params.clear()
            accumulate_params.update(repeated_params)
            accumulate_params.update(new_params)

    # Insert pipe_split nodes at the recorded positions
    nstages = 1
    for node in insert_before_nodes:
        if nstages == max_stages:
            break
        with gm.graph.inserting_before(node):
            gm.graph.call_function(pipe_split, (), {})
        nstages += 1

    # Since we transformed the graph, we need to recompile the module
    gm.recompile()

    return gm, nstages


def split_into_equal_size(
    nstages: int = 1,
) -> Callable[[torch.fx.GraphModule], torch.fx.GraphModule]:
    def _split_into_nstages_equal_size(
        gm: torch.fx.GraphModule,
    ) -> torch.fx.GraphModule:
        param_size = 0
        for param in gm.parameters():
            param_size += param.numel()
        buffer_size = 0
        for buffer in gm.buffers():
            buffer_size += buffer.numel()

        total_size = param_size + buffer_size
        per_stage_size = total_size // nstages
        logging.debug(
            f"Total model size: {total_size}, "
            f"per stage size: {per_stage_size}"
        )

        gm, rv_nstages = _split_on_size_threshold_with_max_stages(
            gm, per_stage_size, nstages
        )
        assert rv_nstages == nstages
        return gm

    return _split_into_nstages_equal_size


class LossWrapper(torch.nn.Module):
    """
    LossWrapper is a convenient abstract class that allows you to wrap up both
    your model as well as its loss function and specify the connectivity between
    the inputs, model, loss function, and output value. Example::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    The above example defines a connectivity where we expect the forward/loss/backward
    training procedure to take two arguments (x and targets), pass x into the module
    to get the output of the feedforward computation, pass the model output and the
    targets value into the loss function, and get and return the loss value, which will
    be backpropagated by PiPPy. The above class would then be instantiated like::

        model = ... # instantiate the model
        loss_fn = torch.nn.MSELoss() # for the sake of demonstration

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """

    def __init__(self, module, loss_fn=None):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This instance of LossWrapper does not have an overridden"
            "forward(). Please implement forward() to specify the arguments, "
            "connection between the module and loss, and loss output "
            "value."
        )


class TrivialLossWrapper(LossWrapper):
    def forward(self, x, targets):
        model_out = self.module(x)
        return self.loss_fn(model_out, targets)

    loss_spec = True


class StageBackward(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, stage_output, output_grads, input_values, stage_info, outputs_with_grads_idxs):
        return StageBackward.stage_backward(stage_output, output_grads, input_values, stage_info, outputs_with_grads_idxs)

    @staticmethod
    def stage_backward(
        stage_output,
        output_grads,
        input_values,
        stage_info: str,
        outputs_with_grads_idxs: List[int],
    ):
        """
        Given the input value(s) and the corresponding gradient for the output
        value(s), compute and accumulate gradients for all parameter values (leaves
        in the autograd trace) as well as return a list of the gradients for the
        input values
        """

        try:
            stage_output_with_grads = [
                stage_output[i] for i in outputs_with_grads_idxs
            ]
            output_grads_with_grads = [
                output_grads[i] for i in outputs_with_grads_idxs
            ]

            # stage_output may be a composite datatype like dict. Extract all individual
            # tensor values here
            stage_output_tensors = []
            output_grad_tensors = []

            def extract_tensors_with_grads(output_val, grad_val):
                if isinstance(output_val, torch.Tensor):
                    if not output_val.requires_grad and output_val.grad_fn is None:
                        return
                    assert isinstance(
                        grad_val, (torch.Tensor, type(None))
                    ), f"Expected Tensor or None gradient but got {type(grad_val)}"
                    stage_output_tensors.append(output_val)
                    output_grad_tensors.append(grad_val)
                elif isinstance(output_val, (tuple, list)):
                    if grad_val is None:
                        return
                    assert isinstance(
                        grad_val, (tuple, list)
                    ), f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                    assert len(output_val) == len(grad_val)
                    for ov, gv in zip(output_val, grad_val):
                        extract_tensors_with_grads(ov, gv)
                elif isinstance(output_val, dict):
                    if grad_val is None:
                        return
                    assert isinstance(grad_val, dict)
                    assert set(output_val.keys()) == set(grad_val.keys())
                    for k in output_val.keys():
                        extract_tensors_with_grads(output_val[k], grad_val[k])
                else:
                    # Output is a non-tensor type; just ignore it
                    pass

            extract_tensors_with_grads(
                stage_output_with_grads, output_grads_with_grads
            )

            torch.autograd.backward(
                # type: ignore[arg-type]
                stage_output_tensors, grad_tensors=output_grad_tensors
            )

            grad_inputs = []
            for val in input_values:
                if isinstance(val, torch.Tensor):
                    grad_inputs.append(val.grad)
                else:
                    grad_inputs.append(None)

            # TODO: use `torch.autograd.grad`
            """
            inputs_with_grad = []
            for val in input_values:
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    inputs_with_grad.append(val)

            grad_inputs = torch.autograd.grad(
                stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
            )
            """

        except Exception as e:
            exc_msg = f"""
            Failed to run backward stage {stage_info}
            Stage output: {map_debug_info(stage_output)}
            Output gradient: {map_debug_info(output_grads)}
            Input: {map_debug_info(input_values)}
            """
            raise RuntimeError(exc_msg) from e

        barrier_token = None
        return grad_inputs, barrier_token


def sync_barrier(loss, barrier_tokens, last_grads):
    return loss, last_grads


# TODO: handling requires_grad=False dynamically. Can we analyze this during initial
# IR emission?
def _null_coalesce_accumulate(lhs, rhs):
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)
