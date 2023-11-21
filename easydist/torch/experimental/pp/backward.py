'''
Adapted from https://github.com/pytorch/PiPPy/blob/83a2308f4a53ae36eba2f0c1b2b262d5d697d37b/pippy/backward.py
'''
import operator
from typing import Dict, List, Optional, Tuple, Union

import torch


class BackStage(torch.nn.Module):
    name = "backward_"

    def __init__(self, forward_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_name = forward_name
        self.name = self.name + str(forward_name)
        self._stage_backward = stage_backward
        self.compiled = False

    def forward(
        self,
        stage_output,
        output_grads,
        input_values,
        outputs_with_grads_idxs: List[int],
        stage_info: str,
    ):
        return self._stage_backward(stage_output, output_grads, input_values,
                                    outputs_with_grads_idxs, stage_info)


def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: List[int],
    stage_info: str,
):
    """
    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """

    try:
        stage_output_with_grads = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads_with_grads = [output_grads[i] for i in outputs_with_grads_idxs]

        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors = []
        output_grad_tensors = []

        def extract_tensors_with_grads(output_val, grad_val):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                assert isinstance(
                    grad_val,
                    (torch.Tensor,
                     type(None))), f"Expected Tensor or None gradient but got {type(grad_val)}"
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

        extract_tensors_with_grads(stage_output_with_grads, output_grads_with_grads)

        torch.autograd.backward(
            # type: ignore[arg-type]
            stage_output_tensors,
            grad_tensors=output_grad_tensors)

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

        def friendly_debug_info(v):
            if isinstance(v, torch.Tensor):
                return f"Tensor({v.shape}, grad={v.requires_grad})"
            else:
                return str(v)

        def map_debug_info(a):
            return torch.fx.node.map_aggregate(a, friendly_debug_info)

        exc_msg = f"""
        {repr(e)}
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


def _insert_stage_symbolic_backward(g: torch.fx.Graph, loss_node: torch.fx.Node,
                                    output_node: torch.fx.Node, return_to_0: bool,
                                    num_stages: int):
    # Collect metadata about tuple output values. TODO: move this to split_module or FX IR
    tuples: Dict[torch.fx.Node, Tuple] = {}
    for node in reversed(g.nodes):
        if node.op == "call_function":
            # In the forward pass, only emit placeholder, module calls, and
            # getitem calls. If we have a target other than getitem in this
            # (forward-only) code, there is a bug.
            assert node.target == operator.getitem, ("Found non-getitem call in forward pass. "
                                                     "Please report a bug to PiPPy")
            assert (len(
                node.args) == 2), "Found malformed getitem call. Please report a bug to PiPPy"
            indexed_value, node_idx = tuple(node.args)

            # indexed_value is a collection that we are indexing into. It could
            # exist in the tuples map if we've processed another `getitem`
            # already.
            existing_list_size = (len(tuples[indexed_value]) if indexed_value in tuples else -1)
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
    val_to_grad: Dict[torch.fx.Node, Optional[torch.fx.Node]] = {loss_node: None}

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
                output_grads: Union[Tuple[Optional[torch.fx.Node], ...], Optional[torch.fx.Node]]
                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple(val_to_grad.get(n, None) for n in tuples[node])
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes
                    ]
                else:
                    stage_output = (node, )
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]

                output_grads = ((output_grads, )
                                if not isinstance(output_grads, tuple) else output_grads)

                back_stage = BackStage(node.target)
                setattr(g.owning_module, back_stage.name, back_stage)
                grad_call = g.call_module(
                    back_stage.name,
                    args=(  # TODO @botbw: better way to do this? use kwargs instead
                        tuple(stage_output),
                        tuple(output_grads),
                        tuple(node.all_input_nodes),
                        tuple(outputs_with_grads_idxs),
                    ),
                )
                # Insert backward stage debug info
                args_copy = list(grad_call.args)
                args_copy.append(f"{grad_call} for stage {node.format_node()}")
                grad_call.args = tuple(args_copy)

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
            barrier_call = g.call_function(sync_barrier,
                                           (output_node.args[0], barrier_tokens, last_grads))
            output_node.args = (barrier_call, )

    return g
