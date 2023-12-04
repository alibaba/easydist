import itertools
import operator
from collections import defaultdict
from typing import cast

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.nn.utils import stateless

from easydist.torch.decomp_utils import EASYDIST_DECOMP_TABLE
from torch._functorch.aot_autograd import (AOT_COUNTER, AOTConfig, create_aot_dispatcher_function,
                                           create_tree_flattened_fn, track_graph_compiling,
                                           create_runtime_wrapper, OutputType, call_func_with_args,
                                           TensorAlias, functionalized_rng_runtime_epilogue)
from torch.fx.experimental.proxy_tensor import is_sym_node
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
        is_export=True,  # to produce fx_g
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
            len(symint_outs_saved_for_bw)

    _indices_of_inps_to_detach = []
    bw_outs = [n for n in bw_module.graph.nodes if n.op == "output"][0].args[0]
    assert len(bw_outs) == len(fw_metadata.input_info) + fw_metadata.num_outputs_rng_offset
    for i, (bw_out) in enumerate(bw_outs):
        if bw_out is None:
            _indices_of_inps_to_detach.append(i)

    class CompiledForward:

        def __init__(self):
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
                disable_amp=torch._C._is_any_autocast_enabled())

        def init_before_forward(self):
            self.saved_tensors = []
            self.symints = None
            self.non_differentiable = None
            self.raw_returns = None

        def __call__(self, *args, **kwargs):
            self.init_before_forward()
            args = (self.named_params, self.named_buffers, *args)
            in_flat, _ = pytree.tree_flatten((args, kwargs))
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
            num_mutated_inputs = self.fw_metadata.num_mutated_inputs
            num_mutated_metadata_only_inputs = (self.fw_metadata.num_mutated_metadata_only_inputs)
            num_forward_returns = self.fw_metadata.num_forward_returns
            num_forward = self.fw_metadata.num_forward

            assert num_forward_returns == len(
                self.fw_metadata.requires_grad_info) + num_intermediate_bases

            # Partitioners must put symint arguments at the end separate from tensor arguments
            tensors_saved_for_backwards = fw_outs[
                self.fw_metadata.tensors_saved_for_backwards_slice]
            assert all(isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards)
            # See Note [Detaching saved tensors in AOTAutograd]
            self.saved_tensors = (x.detach() if x._is_view() else x
                                  for x in tensors_saved_for_backwards)
            symint_outs = fw_outs[self.fw_metadata.symints_saved_for_backwards_slice]
            assert all(
                isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                for x in symint_outs), str([type(x) for x in symint_outs])
            self.symints = symint_outs

            raw_returns = fw_outs[0:num_forward_returns]

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            if num_mutated_metadata_only_inputs > 0:
                for i, idx in enumerate(self.fw_metadata.mutated_inp_indices):
                    # We could make this faster by only looping over inputs with metadata-only mutations
                    # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                    info = self.fw_metadata.input_info[idx]
                    if info.mutates_metadata and not info.mutates_data:
                        raw_returns[i] = TensorAlias(raw_returns[i])

                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_inputs]
                    mut_inp_infos = [
                        x for x in self.fw_metadata.input_info
                        if x.mutates_data or x.mutates_metadata
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
            raw_returns_not_including_intermediate_bases = raw_returns[:num_mutated_inputs +
                                                                       num_outputs]
            fw_outs_not_requiring_grad = [
                x for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
                if isinstance(x, torch.Tensor) and not self.fw_metadata.requires_grad_info[i]
            ]
            self.non_differentiable = fw_outs_not_requiring_grad

            functionalized_rng_runtime_epilogue(self.fw_metadata,
                                                fw_outs[num_forward_returns:num_forward],
                                                return_new_outs=False)
            self.raw_returns = tuple(raw_returns)
            return self.raw_returns

    compiled_fw = CompiledForward()

    class CompiledBackward:

        def __init__(self):
            self.bw_gm = bw_module
            self.compiled_fw = compiled_fw

        def __call__(self, *args,
                     **kwargs):  # something like partial(grad, inputs=state_dict_of_this_stage)
            return self.backward(*args, **kwargs)

        def backward(
            self, *tangent_inputs
        ):  # bw_module needs saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs
            num_mutated_inps = self.compiled_fw.fw_metadata.num_mutated_inputs
            num_intermediate_bases = self.compiled_fw.fw_metadata.num_intermediate_bases
            expected_grad_outs = (self.compiled_fw.fw_metadata.num_outputs + num_mutated_inps +
                                  num_intermediate_bases)

            assert len(tangent_inputs) == expected_grad_outs
            out_info = self.compiled_fw.fw_metadata.output_info
            if (self.compiled_fw.fw_metadata.num_mutated_metadata_only_inputs > 0
                    or self.compiled_fw.fw_metadata.num_outputs_aliased > 0):
                inp_tangents, out_tangents, intermediate_base_tangents = (
                    tangent_inputs[0:num_mutated_inps],
                    tangent_inputs[num_mutated_inps:num_mutated_inps +
                                   self.compiled_fw.fw_metadata.num_outputs],
                    tangent_inputs[num_mutated_inps + self.compiled_fw.fw_metadata.num_outputs:],
                )
                # input_info contains info on *every* input,
                # But in the backward(), we are only given grad outputs for every mutated input.
                # We then need to filter out the grad outputs that correspond to metadata-only mutations.
                mutated_inp_indices = self.compiled_fw.fw_metadata.mutated_inp_indices
                input_info = self.compiled_fw.fw_metadata.input_info
                assert len(inp_tangents) == len(mutated_inp_indices)
                inp_tangents_filtered = [
                    x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
                    if input_info[info_idx].mutates_data
                ]
                # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
                out_tangents_filtered = [
                    x for x, info in zip(out_tangents, out_info) if info.output_type in [
                        OutputType.non_alias, OutputType.unsafe_view_alias,
                        OutputType.custom_function_view
                    ] and issubclass(info.raw_type, torch.Tensor)
                ]
                # intermediate bases always require gradients, and always participate in the backward graph.
                flat_bw_args = itertools.chain(inp_tangents_filtered, out_tangents_filtered,
                                               intermediate_base_tangents)
            else:
                # filter out non-tensor grad_outputs (aka due to ints being returned as outputs in the forward)
                num_mutated_inps = self.compiled_fw.fw_metadata.num_mutated_inputs
                mutated_inp_args = tangent_inputs[:num_mutated_inps] if num_mutated_inps > 0 else []
                user_tangents = tangent_inputs[num_mutated_inps:]
                assert len(user_tangents) == len(out_info)
                filtered_user_tangents = [
                    x for x, info in zip(user_tangents, out_info)
                    if issubclass(info.raw_type, torch.Tensor)
                ]
                flat_bw_args = tuple(mutated_inp_args) + tuple(filtered_user_tangents)

            contiguous_args = [t.contiguous() if torch.is_tensor(t) else t for t in flat_bw_args]

            rng_args = []
            if self.compiled_fw.fw_metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

            all_args = [
                *self.compiled_fw.symints, *self.compiled_fw.saved_tensors, *contiguous_args,
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
        self.compiled_submods = {}

    def run(self, *args, initial_env=None):
        self.compiled_submods.clear()
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):
        assert 'submod' in target

        submod = self.fetch_attr(target)
        compiled_fw, compiled_bw = compile_submod(submod, args, kwargs)
        self.compiled_submods[target] = (compiled_fw, compiled_bw)

        return super().call_module(target, args, kwargs)


def compile_with_fake(gm_split, fake_params_flatten, fake_buffers_flatten, fake_args):
    executor = CompileInterpreter(gm_split)
    with stateless._reparametrize_module(cast(torch.nn.Module, gm_split), {
            **fake_params_flatten,
            **fake_buffers_flatten
    },
                                         tie_weights=True):
        executor.run(*fake_args)
    compiled_submods = executor.compiled_submods
    return compiled_submods


def build_fw_gm(gm_split, compiled_submods):
    forward_graph = fx.Graph()
    forward_env = {}

    for node in gm_split.graph.nodes:
        assert node.op in ['placeholder', 'call_function', 'call_module',
                           'output'], "No other nodes should appear, please report a bug"

        if node.op == 'placeholder':
            forward_env[node] = forward_graph.placeholder(node.name)
        elif node.op == 'call_function':
            assert node.target == operator.getitem, "all function call are getitem"
            args, kwargs = pytree.tree_map_only(fx.Node, lambda x: forward_env[x],
                                                (node.args, node.kwargs))
            forward_env[node] = forward_graph.call_function(node.target, args, kwargs)
        elif node.op == 'call_module':
            assert 'submod' in node.target
            args, kwargs = pytree.tree_map(lambda x: forward_env[x], (node.args, node.kwargs))
            forward_env[node] = forward_graph.call_function(
                compiled_submods[node.target][0].__call__, args, kwargs)
        elif node.op == 'output':
            result = pytree.tree_map(lambda x: forward_env[x], node.args)
            assert len(result) == 1
            forward_env[node] = forward_graph.output(*result)

    forward_graph.eliminate_dead_code()
    forward_graph.lint()

    return fx.GraphModule({}, forward_graph)


# TODO @botbw: check and simpify this
def build_bw_gm(gm_split, compiled_submods):
    compiled_bwds = [(name, bw) for name, (_, bw) in compiled_submods.items()]
    idx = len(compiled_bwds) - 1

    backward_graph = fx.Graph()
    backward_env = defaultdict(list)
    backward_env[compiled_bwds[len(compiled_bwds) - 1][0]].append(
        (0, backward_graph.placeholder("out_grads")))

    output = {}

    class RuntimeWrapper:

        def __init__(self, compiled_bw):
            self.compiled_bw = compiled_bw

        def _get_args(self, out_grads):  # TODO @botbw: try to avoid doing this
            compiled_bw = self.compiled_bw
            expected_grad_outs = (
                compiled_bw.compiled_fw.fw_metadata.num_outputs + \
                    compiled_bw.compiled_fw.fw_metadata.num_mutated_inputs + \
                        compiled_bw.compiled_fw.fw_metadata.num_intermediate_bases
            )
            assert len(out_grads) <= expected_grad_outs
            args = [None] * expected_grad_outs
            args[-len(out_grads):] = out_grads
            return args

        def run_backward(self, *out_grads):
            compiled_bw = self.compiled_bw
            args = self._get_args(out_grads)
            out_all = compiled_bw(*args)
            if not isinstance(out_all, tuple):
                out_all = (out_all, )
            states = {
                **compiled_bw.compiled_fw.named_params,
                **compiled_bw.compiled_fw.named_buffers
            }
            grads = out_all[:len(states)]
            for state, grad in zip(states.values(), grads):
                state.grad = grad
            out_grads = out_all[len(states):]
            if len(out_grads) == 1:
                out_grads = out_grads[0]
            return out_grads

    def accumulate_grad_maybe_none(grad1, grad2):
        if grad1 is None:
            return grad2
        elif grad2 is None:
            return grad1
        else:
            return grad1 + grad2

    for node in reversed(gm_split.graph.nodes):
        assert node.op in ['placeholder', 'call_function', 'call_module',
                           'output'], "No other nodes should appear, please report a bug"
        if node.op == 'placeholder':
            break
        elif node.op == 'call_module':
            assert 'submod' in node.target and len(node.args) > 0
            name, compiled_bw = compiled_bwds[idx]
            wrapper = RuntimeWrapper(compiled_bw)
            args = [None] * len(backward_env[name])
            for id, out in backward_env[name]:
                if args[id] is None:
                    args[id] = out
                else:  # the output of this submod is used by multiple users
                    args[id] = backward_graph.call_function(accumulate_grad_maybe_none,
                                                            (args[id], out))
            args = tuple(x for x in args if x is not None)  # TODO @botbw: correct to do this?
            out = backward_graph.call_function(wrapper.run_backward, args)
            idx -= 1
            if len(node.args) == 1:  # no getitem call
                arg = node.args[0]
                assert arg.op in ['placeholder', 'call_module', 'call_function']
                if arg.op == 'placeholder':  # input node
                    if arg.name not in output:
                        output[arg.name] = out
                    else:
                        output[arg.name] = backward_graph.call_function(
                            accumulate_grad_maybe_none, (output[arg.name], out))
                    continue

                if arg.op == 'call_module':  # submod
                    prev_node = arg.name
                    id = 0
                else:  # getitem call
                    assert (len(arg.args) == 2
                            and isinstance(arg.args[1], int)) and arg.args[0].op == 'call_module'
                    prev_node = arg.args[0].name
                    id = arg.args[1]
                backward_env[prev_node].append((id, out))
            else:
                for i, arg in enumerate(node.args):
                    assert arg.op in ['placeholder', 'call_module', 'call_function']
                    if arg.op == 'placeholder':
                        if arg.name not in output:
                            output[arg.name] = backward_graph.call_function(
                                operator.getitem, (out, i))
                        else:
                            output[arg.name] = backward_graph.call_function(
                                accumulate_grad_maybe_none,
                                (output[arg.name],
                                 backward_graph.call_function(operator.getitem, (out, i))))
                        continue

                    if arg.op == 'call_module':
                        prev_node = arg.name
                        id = 0
                        _out = backward_graph.call_function(operator.getitem, (out, i))
                    else:
                        assert (len(arg.args) == 2 and isinstance(
                            arg.args[1], int)) and arg.args[0].op == 'call_module'
                        prev_node = arg.args[0].name
                        id = arg.args[1]
                        _out = backward_graph.call_function(operator.getitem, (out, i))
                    backward_env[prev_node].append((id, _out))

    seq_out = []
    for node in gm_split.graph.nodes:
        if node.op == 'placeholder':
            seq_out.append(output[node.name])
    backward_graph.output(seq_out)
    backward_graph.eliminate_dead_code()
    backward_graph.lint()

    return fx.GraphModule({}, backward_graph)


def compile_splited(gm_split: fx.GraphModule, *args):
    assert gm_split.training  # TODO @botbw: always compile forward and backward for now

    params_by_submod = {}
    buffers_by_submod = {}
    for name, mod in gm_split.named_children():
        params_by_submod[name] = dict(mod.named_parameters())
        buffers_by_submod[name] = dict(mod.named_buffers())

    fake_params_flatten = dict(gm_split.named_parameters())
    fake_buffers_flatten = dict(gm_split.named_buffers())
    fake_mode = detect_fake_mode(args)
    if not fake_mode: fake_mode = FakeTensorMode()

    def wrap_fake(x):
        if not isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, FakeTensor):
            return x
        else:
            return fake_mode.from_tensor(x)

    fake_params_flatten, fake_buffers_flatten, fake_args = pytree.tree_map(
        lambda x: wrap_fake(x), (fake_params_flatten, fake_buffers_flatten, args))

    compiled_submods = compile_with_fake(gm_split, fake_params_flatten, fake_buffers_flatten,
                                         fake_args)

    for name, (compiled_fw, _) in compiled_submods.items():
        compiled_fw.named_params = params_by_submod[name]
        compiled_fw.named_buffers = buffers_by_submod[name]

    fw_gm = build_fw_gm(gm_split, compiled_submods)
    bw_gm = build_bw_gm(gm_split, compiled_submods)

    class CompiledSplited:

        def forward(self, *args, **kwargs):
            return fw_gm(*args, **kwargs)

        def _get_args(self, out_grads, compiled_bw):  # TODO @botbw: try to avoid doing this
            expected_grad_outs = (
                compiled_bw.compiled_fw.fw_metadata.num_outputs + \
                    compiled_bw.compiled_fw.fw_metadata.num_mutated_inputs + \
                        compiled_bw.compiled_fw.fw_metadata.num_intermediate_bases
            )
            args = [None] * expected_grad_outs
            args[-len(out_grads):] = out_grads
            return args

        def backward(self, *out_grads):
            return bw_gm(*out_grads)

        def named_parameters(self):
            params = {}
            for _, (compiled_fw, _) in compiled_submods.items():
                params.update(compiled_fw.named_params)
            return params

        def named_buffers(self):
            buffers = {}
            for _, (compiled_fw, _) in compiled_submods.items():
                buffers.update(compiled_fw.named_buffers)
            return buffers

        def raw_returns(self):
            return list(compiled_submods.values())[-1][0].raw_returns

        def compiled_submods(self):
            return compiled_submods

    return CompiledSplited()
