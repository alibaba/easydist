# Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict
import logging
import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Type, Union


import torch
import torch.distributed as dist
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg
import torch.utils._pytree as pytree
from torch.distributed._tensor import DeviceMesh, mesh_resources

from easydist.torch.device_mesh import get_device_mesh, get_pp_group, get_pp_rank, spmd_device_mesh
from easydist.torch.experimental.pp.compile_pipeline import (
    CompiledMeta, CompiledStage, StateType,
    graph_outputs_to_func_outputs)
from easydist.torch.experimental.pp.microbatch import (DEFAULT_CHUNK_DIM, CustomReducer, TensorChunkSpec,
                                                       merge_chunks, split_args_kwargs_into_chunks)

logger = logging.getLogger(__name__)


def _make_tensor_from_meta(
    example_value: FakeTensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(example_value.size(), dtype=example_value.dtype, device=device)

class RecvBase:
    def __init__(self, input_name):
        self.input_name = input_name

class RecvInfo(RecvBase):

    def __init__(
        self,
        input_name: str,
        source: int,
        example_tensor: FakeTensor,
    ):
        super().__init__(input_name)
        self.source = source
        self.example_tensor = example_tensor

    def __repr__(self):
        return f"RecvInfo(input={self.input_name}, source={self.source}, shape={self.example_tensor.size()})"

class StageKwargPlaceholder(RecvBase):
    def __init__(self, input_name: str):
        super().__init__(input_name)


class Schedule(ABC):

    def __init__(self, pipeline_stage: 'PipelineStage'):
        assert isinstance(pipeline_stage, PipelineStage)
        self.pipeline_stage = pipeline_stage

    @abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError


class ScheduleGPipe(Schedule):

    def __call__(self) -> None:

        all_fw_send_reqs: List[dist.Work] = []
        all_bw_send_reqs: List[dist.Work] = []

        # Forward pass of all chunks
        for chunk in range(self.pipeline_stage.num_chunks):
            all_fw_send_reqs += self.pipeline_stage.forward_one_chunk()

        # Backward starts here
        if self.pipeline_stage.bw_node is not None:
            for bwd_chunk in range(self.pipeline_stage.num_chunks):
                all_bw_send_reqs += self.pipeline_stage.backward_one_chunk()

        # Wait for all sends to finish
        # TODO okay to delay the sync till completion of all chunks?
        for work in all_fw_send_reqs:
            work.wait()

        # Wait for all sends to finish
        # TODO okay to delay the sync till completion of all chunks?
        for work in all_bw_send_reqs:
            work.wait()

        self.pipeline_stage.merge_output_chunks()

        if self.pipeline_stage.step_node is not None:
            self.pipeline_stage.step()


class ScheduleDAPPLE(Schedule):

    def __init__(self, pipeline_stage: 'PipelineStage'):
        super().__init__(pipeline_stage)
        assert pipeline_stage.bw_node is not None, f"{type(self).__name__} requires backward node"

    def __call__(self) -> None:
        all_fw_send_reqs: List[dist.Work] = []
        all_bw_send_reqs: List[dist.Work] = []
        num_warmup = self.pipeline_stage.num_stages - self.pipeline_stage.stage_idx
        num_warmup = min(num_warmup, self.pipeline_stage.num_chunks)

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(num_warmup):
            all_fw_send_reqs += self.pipeline_stage.forward_one_chunk()

        # 1F1B phase
        for fwd_chunk in range(num_warmup, self.pipeline_stage.num_chunks):
            if self.pipeline_stage.bw_node is not None:
                all_bw_send_reqs += self.pipeline_stage.backward_one_chunk()
            all_fw_send_reqs += self.pipeline_stage.forward_one_chunk()

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.pipeline_stage.num_chunks - num_warmup,
                                self.pipeline_stage.num_chunks):
            all_bw_send_reqs += self.pipeline_stage.backward_one_chunk()

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in all_fw_send_reqs:
            work.wait()

        for work in all_bw_send_reqs:
            work.wait()

        self.pipeline_stage.merge_output_chunks()

        if self.pipeline_stage.step_node is not None:
            self.pipeline_stage.step()


def modify_graph_op_device(
    gm: torch.fx.GraphModule,
    new_device: torch.device,
):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function":
            for arg in node.args:
                if isinstance(arg, torch.device) and arg != new_device:
                    logger.debug(f"Changing device of Node {node.name} from {arg} to {new_device}")
                    arg = new_device
                    modified = True
            if "device" in node.kwargs and node.kwargs["device"] != new_device:
                logger.debug(
                    f"Changing device of Node {node.name} from {node.kwargs['device']} to {new_device}"
                )
                node.update_kwarg("device", new_device)
                modified = True
    if modified:
        gm.recompile()
class PipelineStage:

    def __init__(
        self,
        schedule_cls: Type[Schedule],
        local_gm: fx.GraphModule,
        stage_idx: int,
        compiled_meta: CompiledMeta,
        compiled_stage: CompiledStage,
        node_metas: Dict[str, Dict[str, FakeTensor]],
        num_chunks: int,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec]],
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]],
        returns_chunk_spec: Optional[Tuple[Union[TensorChunkSpec, CustomReducer]]],
        device: torch.device,
        group: dist.ProcessGroup,
        sharded_graph: fx.GraphModule,
        gather_output=True
    ):
        # meta info
        self.name = f'stage_{stage_idx}'
        self._init_fw_bw_step_nodes(local_gm)
        assert issubclass(schedule_cls, Schedule), "schedule_cls must be the Schedule class"
        self.schedule = schedule_cls(self)
        self.stage_idx = stage_idx
        self.compiled_meta = compiled_meta
        self.compiled_stage = compiled_stage
        self.num_chunks = num_chunks
        self.inputs_nodes_chunk_spec = self._get_inputs_nodes_spec(compiled_meta, args_chunk_spec,
                                                                kwargs_chunk_spec)
        self.returns_nodes_chunk_spec = self._get_returns_nodes_spec(compiled_meta, returns_chunk_spec)
        self.device = device
        self.group = group

        self.group_rank = dist.get_rank(group)
        self.num_stages = compiled_meta.nstages
        self.node_to_stage_idx = compiled_meta.node_to_stage_idx  # TODO refactor this mapping?
        self.return_to_all_stages = gather_output
        self.graph = sharded_graph
        if dist.get_world_size(self.group) > self.num_stages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused")

        # runtimes
        self.cur_fw_chunk_id = 0
        self.cur_bw_chunk_id = 0
        self.cur_step_chunk_id = 0
        self.kwargs_chunks = [{} for _ in range(self.num_chunks)]
        self.activations_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_batch = {}  # Activation send requests of all chunk

        # Prepare send/recv infrastructure
        self._init_communication(node_metas)
        # Cast submodule to device
        # self._move_or_init_inject_states()
        # Move ops argument to device
        self._move_ops_to_device()

    def _init_fw_bw_step_nodes(self, local_gm):
        # Find stage forward node in graph
        self.fw_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_fw':  # TODO is it good to use str as a tag?
                assert self.fw_node is None, "Multiple forward nodes found"
                self.fw_node = node
        if not self.fw_node:
            raise AssertionError(f"Cannot find {self.name} in graph")

        # Find stage backward node in graph
        self.bw_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_bw':  # TODO is it good to use str as a tag?
                assert self.bw_node is None, "Multiple backward nodes found"
                self.bw_node = node

        # Find stage step node in graph
        self.step_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_step':  # TODO is it good to use str as a tag?
                assert self.step_node is None, "Multiple step nodes found"
                self.step_node = node

    def _get_inputs_nodes_spec(self, compiled_meta: CompiledMeta, args_chunk_spec,
                                kwargs_chunk_spec):
        node_val_chunk_spec = {}
        args_nodes_flatten, _ = pytree.tree_flatten(compiled_meta.args_nodes_unflatten)
        args_chunk_spec = args_chunk_spec or [None] * len(args_nodes_flatten)  # input could be non tensor, use None instead of TensorChunkSpec
        args_chunk_spec_flatten, _ = pytree.tree_flatten(args_chunk_spec)
        assert len(args_chunk_spec_flatten) == len(args_nodes_flatten)
        for node_name, arg_chunk_spec in zip(compiled_meta.args_nodes_unflatten, args_chunk_spec_flatten):
            node_val_chunk_spec[node_name] = arg_chunk_spec

        kwargs_nodes_flatten, spec1 = pytree.tree_flatten(compiled_meta.kwargs_nodes_unflatten)
        kwargs_chunk_spec = kwargs_chunk_spec or {node_name: TensorChunkSpec(DEFAULT_CHUNK_DIM) for node_name in kwargs_nodes_flatten}
        kwargs_chunk_spec_flatten, spec2 = pytree.tree_flatten(kwargs_chunk_spec)
        assert spec1 == spec2
        for node_name, kwarg_chunk_spec in zip(kwargs_nodes_flatten, kwargs_chunk_spec_flatten):
            node_val_chunk_spec[node_name] = kwarg_chunk_spec
        return node_val_chunk_spec

    def _get_returns_nodes_spec(self, compiled_meta, returns_chunk_spec):
        returns_nodes_chunk_spec = {}
        returns_chunk_spec = returns_chunk_spec or [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(compiled_meta.returns_nodes_flatten)
        returns_chunk_spec_flatten, _ = pytree.tree_flatten(returns_chunk_spec)
        assert len(returns_chunk_spec_flatten) == len(compiled_meta.returns_nodes_flatten)
        for name, spec in zip(compiled_meta.returns_nodes_flatten, returns_chunk_spec_flatten):
            returns_nodes_chunk_spec[name] = spec
        return returns_nodes_chunk_spec

    # def _move_or_init_inject_states(self):
    #     # Move submodule to indicated device if possible
    #     # Note: we cannot move meta module to real devices because meta tensors
    #     # do not support to() method. One needs to do an in-place tensor swap in
    #     # that case.
    #     for name, tensor in self.compiled_stage.fw_gm.injected_states[StateType.PARAMS].items():
    #         if isinstance(tensor, FakeTensor) or tensor.is_meta:
    #             materialize_fn = self.compiled_meta.params_init_helpers[name]
    #             self.compiled_stage.fw_gm.injected_states[StateType.PARAMS][name] = materialize_fn(
    #                 tensor=tensor, materialization_device=self.device)
    #         else:
    #             self.compiled_stage.fw_gm.injected_states[StateType.PARAMS][name] = tensor.to(
    #                 self.device)
    #     for name, tensor in self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS].items():
    #         if isinstance(tensor, FakeTensor) or tensor.is_meta:
    #             materialize_fn = self.compiled_meta.buffers_init_helpers[name]
    #             self.compiled_stage.fw_gm.injected_states[
    #                 StateType.BUFFERS][name] = materialize_fn(tensor=tensor,
    #                                                           materialization_device=self.device)
    #         else:
    #             self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS][name] = tensor.to(
    #                 self.device)
    #     if self.step_node is not None:
    #         for name, tensor in self.compiled_stage.step_gm.injected_states[
    #                 StateType.OPTIMSTATES].items():
    #             if isinstance(tensor, FakeTensor) or tensor.is_meta:
    #                 materialize_fn = self.compiled_meta.optimstates_init_helpers[name]
    #                 self.compiled_stage.step_gm.injected_states[
    #                     StateType.OPTIMSTATES][name] = materialize_fn(
    #                         tensor=tensor, materialization_device=self.device)
    #             else:
    #                 self.compiled_stage.step_gm.injected_states[
    #                     StateType.OPTIMSTATES][name] = tensor.to(self.device)

    def _move_ops_to_device(self):
        # Today PT2 tracer does not treat `x.device` as a symbolic device;
        # instead, the device of tracing time got burned into the generated
        # code.  Here we provide a workaround for users to manually modify the
        # "device" kwarg of operations. Such operation may include:
        # `torch.ones`, `torch.zeros`, `torch.rand`, etc.
        modify_graph_op_device(self.compiled_stage.fw_gm.gm, self.device)
        if self.compiled_stage.has_bw():
            modify_graph_op_device(self.compiled_stage.bw_gm.gm, self.device)
        if self.compiled_stage.has_step():
            modify_graph_op_device(self.compiled_stage.step_gm.gm, self.device)

    def is_first(self):
        return self.stage_idx == 0

    def is_last(self):
        return self.stage_idx == self.num_stages - 1

    def _init_communication(self, node_metas):
        """
        Create send/recv infrastructures for activations (during forward) and
        gradients (during backward)
        """
        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        stage_index_to_group_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(self.group)
        for i in range(self.num_stages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            stage_index_to_group_rank.setdefault(i, peer_rank)
        self.stage_index_to_group_rank = stage_index_to_group_rank

        # chunk : Dict of kwarg buffers
        self.fw_kwargs_recv_info = self._create_recv_info(node_metas, self.fw_node)
        self.fw_act_send_info = self._create_send_info(self.fw_node)

        if self.bw_node is not None:
            self.bw_kwargs_recv_info = self._create_recv_info(node_metas, self.bw_node)
            self.bw_grad_send_info = self._create_send_info(self.bw_node)

    def get_stage_index_of_node_name(
        self,
        node_name: str,
    ):
        return self.node_to_stage_idx[node_name]

    def _create_send_info(self, node):  # TODO @botbw: simplify
        to_sort = []
        for user in node.users:
            assert user.target is operator.getitem, "Output must be a dict"
            out_str = user.args[-1]
            assert isinstance(out_str, str)
            for gi_user in user.users:
                dst_rank = self.find_dst_rank(gi_user)
                to_sort.append((dst_rank, out_str))

        to_sort.sort(key=lambda x: (x[0], x[1])) # send lower rank first and in alphabetical order

        act_send_info_by_stage = defaultdict(list)
        for dst_rank, out_str in to_sort:
            act_send_info_by_stage[dst_rank].append(out_str)

        return act_send_info_by_stage

    def _create_recv_info(
        self,
        node_metas,
        kwarg,
    ):
        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        to_sort = []
        for _, kwarg in kwarg.kwargs.items():
            if kwarg.op == "placeholder":
                to_sort.append((-1, kwarg.name))
                continue
            example_value = node_metas[kwarg.name]["val"]
            src_rank = self.get_stage_index_of_node_name(kwarg.name)
            to_sort.append((src_rank, kwarg.name, example_value))

        to_sort.sort(key=lambda x: (x[0], x[1]))  # receive lower rank first and in alphabetical order

        kwargs_recv_info = defaultdict(list)
        for x in to_sort:
            if x[0] == -1:
                kwargs_recv_info[0].append(StageKwargPlaceholder(x[1])) # args recev with rank 0 (lowest rank)
            else:
                src_rank, name, example_value = x
                kwargs_recv_info[src_rank].append(RecvInfo(name, src_rank, example_value))

        return kwargs_recv_info

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        return self.get_stage_index_of_node_name(user.name)

    def _recv_tensor(self, info: RecvInfo, recv_reqs: List[dist.Work]):
        logger.debug(f"[{self.group_rank}] "
                     f"Receiving tensor '{info.input_name}' from Rank {info.source}: "
                     f"{info.example_tensor.size()}")
        # Use async to parallelize recv of tensors
        peer_rank = self.stage_index_to_group_rank[info.source]
        # Create a buffer for receiving the tensor
        buffer = _make_tensor_from_meta(info.example_tensor, self.device)
        work = dist.irecv(
            buffer,
            peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank),
            group=self.group,
        )
        recv_reqs.append(work)
        return buffer

    def bind_with_recev_tensor_fn(
        self,
        reqs: List[dist.Work],
    ):
        return lambda info: self._recv_tensor(info, reqs)

    def split_input_kwargs(self, kwargs):
        if kwargs:
            _, self.kwargs_chunks = split_args_kwargs_into_chunks(
                (),
                kwargs,
                self.num_chunks,
                None,
                self.inputs_nodes_chunk_spec,
            )

    def _recv_and_fill_inputs(
        self,
        kwargs_recv_info,
        chunk: int,
    ):
        # Receive requests of a chunk
        recv_reqs: List[dist.Work] = []

        act_recv = self.bind_with_recev_tensor_fn(recv_reqs)

        chunk_kwargs = self.kwargs_chunks[chunk]

        composite_kwargs = {}
        for rank, info_list in kwargs_recv_info.items():
            for info in info_list:
                if isinstance(info, RecvInfo):
                    composite_kwargs[info.input_name] = act_recv(info)
                else:
                    composite_kwargs[info.input_name] = chunk_kwargs[info.input_name]

        chunk_kwargs.clear()

        # Wait for all recvs to finish
        for work in recv_reqs:
            work.wait()

        return composite_kwargs

    def _send_output_dict(
        self,
        send_info,
        output_dict,
    ) -> List[dist.Work]:
        assert isinstance(output_dict, dict), "Output must be a dict"

        # Send requests of a chunk
        send_reqs: List[dist.Work] = []

        for dst, nodes in send_info.items():
            for node in nodes:
                val = output_dict[node]
                logger.debug(f"[{self.group_rank}] "
                             f"Sending tensor to Rank {dst}: {val.size()}")
                peer_rank = self.stage_index_to_group_rank[dst]
                work = dist.isend(
                    val.contiguous(),
                    peer_rank if self.group is None else dist.get_global_rank(
                        self.group, peer_rank),  # TODO
                    group=self.group,
                )
                send_reqs.append(work)

        return send_reqs

    def forward_one_chunk(self) -> List[dist.Work]:
        # Receive activations and get args kwargs
        composite_kwargs_chunk = self._recv_and_fill_inputs(self.fw_kwargs_recv_info, self.cur_fw_chunk_id)

        # Compute forward
        outputs_chunk = self.compiled_stage.forward(self.activations_chunks[self.cur_fw_chunk_id],
                                                    self.outputs_chunks[self.cur_fw_chunk_id],
                                                    **composite_kwargs_chunk)

        # next chunk
        self.cur_fw_chunk_id += 1

        # Send activations
        fw_send_reqs = self._send_output_dict(self.fw_act_send_info, outputs_chunk)
        return fw_send_reqs

    def backward_one_chunk(self) -> List[dist.Work]:
        # Receive grads
        composite_kwargs_chunk = self._recv_and_fill_inputs(self.bw_kwargs_recv_info, self.cur_bw_chunk_id)

        # Compute backward
        outputs_chunk = self.compiled_stage.backward(self.activations_chunks[self.cur_bw_chunk_id],
                                                     self.outputs_chunks[self.cur_bw_chunk_id],
                                                     **composite_kwargs_chunk)

        # next chunk
        self.cur_bw_chunk_id += 1

        # send grads
        bw_send_reqs = self._send_output_dict(self.bw_grad_send_info, outputs_chunk)
        return bw_send_reqs

    def backward_one_chunk_if_exists(self) -> List[dist.Work]:
        if self.bw_node is not None:
            return self.backward_one_chunk()
        return []

    def step(self, step_chunk=False):
        if step_chunk:
            self.compiled_stage.step(self.outputs_chunks(self.cur_step_chunk_id))
            self.cur_step_chunk_id += 1
            return
        self.compiled_stage.step(self.outputs_batch)

    def clear_runtime_states(self):
        self.cur_fw_chunk_id = 0
        self.cur_bw_chunk_id = 0
        self.cur_step_chunk_id = 0
        # Caching chunk outputs for final output merge or reduction
        for kwargs_chunk in self.kwargs_chunks:
            kwargs_chunk.clear()
        for act_chunk in self.activations_chunks:
            assert len(act_chunk) == 0, "Activations should be cleared"
        # self.activations_chunks.clear()
        for outputs_chunk in self.outputs_chunks:
            outputs_chunk.clear()
        self.outputs_batch.clear()

    def merge_output_chunks(self) -> Dict[str, Any]:
        maybe_updated_params_names_unflatten = self.compiled_meta.output_params_nodes_unflatten
        maybe_updated_buffers_names_unflatten = self.compiled_meta.output_buffers_nodes_unflatten
        updated_optimstates_names_unflatten = self.compiled_meta.output_optimstates_nodes_unflatten
        nones_or_grads_names_unflatten = self.compiled_meta.nones_or_grads_nodes_unflatten
        returns_names_flatten = self.compiled_meta.returns_nodes_flatten

        params, buffers, optimstates, grads, rets = {}, {}, {}, [], []
        for chunk in self.outputs_chunks:
            grads_chunk, rets_chunk = {}, {node_name: None for node_name in returns_names_flatten}
            for node_name, tensor in chunk.items():
                if node_name in maybe_updated_params_names_unflatten.values():
                    params[node_name] = tensor
                elif node_name in maybe_updated_buffers_names_unflatten.values():
                    buffers[node_name] = tensor
                elif node_name in updated_optimstates_names_unflatten.values():
                    optimstates[node_name] = tensor
                elif node_name in nones_or_grads_names_unflatten.values():
                    grads_chunk[node_name] = tensor
                elif node_name in returns_names_flatten:
                    rets_chunk[node_name] = tensor
                else:
                    raise RuntimeError(f"Unknown output {node_name}")
            chunk.clear()
            grads.append(grads_chunk)
            rets.append(rets_chunk)

        rets = merge_chunks(rets, self.returns_nodes_chunk_spec)
        grads = reduce(lambda a, b: {k: torch.add(a[k], b[k]) for k in a}, grads)

        self.outputs_batch = {**params, **buffers, **optimstates, **grads, **rets}

        return self.outputs_batch

    def state_dict(self):
        return self.compiled_stage.state_dict()

    def load_state_dict(self, state_dict):
        self.compiled_stage.load_state_dict(state_dict)

    def optimizer_state_dict(self):
        return self.compiled_stage.optimizer_state_dict()

    def load_optimizer_state_dict(self, state_dict):
        self.compiled_stage.load_optimizer_state_dict(state_dict)

    # def gather_outputs(self, rank):
    #     outputs_all_stages = [None for _ in range(self.num_stages)]
    #     dist.gather_object(self.outputs_batch,
    #                        outputs_all_stages if self.group_rank == rank else None,
    #                        dst=rank)
    #     all_graph_outputs = {}
    #     if self.group_rank == rank:  # other rank receive None s
    #         for outputs in outputs_all_stages:
    #             all_graph_outputs.update(outputs)
    #     return graph_outputs_to_func_outputs(self.compiled_meta, all_graph_outputs, strict=False)

    def gather_state_dict(self, group_rank):
        state_dicts = [None for _ in range(self.num_stages)]
        state_dict = self.compiled_stage.state_dict()
        dist.gather_object(state_dict, state_dicts if self.group_rank == group_rank else None, dst=group_rank, group=self.group)
        return state_dicts

    def gather_optimizer_state_dict(self, group_rank):
        optimizer_state_dicts = [None for _ in range(self.num_stages)]
        optimizer_state_dict = self.compiled_stage.optimizer_state_dict()
        dist.gather_object(optimizer_state_dict,
                           optimizer_state_dicts if self.group_rank == group_rank else None,
                           dst=group_rank,
                           group=self.group)
        return optimizer_state_dicts

    def all_gather_outputs(self):
        outputs_all_gather = [None for _ in range(self.num_stages)]
        dist.all_gather_object(outputs_all_gather, self.outputs_batch, group=self.group)
        all_graph_outputs = {}
        for output_stage in outputs_all_gather:
            all_graph_outputs.update(output_stage)
        outputs = graph_outputs_to_func_outputs(self.compiled_meta, all_graph_outputs, strict=False)
        outputs = pytree.tree_map_only(torch.Tensor, lambda x: x.to(self.device), outputs)
        return outputs  # TODO @botbw: should be strict, but some states are erased

    def all_gather_returns(self):
        returns_all_gather = [None for _ in range(self.num_stages)]
        returns_nodes_flatten = {node_name: None for node_name in self.compiled_meta.returns_nodes_flatten}
        returns_batch = {node_name: val for node_name, val in self.outputs_batch.items() if node_name in returns_nodes_flatten}
        dist.all_gather_object(returns_all_gather, returns_batch, group=self.group)
        all_returns = {}
        for returns_stage in returns_all_gather:
            for k, v, in returns_stage.items():
                if v is not None:
                    all_returns[k] = v
        ret = graph_outputs_to_func_outputs(self.compiled_meta, all_returns, strict=False)[-1]
        ret = pytree.tree_map_only(torch.Tensor, lambda x: x.to(self.device), ret)
        return ret

    def __call__(self, *args, **kwargs) -> None:
        args_kwargs_vals_flatten, spec_val = pytree.tree_flatten((args, kwargs))
        args_kwargs_nodes_flatten, spec_node = pytree.tree_flatten((self.compiled_meta.args_nodes_unflatten, self.compiled_meta.kwargs_nodes_unflatten))
        assert spec_val == spec_node, "Mismatched args/kwargs"

        input_node_vals = {}
        for node, val in zip(args_kwargs_nodes_flatten, args_kwargs_vals_flatten):
            if isinstance(val, torch.Tensor):
                val = val.to(self.device)
            input_node_vals[node] = val

        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_input_kwargs(input_node_vals)

        self.schedule()

        logger.debug(f"[{self.group_rank}] All sends finished")

        if self.return_to_all_stages:
            ret = self.all_gather_returns()
        else:
            ret = graph_outputs_to_func_outputs(self.compiled_meta, self.outputs_batch, strict=False)[-1]
        return ret

    def run_with_graph(self, graph, *args, **kwargs):
        return self(*args, **kwargs)

def print_tensor_dict(chunk, di):
    print(f'Chunk {chunk}')
    for k, v in di.items():
        print(f'{k} size {v.size()} mean {v.float().mean()}')
