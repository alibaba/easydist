# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import reduce
from abc import ABC, abstractmethod
import logging
import operator
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg

from easydist.torch.experimental.pp.microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from easydist.torch.experimental.pp.utils import modify_graph_op_device
from easydist.torch.experimental.pp.compile_pipeline import CompiledMeta, StateType, CompiledStage, graph_outputs_to_func_outputs_non_strict, func_inputs_to_graph_inputs_by_stages

logger = logging.getLogger(__name__)


def _make_tensor_from_meta(
    example_value: FakeTensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(example_value.size(), dtype=example_value.dtype, device=device)


class RecvInfo:

    def __init__(
        self,
        input_name: str,
        source: int,
        example_tensor: FakeTensor,
    ):
        self.input_name = input_name
        self.source = source
        self.example_tensor = example_tensor

    def __repr__(self):
        return f"RecvInfo(input={self.input_name}, source={self.source}, shape={self.example_tensor.size()})"


class StageKwargPlaceholder:
    pass


class Schedule(ABC):

    def __init__(self, pipeline_stage: 'PipelineStageBase'):
        assert isinstance(pipeline_stage, PipelineStageBase)
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
            all_fw_send_reqs += self.pipeline_stage.forward_one_chunk(chunk)
            logger.debug(f"[{self.pipeline_stage.group_rank}] Forwarded chunk {chunk}")

        # Backward starts here
        if self.pipeline_stage.bw_node is not None:
            for bwd_chunk in range(self.pipeline_stage.num_chunks):
                all_bw_send_reqs += self.pipeline_stage.backward_one_chunk(bwd_chunk)
                logger.debug(f"[{self.pipeline_stage.group_rank}] Backwarded chunk {bwd_chunk}")

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


class Schedule1F1B(Schedule):

    def __call__(self) -> None:
        all_fw_send_reqs: List[dist.Work] = []
        all_bw_send_reqs: List[dist.Work] = []
        num_warmup = self.pipeline_stage.num_stages - self.pipeline_stage.stage_idx
        num_warmup = min(num_warmup, self.pipeline_stage.num_chunks)

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(num_warmup):
            all_fw_send_reqs += self.pipeline_stage.forward_one_chunk(chunk)

        # 1F1B phase
        for fwd_chunk in range(num_warmup, self.pipeline_stage.num_chunks):
            bwd_chunk = fwd_chunk - num_warmup
            all_bw_send_reqs += self.pipeline_stage.backward_one_chunk_if_exists(bwd_chunk)
            all_fw_send_reqs += self.pipeline_stage.forward_one_chunk(fwd_chunk)

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.pipeline_stage.num_chunks - num_warmup,
                               self.pipeline_stage.num_chunks):
            all_bw_send_reqs += self.pipeline_stage.backward_one_chunk(bwd_chunk)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in all_fw_send_reqs:
            work.wait()

        for work in all_bw_send_reqs:
            work.wait()

        self.pipeline_stage.merge_output_chunks()

        if self.pipeline_stage.step_node is not None:
            self.pipeline_stage.step()


class PipelineStageBase:

    def __init__(
        self,
        schedule_cls: Type[Schedule],
        local_gm: fx.GraphModule,
        compiled_meta: CompiledMeta,
        stage_idx: int,
        compiled_stage: CompiledStage,
        node_metas: Dict[str, Dict[str, FakeTensor]],
        num_chunks: int,
        args_chunk_spec: Tuple[Optional[TensorChunkSpec]],
        kwargs_chunk_spec: Dict[str, TensorChunkSpec],
        outputs_chunk_spec: Tuple[Optional[TensorChunkSpec]],
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
    ):
        # meta info
        self.name = f'stage_{stage_idx}'
        self._init_fw_bw_step_nodes(local_gm)
        assert issubclass(schedule_cls, Schedule), "schedule_cls must be the Schedule class itself"
        self.schedule = schedule_cls(self)
        self.compiled_meta = compiled_meta
        self.stage_idx = stage_idx
        self.compiled_stage = compiled_stage
        self.num_chunks = num_chunks
        self.node_val_chunk_spec = self._get_graph_inputs_chunk(compiled_meta, args_chunk_spec,
                                                                kwargs_chunk_spec)
        self.outputs_chunk_spec = outputs_chunk_spec
        self.device = device
        self.group = group

        self.group_rank = dist.get_rank(group)
        self.num_stages = compiled_meta.nstages
        self.node_to_stage_idx = compiled_meta.node_to_stage_idx  # TODO refactor this mapping?

        if dist.get_world_size(self.group) > self.num_stages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused")

        # runtimes
        self.outputs_batch = {}  # Activation send requests of all chunk
        self.activations_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]

        # Prepare send/recv infrastructure
        self._init_communication(node_metas)
        # Cast submodule to device
        self._move_or_init_inject_states()
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

    def _get_graph_inputs_chunk(self, compiled_meta, args_chunk_spec, kwargs_chunk_spec):
        node_val_chunk_spec = {}
        if args_chunk_spec is None:
            args_chunk_spec = [None] * len(compiled_meta.args_names_unflatten)
        assert len(args_chunk_spec) == len(compiled_meta.args_names_unflatten)
        for node_name, arg_chunk_spec in zip(compiled_meta.args_names_unflatten, args_chunk_spec):
            node_val_chunk_spec = arg_chunk_spec
        if kwargs_chunk_spec is None:
            kwargs_chunk_spec = {k: None for k in compiled_meta.kwargs_names_unflatten}
        assert set(kwargs_chunk_spec.keys()) == set(compiled_meta.kwargs_names_unflatten)
        for node_name, kwarg_chunk_spec in kwargs_chunk_spec.items():
            node_val_chunk_spec[node_name] = kwarg_chunk_spec
        return node_val_chunk_spec

    def _move_or_init_inject_states(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        for name, tensor in self.compiled_stage.fw_gm.injected_states[StateType.PARAMS].items():
            if isinstance(tensor, FakeTensor) or tensor.is_meta:
                materialize_fn = self.compiled_meta.params_init_helpers[name]
                self.compiled_stage.fw_gm.injected_states[StateType.PARAMS][name] = materialize_fn(
                    tensor=tensor,
                    materialization_device=self.device
                )
            else:
                self.compiled_stage.fw_gm.injected_states[StateType.PARAMS][name] = tensor.to(
                    self.device
                )
        for name, tensor in self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS].items():
            if isinstance(tensor, FakeTensor) or tensor.is_meta:
                materialize_fn = self.compiled_meta.buffers_init_helpers[name]
                self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS][name] = materialize_fn(
                    tensor=tensor,
                    materialization_device=self.device
                )
            else:
                self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS][name] = tensor.to(
                    self.device)
        if self.step_node is not None:
            for name, tensor in self.compiled_stage.step_gm.injected_states[
                    StateType.OPTIMSTATES].items():
                if isinstance(tensor, FakeTensor) or tensor.is_meta:
                    materialize_fn = self.compiled_meta.optimstates_init_helpers[name]
                    self.compiled_stage.step_gm.injected_states[StateType.OPTIMSTATES][name] = materialize_fn(
                        tensor=tensor,
                        materialization_device=self.device
                    )
                else:
                    self.compiled_stage.step_gm.injected_states[
                        StateType.OPTIMSTATES][name] = tensor.to(self.device)

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
        self.fw_kwargs_recv_info: Dict[int, Dict] = {}
        for chunk in range(self.num_chunks):
            self.fw_kwargs_recv_info[chunk] = self._create_recv_buffers(node_metas, self.fw_node)

        self.fw_act_send_info = self._create_send_info(self.fw_node)

        if self.bw_node is not None:
            self.bw_args_recv_info: Dict[int, Tuple] = {}
            self.bw_kwargs_recv_info: Dict[int, Dict] = {}
            for chunk in range(self.num_chunks):
                self.bw_kwargs_recv_info[chunk] = self._create_recv_buffers(
                    node_metas, self.bw_node)

            self.bw_grad_send_info = self._create_send_info(self.bw_node)

    def get_stage_index_of_node_name(
        self,
        node_name: str,
    ):
        return self.node_to_stage_idx[node_name]

    def _create_send_info(self, node):
        # Output index: List of receiver ranks
        act_send_info: Dict[str, List] = {}

        for user in node.users:
            assert user.target is operator.getitem, "Output must be a dict"
            out_str = user.args[-1]
            assert isinstance(out_str, str)
            # Recursively find the real destination
            gi_dsts = act_send_info.setdefault(out_str, [])
            for gi_user in user.users:
                dst_rank = self.find_dst_rank(gi_user)
                gi_dsts.append(dst_rank)
            # Next `getitem` will point to the next output index

        logger.info(f"[{self.group_rank}] " f"Send info: {act_send_info}")
        return dict(sorted(act_send_info.items()))

    def _create_recv_buffers(
        self,
        node_metas,
        node,
    ):

        def create_recv_tensor(
            input_node,
            output_idx: Optional[int] = None,
        ):
            """
            Create a tensor for receiving the `output_idx`-th value from
            `input_node`
            """
            if input_node.op == "placeholder":
                # Do not create buffer for placeholder
                return StageKwargPlaceholder()

            example_value = node_metas[input_node.name]["val"]

            logger.info(f"[{self.group_rank}] "
                        f"Creating recv buffer for input '{input_node.name}' "
                        f"value index {output_idx}: {example_value.size()}")

            src_rank = self.get_stage_index_of_node_name(input_node.name)

            return RecvInfo(
                input_node.name,
                src_rank,
                example_value,
            )

        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        kwargs_recv_info = map_arg(node.kwargs, create_recv_tensor)

        return dict(sorted(kwargs_recv_info.items()))

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        return self.get_stage_index_of_node_name(user.name)

    def _recv_tensor(self, info, recv_reqs):
        logger.debug(f"[{self.group_rank}] "
                     f"Receiving tensor '{info.input_name}' from Rank {info.source}: "
                     f"{info.example_tensor.size()}")
        # Use async to parallelize recv of tensors
        peer_rank = self.stage_index_to_group_rank[info.source]
        buffer = _make_tensor_from_meta(info.example_tensor, self.device)
        work = dist.irecv(
            buffer,
            peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank),
            group=self.group,
        )
        recv_reqs.append(work)
        return buffer

    def recv_tensor_fn(
        self,
        reqs,
    ):
        return lambda info: self._recv_tensor(info, reqs)

    def split_input_kwargs(self, kwargs):
        self.kwargs_split = None
        if kwargs:
            _, self.kwargs_split = split_args_kwargs_into_chunks(
                (),
                kwargs,
                self.num_chunks,
                None,
                None,
            )

    def _recv_and_fill_inputs(
        self,
        kwargs_split,
        kwargs_recv_info,
        chunk: int,
    ):
        # Receive requests of a chunk
        recv_reqs: List[dist.Work] = []

        act_recv = self.recv_tensor_fn(recv_reqs)

        chunk_kwargs = {}
        if kwargs_split:
            chunk_kwargs = kwargs_split[chunk]

        composite_kwargs = {}

        for kw, info in kwargs_recv_info[chunk].items():
            if isinstance(info, RecvInfo):
                composite_kwargs[kw] = act_recv(info)
            else:
                composite_kwargs[kw] = chunk_kwargs[kw]

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

        for name, dst_stages in send_info.items():
            out = output_dict[name]
            for dst in dst_stages:
                logger.debug(f"[{self.group_rank}] " f"Sending tensor to Rank {dst}: {out.size()}")
                peer_rank = self.stage_index_to_group_rank[dst]
                work = dist.isend(
                    out,
                    peer_rank if self.group is None else dist.get_global_rank(
                        self.group, peer_rank),  # TODO
                    group=self.group,
                )
                send_reqs.append(work)

        return send_reqs

    def forward_one_chunk(
        self,
        chunk: int,
    ) -> List[dist.Work]:
        # Receive activations
        composite_kwargs_chunk = self._recv_and_fill_inputs(self.kwargs_split,
                                                            self.fw_kwargs_recv_info, chunk)

        # Compute forward
        outputs_chunk = self.compiled_stage.forward(self.activations_chunks[chunk],
                                                    self.outputs_chunks[chunk],
                                                    **composite_kwargs_chunk)

        # Send activations
        fw_send_reqs = self._send_output_dict(self.fw_act_send_info, outputs_chunk)
        return fw_send_reqs

    def backward_one_chunk(
        self,
        chunk: int,
    ) -> List[dist.Work]:
        # Receive grads
        composite_kwargs_chunk = self._recv_and_fill_inputs(self.bw_args_recv_info,
                                                            self.bw_kwargs_recv_info, chunk)

        # Compute backward
        outputs_chunk = self.compiled_stage.backward(self.activations_chunks[chunk],
                                                     self.outputs_chunks[chunk],
                                                     **composite_kwargs_chunk)

        # send grads
        bw_send_reqs = self._send_output_dict(self.bw_grad_send_info, outputs_chunk)
        return bw_send_reqs

    def backward_one_chunk_if_exists(self, chunk: int) -> List[dist.Work]:
        if self.bw_node is not None:
            return self.backward_one_chunk(chunk)
        return []

    def step(self):
        self.compiled_stage.step(self.outputs_batch)

    def step_if_exists(self):
        if self.step_node is not None:
            self.step()

    def clear_runtime_states(self):
        # Caching chunk outputs for final output merge or reduction
        for act_chunk in self.activations_chunks:
            assert len(act_chunk) == 0, "Activations should be cleared"
        # self.activations_chunks.clear()
        for outputs_chunk in self.outputs_chunks:
            outputs_chunk.clear()
        self.outputs_batch.clear()

    def merge_output_chunks(self) -> Dict[str, Any]:
        maybe_updated_params_names_unflatten = self.compiled_meta.output_params_names_unflatten
        maybe_updated_buffers_names_unflatten = self.compiled_meta.output_buffers_names_unflatten
        updated_optimstates_names_unflatten = self.compiled_meta.output_optimstates_names_unflatten
        nones_or_grads_names_unflatten = self.compiled_meta.nones_or_grads_names_unflatten
        returns_names_unflatten = self.compiled_meta.returns_names_unflatten

        params = {}
        buffers = {}
        optimstates = {}
        grads = []
        rets = []
        for chunk in self.outputs_chunks:
            grads_chunk = {}
            rets_chunk = [None for _ in range(len(returns_names_unflatten))]
            for node_name, tensor in chunk.items():
                if node_name in maybe_updated_params_names_unflatten.values():
                    params[node_name] = tensor
                elif node_name in maybe_updated_buffers_names_unflatten.values():
                    buffers[node_name] = tensor
                elif node_name in updated_optimstates_names_unflatten.values():
                    optimstates[node_name] = tensor
                elif node_name in nones_or_grads_names_unflatten.values():
                    grads_chunk[node_name] = tensor
                elif node_name in returns_names_unflatten:
                    rets_chunk[returns_names_unflatten.index(node_name)] = tensor
                else:
                    raise RuntimeError(f"Unknown output {node_name}")
            grads.append(grads_chunk)
            rets.append(rets_chunk)

        rets = merge_chunks(rets, self.outputs_chunk_spec)
        rets = {k: v for k, v in zip(returns_names_unflatten, rets)}
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

    def all_gather_outputs(self, rank):
        outputs_all_stages = [None for _ in range(self.num_stages)]
        outputs = graph_outputs_to_func_outputs_non_strict(self.compiled_meta, self.outputs_batch)
        dist.gather_object(outputs,
                           outputs_all_stages if self.group_rank == rank else None,
                           dst=rank)
        return outputs_all_stages

    def all_gather_state_dict(self, rank):
        state_dicts = [None for _ in range(self.num_stages)]
        state_dict = self.compiled_stage.state_dict()
        dist.gather_object(state_dict, state_dicts if self.group_rank == rank else None, dst=rank)
        return state_dicts

    def all_gather_optimizer_state_dict(self, rank):
        optimizer_state_dicts = [None for _ in range(self.num_stages)]
        optimizer_state_dict = self.compiled_stage.optimizer_state_dict()
        dist.gather_object(optimizer_state_dict,
                           optimizer_state_dicts if self.group_rank == rank else None,
                           dst=rank)
        return optimizer_state_dicts

    def __call__(self, *args, **kwargs) -> None:
        node_input_this_stage = [None]
        if self.is_first():
            node_inputs_all_stages = func_inputs_to_graph_inputs_by_stages(self.compiled_meta,
                                                                           *args,
                                                                           **kwargs,
                                                                           move_to_device=False)
        else:
            node_inputs_all_stages = [None] * self.num_stages

        # TODO: this is slow
        dist.scatter_object_list(node_input_this_stage, node_inputs_all_stages, src=0)

        node_input_this_stage = node_input_this_stage[0]
        for k, v in node_input_this_stage.items():
            if isinstance(v, torch.Tensor):
                node_input_this_stage[k] = v.to(self.device)

        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_input_kwargs(node_input_this_stage)

        self.schedule()

        logger.debug(f"[{self.group_rank}] All sends finished")

        return graph_outputs_to_func_outputs_non_strict(self.compiled_meta, self.outputs_batch)[-1]


def print_tensor_dict(chunk, di):
    print(f'Chunk {chunk}')
    for k, v in di.items():
        print(f'{k} size {v.size()} mean {v.float().mean()}')
