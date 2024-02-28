# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import reduce
import logging
import operator
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg

from easydist.torch.experimental.pp.microbatch import merge_chunks, split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.utils import modify_graph_op_device, map_debug_info
from easydist.torch.experimental.pp.compile_pipeline import CompiledMeta, StateType, CompiledStage, process_outputs_non_strict

logger = logging.getLogger(__name__)


def _make_tensor_from_meta(
    example_value: FakeTensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        example_value.size(), dtype=example_value.dtype, device=device
    )


class RecvInfo:
    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        self.input_name = input_name
        self.source = source
        self.buffer = buffer

    def __repr__(self):
        return f"RecvInfo(input={self.input_name}, source={self.source}, shape={self.buffer.size()})"


class StageArgPlaceholder:
    pass


class StageKwargPlaceholder:
    pass


class PipelineStage:
    def __init__(
        self,
        local_gm: fx.GraphModule,
        compiled_meta: CompiledMeta,
        stage_idx: int,
        compiled_stage: CompiledStage,
        node_metas,
        num_chunks,
        args_chunk_spec,
        kwargs_chunk_spec,
        outputs_chunk_spec,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        self.local_gm = local_gm
        self.compiled_meta = compiled_meta
        self.stage_idx = stage_idx
        self.compiled_stage = compiled_stage
        self.node_metas = node_metas
        self.num_chunks = num_chunks
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.outputs_chunk_spec = outputs_chunk_spec
        self.device = device
        self.group = group

        self.num_stages = compiled_meta.nstages
        self.node_to_stage_idx = compiled_meta.node_to_stage_idx
        self.name = f'stage_{stage_idx}'

        self.outputs_batch = {}


        if dist.get_world_size(self.group) > self.num_stages:
            raise RuntimeError("Number of ranks is larger than number of stages, some ranks are unused")

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(group)

        # Activation send requests of all chunk
        self.all_fw_send_reqs: List[dist.Work] = []
        # Grad send requests of all chunk
        self.all_bw_send_reqs: List[dist.Work] = []
        # Caching chunk outputs for final output merge or reduction
        self.activations_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        # Find my forward node in graph
        self.fw_node = None
        for node in self.local_gm.graph.nodes:
            if node.name == f'{self.name}_fw':
                assert self.fw_node is None, "Multiple forward nodes found"
                self.fw_node = node
        if not self.fw_node:
            raise AssertionError(f"Cannot find {self.name} in graph")
        
        # Find my backward node in graph
        self.bw_node = None
        for node in self.local_gm.graph.nodes:
            if node.name == f'{self.name}_bw':
                assert self.bw_node is None, "Multiple backward nodes found"
                self.bw_node = node

        # Find my step node in graph
        self.step_node = None
        for node in self.local_gm.graph.nodes:
            if node.name == f'{self.name}_step':
                assert self.step_node is None, "Multiple step nodes found"
                self.step_node = node

        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        self.stage_index_to_group_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(group)
        for i in range(self.num_stages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            self.stage_index_to_group_rank.setdefault(i, peer_rank)

        # Prepare send/recv infrastructure
        self._prepare_send_recv_infra()
        del self.node_metas
        # Cast submodule to device
        self._move_inject_states_to_device()
        # Move ops argument to device
        self._move_ops_to_device()

    def _move_inject_states_to_device(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        for name, tensor in self.compiled_stage.fw_gm.injected_states[StateType.PARAMS].items():
            assert not (isinstance(tensor, FakeTensor) or tensor.is_meta)
            self.compiled_stage.fw_gm.injected_states[StateType.PARAMS][name] = tensor.to(self.device)
        for name, tensor in self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS].items():
            assert not (isinstance(tensor, FakeTensor) or tensor.is_meta)
            self.compiled_stage.fw_gm.injected_states[StateType.BUFFERS][name] = tensor.to(self.device)
        if self.step_node is not None:
            for name, tensor in self.compiled_stage.step_gm.injected_states[StateType.OPTIMSTATES].items():
                assert not (isinstance(tensor, FakeTensor) or tensor.is_meta)
                self.compiled_stage.step_gm.injected_states[StateType.OPTIMSTATES][name] = tensor.to(self.device)

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

    def _prepare_send_recv_infra(self):
        """
        Create send/recv infrastructures for activations (during forward) and
        gradients (during backward)
        """

        # chunk : Dict of kwarg buffers
        self.fw_kwargs_recv_info: Dict[int, Dict] = {}
        for chunk in range(self.num_chunks):
            self.fw_kwargs_recv_info[chunk] = self._create_act_recv_buffers(self.fw_node)
        
        self.fw_act_send_info = self._create_act_send_info(self.fw_node)

        if self.bw_node is not None:
            self.bw_args_recv_info: Dict[int, Tuple] = {}
            self.bw_kwargs_recv_info: Dict[int, Dict] = {}
            for chunk in range(self.num_chunks):
                self.bw_kwargs_recv_info[chunk] = self._create_act_recv_buffers(self.bw_node)

            self.bw_act_send_info = self._create_act_send_info(self.bw_node)


    def get_stage_index_of_node_name(
        self,
        node_name: str,
    ):
        return self.node_to_stage_idx[node_name]

    def _create_act_send_info(self, node):
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

    def _create_act_recv_buffers(
        self,
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
                return StageArgPlaceholder()

            example_value = self.node_metas[input_node.name]["val"]

            logger.info(
                f"[{self.group_rank}] "
                f"Creating recv buffer for input '{input_node.name}' "
                f"value index {output_idx}: {example_value.size()}"
            )

            src_rank = self.get_stage_index_of_node_name(input_node.name)
            buffer = _make_tensor_from_meta(example_value, self.device)

            return RecvInfo(
                input_node.name,
                src_rank,
                buffer,
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
        logger.debug(
            f"[{self.group_rank}] "
            f"Receiving tensor '{info.input_name}' from Rank {info.source}: "
            f"{info.buffer.size()}"
        )
        # Use async to parallelize recv of tensors
        peer_rank = self.stage_index_to_group_rank[info.source]
        work = dist.irecv(
            info.buffer,
            peer_rank
            if self.group is None
            else dist.get_global_rank(self.group, peer_rank),
            group=self.group,
        )
        recv_reqs.append(work)
        return info.buffer

    def recv_tensor_fn(
        self,
        reqs,
    ):
        return lambda info: self._recv_tensor(info, reqs)

    def split_inputs(self, args, kwargs):

        self.kwargs_split = None
        if args or kwargs:
            _, self.kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.num_chunks,
                None, #self.pipe.args_chunk_spec,
                None, #self.pipe.kwargs_chunk_spec,
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
        # Send requests of a chunk
        send_reqs: List[dist.Work] = []

        for name, dst_stages in send_info.items():
            out = output_dict[name]
            for dst in dst_stages:
                logger.debug(
                    f"[{self.group_rank}] "
                    f"Sending tensor to Rank {dst}: {out.size()}"
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                work = dist.isend(
                    out,
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank),  # TODO
                    group=self.group,
                )
                send_reqs.append(work)

        return send_reqs

    def forward_one_chunk(
        self,
        chunk: int,
    ):
        composite_kwargs_chunk = self._recv_and_fill_inputs(
            self.kwargs_split,
            self.fw_kwargs_recv_info,
            chunk
        )

        # Compute forward
        outputs_chunk = self.compiled_stage.forward(self.activations_chunks[chunk], self.outputs_chunks[chunk], **composite_kwargs_chunk)

        assert isinstance(outputs_chunk, dict), "Only dict output is supported"
        logger.debug(map_debug_info(outputs_chunk))

        # Send activations
        send_reqs = self._send_output_dict(self.fw_act_send_info, outputs_chunk)
        self.all_fw_send_reqs += send_reqs

    def backward_one_chunk(
        self,
        chunk: int,
    ):
        composite_kwargs_chunk = self._recv_and_fill_inputs(
            self.bw_args_recv_info,
            self.bw_kwargs_recv_info,
            chunk
        )

        # `stage_backward` node does not have `args`, only `kwargs`
        outputs_chunk = self.compiled_stage.backward(self.activations_chunks[chunk], self.outputs_chunks[chunk], **composite_kwargs_chunk)

        grad_send_reqs = self._send_output_dict(self.bw_act_send_info, outputs_chunk)
        self.all_bw_send_reqs += grad_send_reqs

    def step(self):
        self.compiled_stage.step(self.outputs_batch)

    def clear_runtime_states(self):
        # Activation send requests of all chunk
        self.all_fw_send_reqs.clear()
        # Grad send requests of all chunk
        self.all_bw_send_reqs.clear()
        # Caching chunk outputs for final output merge or reduction
        self.activations_chunks.clear()
        self.outputs_chunks.clear()
        for _ in range(self.num_chunks):
            self.activations_chunks.append({})
            self.outputs_chunks.append({})
        self.outputs_batch.clear()

    def _merge_output_chunks(self) -> Dict[str, Any]:
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

        self.outputs_batch = {
            **params,
            **buffers,
            **optimstates,
            **grads,
            **rets
        }

        return self.outputs_batch

    def __call__(self, *args, **kwargs) -> None:
        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_inputs(args, kwargs)

        # Forward pass of all chunks
        for chunk in range(self.num_chunks):
            self.forward_one_chunk(chunk)
            logger.debug(f"[{self.group_rank}] Forwarded chunk {chunk}")

        # Backward starts here
        if self.bw_node is not None:
            for bwd_chunk in range(self.num_chunks):
                self.backward_one_chunk(bwd_chunk)
                logger.debug(f"[{self.group_rank}] Backwarded chunk {bwd_chunk}")

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_fw_send_reqs:
            work.wait()

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        if self.bw_node is not None:
            for work in self.all_bw_send_reqs:
                work.wait()

        self._merge_output_chunks()

        # if self.step_node is not None:
        #     self.step()

        logger.debug(f"[{self.group_rank}] All sends finished")

    def all_gather_outputs(self, rank):
        outputs_all_stages = [None for _ in range(self.num_stages)]
        outputs = process_outputs_non_strict(self.compiled_meta, self.outputs_batch)
        dist.gather_object(
            outputs,
            outputs_all_stages if self.group_rank == rank else None,
            dst=rank
        )
        return outputs_all_stages

    def all_gather_state_dict(self, rank):
        state_dicts = [None for _ in range(self.num_stages)]
        state_dict = self.compiled_stage.state_dict()
        dist.gather_object(
            state_dict,
            state_dicts if self.group_rank == rank else None,
            dst=rank
        )
        return state_dicts
    
    def all_gather_optimizer_state_dict(self, rank):
        optimizer_state_dicts = [None for _ in range(self.num_stages)]
        optimizer_state_dict = self.compiled_stage.optimizer_state_dict()
        dist.gather_object(
            optimizer_state_dict,
            optimizer_state_dicts if self.group_rank == rank else None,
            dst=rank
        )
        return optimizer_state_dicts


def print_tensor_dict(chunk, di):
    print(f'Chunk {chunk}')
    for k, v in di.items():
        print(f'{k} size {v.size()} mean {v.float().mean()}')
