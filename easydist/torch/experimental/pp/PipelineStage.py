# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from platform import node
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_aggregate, map_arg
from torch.nn.parallel import DistributedDataParallel

from easydist.torch.experimental.pp.microbatch import merge_chunks, split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.utils import flatten_args, modify_graph_op_device, QualnameMapMixin, map_debug_info

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


class PipelineStage: #(torch.nn.Module, QualnameMapMixin):
    def __init__(
        self,
        node_to_stage,
        node_metas,
        split_gm,
        compiled_stages,
        num_stages,
        num_chunks,
        args_chunk_spec,
        kwargs_chunk_spec,
        outputs_chunk_spec,
        stage_index: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        self.node_to_stage = node_to_stage
        self.node_metas = node_metas
        compiled_stage = compiled_stages[stage_index]
        self.stage_index = stage_index
        self.nstages = num_stages
        self.chunks = num_chunks
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.outputs_chunk_spec = outputs_chunk_spec
        self.device = device
        self.group = group

        if dist.get_world_size(self.group) > self.nstages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused"
            )

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(group)

        # Activation send requests of all chunk
        self.all_fw_send_reqs: List[dist.Work] = []
        # Grad send requests of all chunk
        self.all_bw_send_reqs: List[dist.Work] = []
        # Caching chunk outputs for final output merge or reduction
        self.saved_for_bw_chunks: List[Any] = [{} for _ in range(self.chunks)]
        self.outputs_chunks: List[Any] = [{} for _ in range(self.chunks)]

        self.split_gm = split_gm
        self.stage = compiled_stage
        self.name, self.submod = f'stage_{stage_index}', compiled_stage.fw_gm
        logger.info(
            f"[{self.group_rank}] "
            f"Creating PipelineStage:\n"
            f"{self.submod}"
        )

        # Find my forward node in graph
        self.fw_node = None
        for node in self.split_gm.graph.nodes:
            if node.name == f'{self.name}_fw':
                assert self.fw_node is None, "Multiple forward nodes found"
                self.fw_node = node
        if not self.fw_node:
            raise AssertionError(f"Cannot find {self.name} in graph")
        
        # Find my backward node in graph
        self.bw_node = None
        for node in self.split_gm.graph.nodes:
            if node.name == f'{self.name}_bw':
                assert self.bw_node is None, "Multiple backward nodes found"
                self.bw_node = node

        # Find my step node in graph
        self.step_node = None
        for node in self.split_gm.graph.nodes:
            if node.name == f'{self.name}_step':
                assert self.step_node is None, "Multiple step nodes found"
                self.step_node = node

        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        self.stage_index_to_group_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(group)
        for i in range(self.nstages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            self.stage_index_to_group_rank.setdefault(i, peer_rank)

        # Prepar
        # Prepare send/recv infrastructure
        self._prepare_send_recv_infra()
        # Cast submodule to device
        self._move_inject_states_to_device()
        # Move ops argument to device
        self._move_ops_to_device()

    def _move_inject_states_to_device(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        assert not any(
            isinstance(p, FakeTensor) or p.is_meta
            for p in self.submod.parameters()
        )
        for _, tensor in self.stage.fw_gm.injected_states.items():
            tensor.to(self.device)

    def _move_ops_to_device(self):
        # Today PT2 tracer does not treat `x.device` as a symbolic device;
        # instead, the device of tracing time got burned into the generated
        # code.  Here we provide a workaround for users to manually modify the
        # "device" kwarg of operations. Such operation may include:
        # `torch.ones`, `torch.zeros`, `torch.rand`, etc.
        modify_graph_op_device(self.stage.fw_gm, self.device)
        modify_graph_op_device(self.stage.bw_gm, self.device)
        if self.stage.has_step():
            modify_graph_op_device(self.stage.step_gm, self.device)

    def is_first(self):
        return self.stage_index == 0

    def is_last(self):
        return self.stage_index == self.nstages - 1

    def _prepare_send_recv_infra(self):
        """
        Create send/recv infrastructures for activations (during forward) and
        gradients (during backward)
        """
        # chunk : Tuple of arg buffers
        self.fw_args_recv_info: Dict[int, Tuple] = {}
        # chunk : Dict of kwarg buffers
        self.fw_kwargs_recv_info: Dict[int, Dict] = {}
        for chunk in range(self.chunks):
            (
                self.fw_args_recv_info[chunk],
                self.fw_kwargs_recv_info[chunk],
            ) = self._create_act_recv_buffers(self.fw_node)

        self.bw_args_recv_info: Dict[int, Tuple] = {}
        self.bw_kwargs_recv_info: Dict[int, Dict] = {}
        for chunk in range(self.chunks):
            (
                self.bw_args_recv_info[chunk],
                self.bw_kwargs_recv_info[chunk],
            ) = self._create_act_recv_buffers(self.bw_node)

        # Send info during forward for each activation
        self.fw_act_send_info = self._create_act_send_info(self.fw_node)
        self.bw_act_send_info = self._create_act_send_info(self.bw_node)


    def get_stage_index_of_node_name(
        self,
        node_name: str,
    ):
        return self.node_to_stage[node_name]

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

        # `args` is a Tuple, hence we will have:
        # Tuple[RecvInfo]
        args_recv_info = map_arg(node.args, create_recv_tensor)

        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        kwargs_recv_info = map_arg(node.kwargs, create_recv_tensor)

        logger.info(
            f"[{self.group_rank}] "
            f"Activation recv / args info: {args_recv_info}"
        )
        return args_recv_info, kwargs_recv_info

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        return self.get_stage_index_of_node_name(user.name)


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
        return act_send_info

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
        self.args_split = None
        self.kwargs_split = None
        if args or kwargs:
            self.args_split, self.kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
                None, #self.pipe.args_chunk_spec,
                None, #self.pipe.kwargs_chunk_spec,
            )

    def _recv_and_fill_inputs(
        self,
        args_split,
        kwargs_split,
        args_recv_info,
        kwargs_recv_info,
        chunk: int,
    ):
        # Receive requests of a chunk
        recv_reqs: List[dist.Work] = []

        act_recv = self.recv_tensor_fn(recv_reqs)

        chunk_args_list: List = []
        if args_split:
            chunk_args = args_split[chunk]
            chunk_args_list = list(chunk_args)

        def recv_args(info):
            if isinstance(info, RecvInfo):
                # This is an activation to receive
                return act_recv(info)
            else:
                return chunk_args_list.pop(0)  # type: ignore[has-type]

        composite_args = map_aggregate(
            args_recv_info[chunk],
            recv_args,
        )

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

        return composite_args, composite_kwargs

    def _send_output_dict(
        self,
        send_info,
        output_dict,
    ) -> List[dist.Work]:
        # Send requests of a chunk
        send_reqs: List[dist.Work] = []

        for name, out in output_dict.items():
            dst_stages = send_info[name]
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
        _, composite_kwargs = self._recv_and_fill_inputs(
            self.args_split,
            self.kwargs_split,
            self.fw_args_recv_info,
            self.fw_kwargs_recv_info,
            chunk
        )

        # Compute forward
        output = self.stage.forward(self.saved_for_bw_chunks[chunk], self.outputs_chunks[chunk], **composite_kwargs)

        assert isinstance(output, dict), "Only dict output is supported"
        logger.debug(map_debug_info(output))

        # Send activations
        send_reqs = self._send_output_dict(self.fw_act_send_info, output)
        self.all_fw_send_reqs += send_reqs

    def backward_one_chunk(
        self,
        bwd_chunk: int,
    ):
        if not self.bw_node:
            return {}

        _, composite_kwargs = self._recv_and_fill_inputs(
            None,
            None,
            self.bw_args_recv_info,
            self.bw_kwargs_recv_info,
            bwd_chunk
        )

        # `stage_backward` node does not have `args`, only `kwargs`
        grads_input = self.stage.backward(self.saved_for_bw_chunks[bwd_chunk], self.outputs_chunks[bwd_chunk], **composite_kwargs)

        grad_send_reqs = self._send_output_dict(self.bw_act_send_info, grads_input)
        self.all_bw_send_reqs += grad_send_reqs

        if self.step_node is not None:
            self.stage.step(self.saved_for_bw_chunks[bwd_chunk], self.outputs_chunks[bwd_chunk])

    def clear_runtime_states(self):
        # Activation send requests of all chunk
        self.all_fw_send_reqs.clear()
        # Grad send requests of all chunk
        self.all_bw_send_reqs.clear()
        # Caching chunk outputs for final output merge or reduction
        self.saved_for_bw_chunks.clear()
        self.outputs_chunks.clear()
        for _ in range(self.chunks):
            self.saved_for_bw_chunks.append({})
            self.outputs_chunks.append({})

    def merge_output_chunks(self) -> Dict[str, Any]:
        return merge_chunks(
            self.outputs_chunks,
            self.outputs_chunk_spec,
        )

    @torch.no_grad
    def __call__(self, *args, **kwargs):
        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_inputs(args, kwargs)

        # Forward pass of all chunks
        for chunk in range(self.chunks):
            self.forward_one_chunk(chunk)
            logger.debug(f"[{self.group_rank}] Forwarded chunk {chunk}")

        # Backward starts here
        for bwd_chunk in range(self.chunks):
            self.backward_one_chunk(bwd_chunk)
            logger.debug(f"[{self.group_rank}] Backwarded chunk {bwd_chunk}")

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_fw_send_reqs:
            work.wait()

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_bw_send_reqs:
            work.wait()

        logger.debug(f"[{self.group_rank}] All sends finished")

    def all_gather_outputs(self, rank):
        outputs = [None for _ in range(self.nstages)]
        dist.gather_object(
            self.merge_output_chunks(),
            outputs if self.group_rank == rank else None,
            dst=rank
        )
        return outputs

# class PipelineStage1F1B(PipelineStage):
#     def __init__(
#         self,
#         pipe: Pipe,
#         rank: int,
#         device: torch.device,
#         group: dist.ProcessGroup = None,
#     ):
#         super().__init__(
#             pipe,
#             rank,
#             device,
#             group=group,
#         )

#     def forward(self, *args, **kwargs):
#         # Clean per iteration
#         self.clear_runtime_states()

#         # Split inputs into chunks
#         self.split_inputs(args, kwargs)

#         warmup_chunks = cooldown_chunks = self.nstages

#         # Warm-up phase: forward number of chunks equal to pipeline depth.
#         for chunk in range(warmup_chunks):
#             self.forward_one_chunk(chunk)

#         # 1F1B phase
#         for bwd_chunk in range(0, self.chunks - cooldown_chunks):
#             # Schedule backward for one warmed up chunk
#             self.backward_one_chunk(bwd_chunk)

#             # Schedule forward for one new chunk
#             fwd_chunk = bwd_chunk + warmup_chunks
#             self.forward_one_chunk(fwd_chunk)

#         # Cool-down phase: backward for the rest of the chunks
#         for bwd_chunk in range(self.chunks - cooldown_chunks, self.chunks):
#             self.backward_one_chunk(bwd_chunk)

#         # Wait for all sends to finish
#         # TODO: okay to delay the sync till completion of all chunks?
#         for work in self.all_act_send_reqs:
#             work.wait()

#         for work in self.all_grad_send_reqs:
#             work.wait()

#         # Last rank return merged results per original format
#         if self.is_last():
#             return self.merge_output_chunks()
#         else:
#             return None
