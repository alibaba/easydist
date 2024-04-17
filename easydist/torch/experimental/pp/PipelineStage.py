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

from easydist.metashard.metair import VarSPMDStrategy
from easydist.torch.device_mesh import get_device_mesh, get_pp_group, get_pp_rank, get_spmd_rank, spmd_device_mesh
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
    def __init__(
            self, 
            input_name: str,
            source: int,
            example_tensor: FakeTensor
        ):
        self.input_name = input_name
        self.source = source
        self.example_tensor = example_tensor
    
    def get_buffer(self, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

class RecvDynamicBuffer(RecvBase):

    def __init__(
        self,
        input_name: str,
        source: int,
        example_tensor: FakeTensor,
    ):
        super().__init__(input_name, source, example_tensor)
    
    def get_buffer(self, device: torch.device) -> torch.Tensor:
        return _make_tensor_from_meta(self.example_tensor, device)

class RecvStaticBuffer(RecvBase):
    
        def __init__(
            self,
            input_name: str,
            source: int,
            example_tensor: FakeTensor
        ):
            super().__init__(input_name, source, example_tensor)
            self.buffer = None
    
        def get_buffer(self, device: torch.device) -> torch.Tensor:
            if self.buffer is None:
                self.buffer = _make_tensor_from_meta(self.example_tensor, device)
            return self.buffer

class StageKwargPlaceholder:
    def __init__(self, input_name: str):
        self.input_name = input_name


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
        pp_group: dist.ProcessGroup,
        sharded_graph: fx.GraphModule,
        dynamic_buffer=False,
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
        self.pp_group = pp_group

        self.pp_rank = dist.get_rank(pp_group)
        self.num_stages = compiled_meta.nstages
        self.node_to_stage_idx = compiled_meta.node_to_stage_idx  # TODO refactor this mapping?
        self.return_to_all_stages = gather_output
        self.graph = sharded_graph

        if dist.get_world_size(self.pp_group) > self.num_stages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused")

        # communication infra
        self._init_communication(node_metas, dynamic_buffer)

        # runtime states
        self.cur_fw_chunk_id = 0
        self.cur_bw_chunk_id = 0
        self.cur_step_chunk_id = 0
        self.kwargs_chunks = [{} for _ in range(self.num_chunks)]
        self.activations_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_batch = {}  # Activation send requests of all chunk

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

    def _init_communication(self, node_metas, dynamic_buffer):
        """
        Create send/recv infrastructures for activations (during forward) and
        gradients (during backward)
        """
        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        stage_index_to_pp_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(self.pp_group)
        assert pg_world_size == self.num_stages, "Currently only support 1 rank per stage"  # TODO @botbw
        for i in range(self.num_stages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            stage_index_to_pp_rank.setdefault(i, peer_rank)
        self.stage_index_to_group_rank = stage_index_to_pp_rank

        # chunk : Dict of kwarg buffers
        self.fw_kwargs_recv_info = self._create_recv_info(dynamic_buffer, node_metas, self.fw_node)
        self.fw_act_send_info = self._create_send_info(self.fw_node)

        if self.bw_node is not None:
            self.bw_kwargs_recv_info = self._create_recv_info(dynamic_buffer, node_metas, self.bw_node)
            self.bw_grad_send_info = self._create_send_info(self.bw_node)

    def _init_fw_bw_step_nodes(self, local_gm):
        # Find stage forward node in graph
        self.fw_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_fw':
                assert self.fw_node is None, "Multiple forward nodes found"
                self.fw_node = node
        if not self.fw_node:
            raise AssertionError(f"Cannot find {self.name} in graph")

        # Find stage backward node in graph
        self.bw_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_bw':
                assert self.bw_node is None, "Multiple backward nodes found"
                self.bw_node = node

        # Find stage step node in graph
        self.step_node = None
        for node in local_gm.graph.nodes:
            if node.name == f'{self.name}_step':
                assert self.step_node is None, "Multiple step nodes found"
                self.step_node = node

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
        dynamic_buffer,
        node_metas,
        kwarg,
    ):
        info_cls = RecvDynamicBuffer if dynamic_buffer else RecvStaticBuffer
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
                kwargs_recv_info[src_rank].append(info_cls(name, src_rank, example_value))

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

    def _recv_tensor(self, info: RecvBase, recv_reqs: List[dist.Work]):
        logger.debug(f"[{self.pp_rank}] "
                     f"Receiving tensor '{info.input_name}' from Rank {info.source}: "
                     f"{info.example_tensor.size()}")
        # Use async to parallelize recv of tensors
        peer_rank = self.stage_index_to_group_rank[info.source]
        # Create a buffer for receiving the tensor
        buffer = info.get_buffer(self.device)
        work = dist.irecv(
            buffer,
            peer_rank if self.pp_group is None else dist.get_global_rank(self.pp_group, peer_rank),
            group=self.pp_group,
        )
        recv_reqs.append(work)
        return buffer

    def bind_with_recev_tensor_fn(
        self,
        reqs: List[dist.Work],
    ):
        return lambda info: self._recv_tensor(info, reqs)

    def split_input_kwargs(self, kwargs):
        return split_args_kwargs_into_chunks(
            (),
            kwargs,
            self.num_chunks,
            None,
            self.inputs_nodes_chunk_spec,
        )[1]

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
                if isinstance(info, RecvBase):
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
                logger.debug(f"[{self.pp_rank}] "
                             f"Sending tensor to Rank {dst}: {val.size()}")
                peer_rank = self.stage_index_to_group_rank[dst]
                work = dist.isend(
                    val.contiguous(),
                    peer_rank if self.pp_group is None else dist.get_global_rank(
                        self.pp_group, peer_rank),  # TODO
                    group=self.pp_group,
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

    def load_state_dict(self, state_dict):
        self.compiled_stage.load_state_dict(state_dict)

    def optimizer_state_dict(self, world_rank=None):
        if world_rank is None:
            return self.compiled_stage.optimizer_state_dict()
        else:
            return self._gather_optimizer_state_dict(world_rank)

    def state_dict(self, world_rank=None):
        if world_rank is None:
            return self.compiled_stage.state_dict()
        else:
            return self._gather_state_dict(world_rank)

    def load_optimizer_state_dict(self, state_dict):
        self.compiled_stage.load_optimizer_state_dict(state_dict)

    # def _gather_named_params(self, group_rank):
    #     params = [None for _ in range(self.num_stages)]
    #     named_params = self.compiled_stage.named_params()
    #     dist.gather_object(named_params, params if self.group_rank == group_rank else None, dst=group_rank, group=self.group)
    #     return params
    
    def _gather_state_dict(self, world_rank):
        state_dicts = [None for _ in range(self.num_stages)]
        state_dict = self.compiled_stage.state_dict()  # gather spmd0, spmd1
        device_mesh = get_device_mesh()
        spmd0, spmd1 = get_spmd_rank()
        if (spmd0, spmd1) == (device_mesh.mesh == world_rank).nonzero(as_tuple=True)[:2]:
            dist.gather_object(state_dict, state_dicts if device_mesh.get_rank() == world_rank else None, dst=world_rank, group=self.pp_group)
        return state_dicts

    def _gather_optimizer_state_dict(self, world_rank):
        optimizer_state_dicts = [None for _ in range(self.num_stages)]
        optimizer_state_dict = self.compiled_stage.optimizer_state_dict()
        device_mesh = get_device_mesh()
        spmd0, spmd1 = get_spmd_rank()
        if (spmd0, spmd1) == (device_mesh.mesh == world_rank).nonzero(as_tuple=True)[:2]:
            dist.gather_object(optimizer_state_dict, optimizer_state_dicts if device_mesh.get_rank() == world_rank else None, dst=world_rank, group=self.pp_group)
        return optimizer_state_dicts

    def _all_gather_returns(self):
        returns_all_gather = [None for _ in range(self.num_stages)]
        returns_nodes_flatten = {node_name: None for node_name in self.compiled_meta.returns_nodes_flatten}
        returns_batch = {node_name: val for node_name, val in self.outputs_batch.items() if node_name in returns_nodes_flatten}
        dist.all_gather_object(returns_all_gather, returns_batch, group=self.pp_group)
        all_returns = {}
        for returns_stage in returns_all_gather:
            for k, v, in returns_stage.items():
                if v is not None:
                    all_returns[k] = v
        ret = graph_outputs_to_func_outputs(self.compiled_meta, all_returns, strict=False)[-1]
        ret = pytree.tree_map_only(torch.Tensor, lambda x: x.to(self.device), ret)
        return ret

    def __call__(self, *args, **kwargs) -> None:
            # Clean per iteration
        self.clear_runtime_states()

        args_kwargs_vals_flatten, spec_val = pytree.tree_flatten((args, kwargs))
        args_kwargs_nodes_flatten, spec_node = pytree.tree_flatten((self.compiled_meta.args_nodes_unflatten, self.compiled_meta.kwargs_nodes_unflatten))
        assert spec_val == spec_node, "Mismatched args/kwargs"

        input_node_vals = {}
        for node, val in zip(args_kwargs_nodes_flatten, args_kwargs_vals_flatten):
            if isinstance(val, torch.Tensor):
                val = val.to(self.device)
            input_node_vals[node] = val

        # Split inputs into chunks
        self.kwargs_chunks = self.split_input_kwargs(input_node_vals)

        self.schedule()

        logger.debug(f"[{self.pp_rank}] All sends finished")

        if self.return_to_all_stages:
            ret = self._all_gather_returns()
        else:
            ret = graph_outputs_to_func_outputs(self.compiled_meta, self.outputs_batch, strict=False)[-1]
        return ret

    def run_with_graph(self, graph, *args, **kwargs):  # could construct a partial graph
        return self(*args, **kwargs)

def print_tensor_dict(chunk, di):
    print(f'Chunk {chunk}')
    for k, v in di.items():
        print(f'{k} size {v.size()} mean {v.float().mean()}')
