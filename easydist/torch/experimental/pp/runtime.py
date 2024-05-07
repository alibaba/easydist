# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import defaultdict
import logging
import operator
from abc import ABC, abstractmethod
from functools import reduce
from tkinter import Place
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.fx as fx
from torch.profiler import record_function
from torch._subclasses.fake_tensor import FakeTensor
import torch.utils._pytree as pytree

from easydist.torch.device_mesh import get_device_mesh, get_spmd_rank
from easydist.torch.experimental.pp.compile_pipeline import (CompiledMeta, CompiledStage,
                                                             graph_outputs_to_func_outputs)
from easydist.torch.experimental.pp.microbatch import (DEFAULT_CHUNK_DIM, CustomReducer,
                                                       TensorChunkSpec, merge_chunks,
                                                       split_args_kwargs_into_chunks)
from easydist.torch.init_helper import materialize_zero

logger = logging.getLogger(__name__)

def maybe_batch_isend_irecv(p2p_op_list):
    '''
    note: this might lead to hang when the first collective call in the group
    see dist.batch_isend_irecv for more details
    '''
    if len(p2p_op_list) == 0:
        return []
    return dist.batch_isend_irecv(p2p_op_list)

class Placeholder:
    def __init__(self, input_name: str):
        self.input_name = input_name

    def __repr__(self):
        return f"{type(self).__class__}({self.input_name})"

class StageKwargPlaceholder(Placeholder):

    def __init__(self, input_name: str):
        super().__init__(input_name)

class RecevPlaceholder(Placeholder):

    def __init__(self, input_name: str, source: int, example_tensor: FakeTensor, device: torch.device):
        super().__init__(input_name)
        self.source = source
        self.buffer = materialize_zero(example_tensor, device)

class PipelineStage:

    def __init__(self,
                 schedule_cls: Type['Schedule'],
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
                 return_to_all_stages: bool,
                 accumulate_grads_inplace: bool):
        # meta info
        self.name = f'stage_{stage_idx}'
        self._init_fw_bw_step_nodes(local_gm)
        assert issubclass(schedule_cls, Schedule), "schedule_cls must be the Schedule class"
        self.stage_idx = stage_idx
        self.compiled_meta = compiled_meta
        self.compiled_stage = compiled_stage
        self.num_chunks = num_chunks
        self._init_inputs_nodes_spec(compiled_meta, args_chunk_spec, kwargs_chunk_spec)
        self._init_returns_nodes_spec(compiled_meta, returns_chunk_spec)
        self.device = device
        self.pp_group = pp_group

        self.pp_rank = dist.get_rank(pp_group)
        self.num_stages = compiled_meta.nstages
        self.node_to_stage_idx = compiled_meta.node_to_stage_idx  # TODO refactor this mapping?
        self.graph = sharded_graph
        self.return_to_all_stages = return_to_all_stages
        self.accumulate_grads_inplace = accumulate_grads_inplace

        if dist.get_world_size(self.pp_group) > self.num_stages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused")

        # communication infra
        self._init_communication(node_metas)
        # self._init_batch_isend_irecv()

        # runtime states
        self._init_runtime_states()

        # post init here (schedule_cls requires PipelineStage initialized)
        self.schedule = schedule_cls(self)


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

    def _init_inputs_nodes_spec(self, compiled_meta: CompiledMeta, args_chunk_spec,
                               kwargs_chunk_spec):
        node_val_chunk_spec = {}
        args_nodes_flatten, _ = pytree.tree_flatten(compiled_meta.args_nodes_unflatten)
        args_chunk_spec = args_chunk_spec or [None] * len(
            args_nodes_flatten)  # input could be non tensor, use None instead of TensorChunkSpec
        args_chunk_spec_flatten, _ = pytree.tree_flatten(args_chunk_spec)
        assert len(args_chunk_spec_flatten) == len(args_nodes_flatten)
        for node_name, arg_chunk_spec in zip(compiled_meta.args_nodes_unflatten,
                                             args_chunk_spec_flatten):
            node_val_chunk_spec[node_name] = arg_chunk_spec

        kwargs_nodes_flatten, spec1 = pytree.tree_flatten(compiled_meta.kwargs_nodes_unflatten)
        kwargs_chunk_spec = kwargs_chunk_spec or {
            node_name: TensorChunkSpec(DEFAULT_CHUNK_DIM)
            for node_name in kwargs_nodes_flatten
        }
        kwargs_chunk_spec_flatten, spec2 = pytree.tree_flatten(kwargs_chunk_spec)
        assert spec1 == spec2
        for node_name, kwarg_chunk_spec in zip(kwargs_nodes_flatten, kwargs_chunk_spec_flatten):
            node_val_chunk_spec[node_name] = kwarg_chunk_spec

        self.inputs_nodes_chunk_spec = node_val_chunk_spec

    def _init_returns_nodes_spec(self, compiled_meta, returns_chunk_spec):
        returns_nodes_chunk_spec = {}
        returns_chunk_spec = returns_chunk_spec or [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(
            compiled_meta.returns_nodes_flatten)
        returns_chunk_spec_flatten, _ = pytree.tree_flatten(returns_chunk_spec)
        assert len(returns_chunk_spec_flatten) == len(compiled_meta.returns_nodes_flatten)
        for name, spec in zip(compiled_meta.returns_nodes_flatten, returns_chunk_spec_flatten):
            returns_nodes_chunk_spec[name] = spec

        self.returns_nodes_chunk_spec = returns_nodes_chunk_spec

    def _init_communication(self, node_metas):
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
        self.fw_kwargs_recv_info = self._create_recv_info(node_metas, self.fw_node, is_forward=True)
        self.fw_act_send_info = self._create_send_info(self.fw_node, is_forward=True)

        if self.bw_node is not None:
            self.bw_kwargs_recv_info = self._create_recv_info(node_metas, self.bw_node, is_forward=False)
            self.bw_grad_send_info = self._create_send_info(self.bw_node, is_forward=False)

    def _init_batch_isend_irecv(self):
        '''
        See batch_isend_irecv in torch.distributed for more details:
            .. note:: Note that when this API is used with the NCCL PG backend, users must set
                the current GPU device with `torch.cuda.set_device`, otherwise it will
                lead to unexpected hang issues.

            In addition, if this API is the first collective call in the ``group``
            passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
            this API call; otherwise, the behavior is undefined. If this API call is
            not the first collective call in the ``group``, batched P2P operations
            involving only a subset of ranks of the ``group`` are allowed.
        '''
        torch.cuda.set_device(self.device)
        succeesor = self.stage_index_to_group_rank[(self.stage_idx + 1) % self.num_stages]
        succeesor_global = succeesor if self.pp_group is None else dist.get_global_rank(self.pp_group, succeesor)
        succeesor_recv = torch.zeros(1, device=self.device)
        ancessor = self.stage_index_to_group_rank[(self.stage_idx - 1) % self.num_stages]
        ancessor_global = ancessor if self.pp_group is None else dist.get_global_rank(self.pp_group, ancessor)
        ancessor_recv = torch.zeros(1, device=self.device)
        send_tensor = torch.zeros(1, device=self.device) + self.stage_idx
        ops = [
            dist.P2POp(dist.isend, send_tensor, succeesor_global, self.pp_group),
            dist.P2POp(dist.isend, send_tensor, ancessor_global, self.pp_group),
            dist.P2POp(dist.irecv, succeesor_recv, succeesor_global, self.pp_group),
            dist.P2POp(dist.irecv, ancessor_recv, ancessor_global, self.pp_group)
        ]
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
        assert succeesor_recv.item() == (self.stage_idx + 1) % self.num_stages
        assert ancessor_recv.item() == (self.stage_idx - 1) % self.num_stages

    def _init_runtime_states(self):
        self.cur_fw_send_chunk = None
        self.cur_bw_send_chunk = None
        self.cur_fw_chunk_id = 0
        self.cur_bw_chunk_id = 0
        self.cur_step_chunk_id = 0
        self.kwargs_chunks = [{} for _ in range(self.num_chunks)]
        self.activations_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_chunks: List[Dict[str, Any]] = [{} for _ in range(self.num_chunks)]
        self.outputs_batch = {}  # Activation send requests of all chunk
        if self.accumulate_grads_inplace:
            self.grads = {}

    def _create_send_info(self, node: fx.Node, is_forward: bool) -> Dict[int, List[str]]:  # TODO @botbw: simplify
        to_sort = []
        for user in node.users:
            assert user.target is operator.getitem, "Output must be a dict"
            out_str = user.args[-1]
            assert isinstance(out_str, str)
            for gi_user in user.users:
                dst_rank = self.node_to_stage_idx[gi_user.name]
                to_sort.append((dst_rank, out_str))
        if is_forward:
            to_sort.sort(key=lambda x: (x[0], x[1]))  # send lower to rank first and in alphabetical order
        else:
            to_sort.sort(key=lambda x: (-x[0], x[1])) # send higher to rank first and in alphabetical order

        send_info_by_stage = defaultdict(list)
        for dst_rank, out_str in to_sort:
            send_info_by_stage[dst_rank].append(out_str)

        return send_info_by_stage

    def _create_send_ops(self, send_info: Dict[int, List[str]], output_dict: Dict[str, torch.Tensor]):
        # Send requests of a chunk
        send_ops_by_dst = defaultdict(list)
        for dst, nodes in send_info.items():
            peer_rank = self.stage_index_to_group_rank[dst]
            peer_global_rank = peer_rank if self.pp_group is None else dist.get_global_rank(self.pp_group, peer_rank)
            for node in nodes:
                val = output_dict[node]
                send_ops_by_dst[dst].append(dist.P2POp(dist.isend, val, peer_global_rank, self.pp_group))

        return [send_ops_by_dst[dst] for dst in sorted(send_info.keys())]

    def _create_recv_info(
        self,
        node_metas: Dict[str, Dict],
        node: fx.Node,
        is_forward: bool,
    ) -> Dict[int, List[Placeholder]]:
        to_sort = []
        for _, node in node.kwargs.items():
            if node.op == "placeholder":
                to_sort.append((-1, node.name))
                continue
            example_value = node_metas[node.name]["val"]
            src_rank = self.node_to_stage_idx[node.name]
            global_src_rank = src_rank if self.pp_group is None else dist.get_global_rank(self.pp_group, src_rank)
            to_sort.append((global_src_rank, node.name, example_value))

        if is_forward:
            # receive lower rank first and in alphabetical order
            to_sort.sort(key=lambda x: (x[0], x[1]))
        else:
            # receive higer rank first and in alphabetical order
            to_sort.sort(key=lambda x: (-x[0], x[1]))
        kwargs_recv_info = defaultdict(list)
        for x in to_sort:
            if x[0] == -1:
                assert is_forward
                kwargs_recv_info[0].append(StageKwargPlaceholder(
                    x[1]))  # args recev with rank 0 (lowest rank)
            else:
                global_src_rank, name, example_value = x
                kwargs_recv_info[global_src_rank].append(RecevPlaceholder(name, global_src_rank, example_value, self.device))

        return kwargs_recv_info

    def _create_recv_ops(self, recv_info: Dict[int, List[Placeholder]]):
        recv_ops_by_src = []
        for src, ph_list in recv_info.items():
            rec_ops = []
            for ph in ph_list:
                if isinstance(ph, RecevPlaceholder):
                    rec_ops.append(dist.P2POp(dist.irecv, ph.buffer, src, self.pp_group))
            recv_ops_by_src.append(rec_ops)
        return recv_ops_by_src

    def collect_kwargs(
        self,
        recv_info: Dict[int, List[Placeholder]],
        chunk: int,
    ):
        chunk_kwargs = self.kwargs_chunks[chunk]

        composite_kwargs = {}
        for rank, ph_list in recv_info.items():
            for ph in ph_list:
                if isinstance(ph, RecevPlaceholder):
                    composite_kwargs[ph.input_name] = ph.buffer
                else:
                    composite_kwargs[ph.input_name] = chunk_kwargs[ph.input_name]

        return composite_kwargs

    def forward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        # Receive activations
        recv_ops_by_src = self._create_recv_ops(self.fw_kwargs_recv_info)
        recv_reqs = []
        for ops in recv_ops_by_src:
            recv_reqs += maybe_batch_isend_irecv(ops)
        if wait:
            for req in recv_reqs:
                req.wait()
        return recv_reqs


    def forward_compute_one_chunk(self):
        # Collect activations and kwargs
        composite_kwargs_chunk = self.collect_kwargs(self.fw_kwargs_recv_info,
                                                            self.cur_fw_chunk_id)

        # Compute forward
        self.cur_fw_send_chunk = self.compiled_stage.forward(self.activations_chunks[self.cur_fw_chunk_id],
                                                    self.outputs_chunks[self.cur_fw_chunk_id],
                                                    **composite_kwargs_chunk)
        # Update runtime states
        self.cur_fw_chunk_id += 1

    def forward_send_one_chunk(self) -> List[dist.Work]:
        # Send activations
        send_ops_by_dst = self._create_send_ops(self.fw_act_send_info, self.cur_fw_send_chunk)
        reqs = []
        for ops in send_ops_by_dst:
            reqs += maybe_batch_isend_irecv(ops)
        return reqs

    def backward_recv_one_chunk(self, wait=True) -> List[dist.Work]:
        # Receive grads
        recv_ops_by_src = self._create_recv_ops(self.bw_kwargs_recv_info)
        recv_reqs = []
        for ops in recv_ops_by_src:
            recv_reqs += maybe_batch_isend_irecv(ops)
        if wait:
            for req in recv_reqs:
                req.wait()
        return recv_reqs

    def backward_compute_one_chunk(self):
        # Collect grads and kwargs
        composite_kwargs_chunk = self.collect_kwargs(self.bw_kwargs_recv_info,
                                                            self.cur_bw_chunk_id)

        # Compute backward
        self.cur_bw_send_chunk = self.compiled_stage.backward(self.activations_chunks[self.cur_bw_chunk_id],
                                                     self.outputs_chunks[self.cur_bw_chunk_id],
                                                     **composite_kwargs_chunk)
        if self.accumulate_grads_inplace:
            grads_nodes = dict.fromkeys(self.compiled_meta.nones_or_grads_nodes_unflatten.values())
            to_pop = []
            for k, v in self.outputs_chunks[self.cur_bw_chunk_id].items():
                if k in grads_nodes:
                    to_pop.append(k)
                    if k in self.grads:
                        self.grads[k] += v
                    else:
                        self.grads[k] = v
            for k in to_pop:
                self.outputs_chunks[self.cur_bw_chunk_id].pop(k)
        # Update runtime states
        self.cur_bw_chunk_id += 1

    def backward_send_one_chunk(self) -> List[dist.Work]:
        # Send grads
        send_ops_by_dst = self._create_send_ops(self.bw_grad_send_info, self.cur_bw_send_chunk)
        reqs = []
        for ops in send_ops_by_dst:
            reqs += maybe_batch_isend_irecv(ops)
        return reqs

    def step(self):
        self.compiled_stage.step(self.outputs_batch)

    def clear_runtime_states(self):
        self.cur_fw_send_chunk = None
        self.cur_bw_send_chunk = None
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
        if self.accumulate_grads_inplace:
            self.grads.clear()

    def split_input_kwargs(self, kwargs):
        return split_args_kwargs_into_chunks(
            (),
            kwargs,
            self.num_chunks,
            None,
            self.inputs_nodes_chunk_spec,
        )[1]

    @torch.no_grad
    def merge_output_chunks(self) -> Dict[str, Any]:
        params_nodes = dict.fromkeys(self.compiled_meta.output_params_nodes_unflatten.values())
        buffers_nodes = dict.fromkeys(self.compiled_meta.output_buffers_nodes_unflatten.values())
        optimstates_nodes = dict.fromkeys(self.compiled_meta.output_optimstates_nodes_flatten)
        nones_or_grads_nodes = dict.fromkeys(self.compiled_meta.nones_or_grads_nodes_unflatten.values())
        returns_names_flatten = dict.fromkeys(self.compiled_meta.returns_nodes_flatten)

        params, buffers, optimstates, grads, rets = {}, {}, {}, [], []
        for chunk in self.outputs_chunks:
            grads_chunk, rets_chunk = {}, {node_name: None for node_name in returns_names_flatten}
            for node_name, tensor in chunk.items():
                if node_name in params_nodes:
                    params[node_name] = tensor
                elif node_name in buffers_nodes:
                    buffers[node_name] = tensor
                elif node_name in optimstates_nodes:
                    optimstates[node_name] = tensor
                elif node_name in nones_or_grads_nodes:
                    if not self.accumulate_grads_inplace:
                        grads_chunk[node_name] = tensor
                elif node_name in returns_names_flatten:
                    rets_chunk[node_name] = tensor
                else:
                    raise RuntimeError(f"Unknown output {node_name}")
            chunk.clear()
            grads.append(grads_chunk)
            rets.append(rets_chunk)

        rets = merge_chunks(rets, self.returns_nodes_chunk_spec)
        if self.accumulate_grads_inplace:
            grads = self.grads
        else:
            grads = reduce(lambda a, b: {k: torch.add(a[k], b[k]) for k in a}, grads)


        self.outputs_batch.update({**params, **buffers, **optimstates, **grads, **rets})

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

    def _gather_state_dict(self, world_rank):
        state_dicts = [None for _ in range(self.num_stages)]
        state_dict = self.compiled_stage.state_dict()  # gather spmd0, spmd1
        device_mesh = get_device_mesh()
        spmd0, spmd1 = get_spmd_rank()
        if (spmd0, spmd1) == (device_mesh.mesh == world_rank).nonzero(as_tuple=True)[:2]:
            dist.gather_object(state_dict,
                               state_dicts if device_mesh.get_rank() == world_rank else None,
                               dst=world_rank,
                               group=self.pp_group)
        return state_dicts

    def _gather_optimizer_state_dict(self, world_rank):
        optimizer_state_dicts = [None for _ in range(self.num_stages)]
        optimizer_state_dict = self.compiled_stage.optimizer_state_dict()
        device_mesh = get_device_mesh()
        spmd0, spmd1 = get_spmd_rank()
        if (spmd0, spmd1) == (device_mesh.mesh == world_rank).nonzero(as_tuple=True)[:2]:
            dist.gather_object(
                optimizer_state_dict,
                optimizer_state_dicts if device_mesh.get_rank() == world_rank else None,
                dst=world_rank,
                group=self.pp_group)
        return optimizer_state_dicts

    def _all_gather_returns(self):
        returns_all_gather = [None for _ in range(self.num_stages)]
        returns_nodes_flatten = {
            node_name: None
            for node_name in self.compiled_meta.returns_nodes_flatten
        }
        returns_batch = {
            node_name: val
            for node_name, val in self.outputs_batch.items() if node_name in returns_nodes_flatten
        }
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
        args_kwargs_nodes_flatten, spec_node = pytree.tree_flatten(
            (self.compiled_meta.args_nodes_unflatten, self.compiled_meta.kwargs_nodes_unflatten))
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
            ret = graph_outputs_to_func_outputs(self.compiled_meta,
                                                self.outputs_batch,
                                            strict=False)[-1]
        return ret

    def run_with_graph(self, graph, *args, **kwargs):  # TODO @botbw: could construct a partial graph here
        return self(*args, **kwargs)


def print_tensor_dict(chunk, di):
    print(f'Chunk {chunk}')
    for k, v in di.items():
        print(f'{k} size {v.size()} mean {v.float().mean()}')


# TODO @botbw: refactor to mixin
class Schedule(ABC):

    def __init__(self, pipeline_stage: PipelineStage):
        assert isinstance(pipeline_stage, PipelineStage)
        self.pipeline_stage = pipeline_stage

    @abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self.pipeline_stage, name)


class ScheduleGPipe(Schedule):

    def __call__(self) -> None:

        all_send_reqs: List[dist.Work] = []

        # Forward all chunks
        for fw_chunk in range(self.num_chunks):
            self.forward_recv_one_chunk()
            self.forward_compute_one_chunk()
            all_send_reqs += self.forward_send_one_chunk()

        # Backward all chunks
        if self.bw_node is not None:
            for bwd_chunk in range(self.num_chunks):
                self.backward_recv_one_chunk()
                self.backward_compute_one_chunk()
                all_send_reqs += self.backward_send_one_chunk()

        for work in all_send_reqs:
            work.wait()

        self.merge_output_chunks()

        if self.step_node is not None:
            self.step()


class ScheduleDAPPLE(Schedule):

    def __init__(self, pipeline_stage: PipelineStage):
        super().__init__(pipeline_stage)
        assert pipeline_stage.bw_node is not None, f"{type(self).__name__} requires backward node"
        num_warmup = self.pipeline_stage.num_stages - self.pipeline_stage.stage_idx
        self.num_warmup = min(num_warmup, self.pipeline_stage.num_chunks)

    def __call__(self) -> None:
        all_send_reqs: List[dist.Work] = []

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(self.num_warmup):
            self.forward_recv_one_chunk()
            self.forward_compute_one_chunk()
            all_send_reqs += self.forward_send_one_chunk()

        # 1F1B phase
        for fwd_chunk in range(self.num_warmup, self.num_chunks):
            # recv backward first
            self.backward_recv_one_chunk()
            # IMPORTANT: recv forward after recv backward and before send backward
            reqs = self.forward_recv_one_chunk(wait=False)
            self.backward_compute_one_chunk()
            all_send_reqs += self.backward_send_one_chunk()
            for req in reqs:
                req.wait()
            self.forward_compute_one_chunk()
            all_send_reqs += self.forward_send_one_chunk()

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.num_chunks - self.num_warmup, self.num_chunks):
            self.backward_recv_one_chunk()
            self.backward_compute_one_chunk()
            all_send_reqs += self.backward_send_one_chunk()

        for work in all_send_reqs:
            work.wait()

        self.merge_output_chunks()

        if self.step_node is not None:
            self.step()
