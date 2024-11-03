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

import logging

import torch
import torch.utils._pytree as pytree
from torch.fx.node import _get_qualified_name
if torch.__version__ >= (2, 4):
    from torch.distributed.rpc.api import _wait_all
else:
    from torch.distributed._functional_collectives_impl import _wait_all

import easydist.config as mdconfig
from easydist.torch.utils import EDNodeType
from easydist.torch.init_helper import materialize_random
from easydist.torch.graph_profile_db import PerfDB

from .sharding import view_op_map

logger = logging.getLogger(__name__)


def _benchmark_op(node, materialized_inputs):
    # profile and get runtime in ms
    start_evt_, end_evt_ = [], []
    for _ in range(0, mdconfig.profile_trials):
        start_evt_.append(torch.cuda.Event(enable_timing=True))
        end_evt_.append(torch.cuda.Event(enable_timing=True))

    for trial_idx_ in range(0, mdconfig.profile_trials + mdconfig.profile_warmup_trials):
        evt_idx = trial_idx_ - mdconfig.profile_warmup_trials

        if evt_idx >= 0:
            start_evt_[evt_idx].record()

        _ = node.target(*materialized_inputs, **node.kwargs)
        if node.ed_info.node_type == EDNodeType.COMMUNICATION:
            _wait_all()

        if evt_idx >= 0:
            end_evt_[evt_idx].record()

    torch.cuda.synchronize()
    ops_elapsed_time_ = 0
    for evt_idx in range(0, mdconfig.profile_trials):
        # time elapsed in **milliseconds**
        ops_elapsed_time_ += start_evt_[evt_idx].elapsed_time(end_evt_[evt_idx])
    ops_elapsed_time_ = ops_elapsed_time_ / mdconfig.profile_trials

    return ops_elapsed_time_


def get_tiled_inputs(materialized_inputs, sharding_ann, tile_dim, num_tiles):
    flatten_inputs, args_specs = pytree.tree_flatten(materialized_inputs)
    tiled_inputs = []
    inputs_tensor_index = 0
    for input_ in flatten_inputs:
        if isinstance(input_, torch.Tensor):
            input_sharding_ann = sharding_ann[inputs_tensor_index]
            input_sharding_ann = [sharddim.shard_dim_id for sharddim in input_sharding_ann]
            if tile_dim in input_sharding_ann:
                input_tile_dim = input_sharding_ann.index(tile_dim)
                tiled_input = torch.chunk(input_, chunks=num_tiles, dim=input_tile_dim)[0]
                tiled_inputs.append(tiled_input)
            else:
                tiled_inputs.append(input_)
            inputs_tensor_index += 1
        else:
            tiled_inputs.append(input_)

    return pytree.tree_unflatten(tiled_inputs, args_specs)


def runtime_prof(fx_module: torch.fx.GraphModule, tiling_prof=False) -> torch.fx.GraphModule:
    logger.info(f"runtime profiling pass, tiling_prof={tiling_prof}")

    # make sure all rank have same runtime profiling result
    runtime_prof_result = dict()
    tiled_runtime_prof_result = dict()

    perf_db = PerfDB()

    for node in fx_module.graph.nodes:
        if hasattr(node, "ed_info"):
            if node.ed_info.node_type in [EDNodeType.COMMUNICATION, EDNodeType.COMPUTATION]:
                # get the input signature of this node and checl if it is in the database
                inputs_signature = pytree.tree_map_only(torch.fx.Node, lambda n: n.meta['val'],
                                                        node.args)

                qualified_name = _get_qualified_name(node.target)
                db_record = perf_db.get_op_perf(qualified_name, inputs_signature.__str__())
                if db_record is not None:
                    runtime_prof_result[node.name] = db_record
                    continue

                # materialize the input of this node
                materialized_inputs = pytree.tree_map_only(torch.Tensor, materialize_random,
                                                           inputs_signature)

                ops_elapsed_time_ = _benchmark_op(node, materialized_inputs)

                # profiling the tiling performance
                if tiling_prof is True and node.ed_info.spmd_annotation is not None:
                    # for all tile dim
                    num_tiles = 2
                    tiled_runtime_prof_result[node.name] = {}
                    for tile_dim in node.ed_info.spmd_annotation["combination_ann"]:
                        if node.target in view_op_map:
                            tiled_runtime_prof_result[node.name][tile_dim] = ops_elapsed_time_
                            continue
                        tiled_inputs = get_tiled_inputs(
                            materialized_inputs, node.ed_info.spmd_annotation['sharding_ann'],
                            tile_dim, num_tiles)
                        tiled_ops_elapsed_time_ = _benchmark_op(node, tiled_inputs) * num_tiles
                        tiled_runtime_prof_result[node.name][tile_dim] = tiled_ops_elapsed_time_

                runtime_prof_result[node.name] = ops_elapsed_time_
                perf_db.record_op_perf(qualified_name, inputs_signature.__str__(),
                                       ops_elapsed_time_)

    min_runtime_ms = float('inf')
    for node in fx_module.graph.nodes:
        if hasattr(node, "ed_info") and node.name in runtime_prof_result:
            node.ed_info.runtime_ms = runtime_prof_result[node.name]
            if node.name in tiled_runtime_prof_result:
                node.ed_info.tiled_runtime_ms = tiled_runtime_prof_result[node.name]
            min_runtime_ms = min(node.ed_info.runtime_ms, min_runtime_ms)
    for node in fx_module.graph.nodes:
        if hasattr(node, "ed_info"):
            if node.name in runtime_prof_result:
                # Avoid overflow
                node.ed_info.normalized_int_runtime_ms = min(
                    max(int(node.ed_info.runtime_ms / min_runtime_ms / 100), 1), int(32768 / 10))
                #node.ed_info.normalized_int_runtime_ms = min(
                #    int(node.ed_info.runtime_ms / min_runtime_ms), 32768
                #)
            else:
                node.ed_info.normalized_int_runtime_ms = 1

    if mdconfig.dump_prof_db and torch.distributed.get_rank() == 0:
        perf_db.persistent()

    # broadcast ed_info for all nodes, note that ori_meta cannot pickle
    # (TODO): move to other place
    md_info_all = {}
    ori_meta_all = {}
    for node in fx_module.graph.nodes:
        if hasattr(node, "ed_info"):
            ori_meta_all[node.name] = node.ed_info.ori_meta
            node.ed_info.ori_meta = None
            md_info_all[node.name] = node.ed_info

    broadcast_result = [md_info_all]
    torch.distributed.broadcast_object_list(broadcast_result, src=0, device="cuda")
    md_info_all = broadcast_result[0]

    for node in fx_module.graph.nodes:
        if hasattr(node, "ed_info"):
            node.ed_info = md_info_all[node.name]
            node.ed_info.ori_meta = ori_meta_all[node.name]

    return fx_module
