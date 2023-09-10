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
from typing import Dict, Tuple

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.placement_types import (DTensorSpec, Replicate,
                                                       Shard, _Partial)
from torch.fx.node import _get_qualified_name

from easydist.torch.device_mesh import get_device_mesh

# used to select proper output for the same function
# [CAUTION] only support single static graph
global graph_cnt
graph_cnt: Dict[str, Tuple[int, int]] = {}
global graph_output_struct
graph_output_struct: Dict[str, str] = {}

logger = logging.getLogger(__name__)

aten = torch.ops.aten


def _transform_to_Placemnet(varstrtg):
    res = []
    for strtg in varstrtg:
        if strtg.is_replicate():
            res.append(Replicate())
        elif strtg.is_shard():
            res.append(Shard(strtg.args['dim']))
        elif strtg.is_partial():
            res.append(_Partial(eval('c10d.' + strtg.args['ops'].__str__())))
        else:
            RuntimeError('Rule_override: strtg not recognized')
    return res


def rule_override_by_graph(fx_module: torch.fx.GraphModule, opt_strategy, fwd_shape_info):

    def operator_rule_factory(op_name, graph_rules):

        def operator_rule(op_schema: OpSchema) -> OutputSharding:
            current_cnt, lim = graph_cnt[op_name]

            mesh = get_device_mesh()
            dtensor_specs = []
            tot_kwargs = graph_rules[op_name][current_cnt]

            for kwargs in tot_kwargs:
                kwargs['mesh'] = mesh
                dtensor_specs.append(DTensorSpec(**kwargs))

            struct = graph_output_struct[op_name]
            if struct == 'elem':
                res_specs = dtensor_specs[0]
            elif struct == 'list':
                res_specs = list(dtensor_specs)
            elif struct == 'tuple':
                res_specs = tuple(dtensor_specs)

            graph_cnt[op_name] = ((current_cnt + 1) % lim, lim)
            return OutputSharding(output_spec=res_specs)

        return operator_rule

    graph_rules: dict[str, list] = {}

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            op_name = _get_qualified_name(node.target)

            if node.name.__contains__('redist_tensor_func') \
                    or node.name.__contains__('getitem') \
                    or node.name.__contains__('native_layer_norm_backward'):
                continue
            strategy = opt_strategy[node.name]['strategy']
            output_spec_list = []

            output_shapes = fwd_shape_info[node.name]
            if not isinstance(output_shapes, tuple):
                if isinstance(output_shapes, list):
                    graph_output_struct[op_name] = 'list'
                    tot_info = tuple(output_shapes)
                else:
                    graph_output_struct[op_name] = 'elem'
                    tot_info = (output_shapes,)
            else:
                graph_output_struct[op_name] = 'tuple'
                tot_info = output_shapes

            for var_strtg, tensor_info in zip(strategy.out_strtg_group,
                                              tot_info):
                spec = _transform_to_Placemnet(var_strtg.var_spmd_strategy)
                shape = tensor_info['shape']
                cur_kwargs = {
                    'placements': spec,
                    'ndim': len(shape),
                    'shape': shape
                }
                output_spec_list.append(cur_kwargs)

            if graph_rules.get(op_name) is None:
                graph_rules[op_name] = []
                graph_cnt[op_name] = (0, 0)

            graph_rules[op_name].append(output_spec_list)

            graph_cnt[op_name] = (0, graph_cnt[op_name][1] + 1)

    for op_name in graph_rules:
        DTensor._propagator.register_sharding_prop_rule(
            eval(op_name),
            operator_rule_factory(op_name, graph_rules)
        )

    return fx_module
