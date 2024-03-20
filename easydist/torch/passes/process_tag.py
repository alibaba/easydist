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

import torch
import torch._custom_ops

from easydist.torch.device_mesh import get_device_mesh
from easydist.torch.passes.sharding import all_reduce_start, all_reduce_end


@torch._custom_ops.custom_op("easydist::tag")
def tag(input: torch.Tensor, tag: str) -> torch.Tensor:
    ...

@torch._custom_ops.impl_abstract("easydist::tag")
def tag_impl_abstract(input: torch.Tensor, tag: str) -> torch.Tensor:
    return torch.empty_like(input)


@torch._custom_ops.impl("easydist::tag")
def tag_impl(input: torch.Tensor, tag: str) -> torch.Tensor:
    return input


def process_tag(traced_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:

    device_mesh = get_device_mesh()

    for node in traced_graph.graph.nodes:
        if node.target == torch.ops.easydist.tag.default:
            if node.args[1] == "allreduce[sum]":
                assert "tp" in device_mesh.mesh_dim_names
                tp_mesh = device_mesh["tp"]
                reduceOp = "sum"
                ranks = tp_mesh.mesh.flatten().tolist()
                with traced_graph.graph.inserting_before(node):
                    all_reduce_start_node = traced_graph.graph.call_function(all_reduce_start,
                                                                                args=(node.args[0],
                                                                                    reduceOp,
                                                                                    ranks))
                    all_reduce_end_node = traced_graph.graph.call_function(
                        all_reduce_end, args=(all_reduce_start_node, reduceOp, ranks))
                
                node.replace_all_uses_with(all_reduce_end_node)

    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    return traced_graph