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
from torch.distributed._tensor import DTensor, Replicate
from torch.fx.node import Node, _get_qualified_name, map_arg

from metadist.torch.device_mesh import device_mesh_rank, get_device_mesh
from metadist.utils.testing import MockDeviceMesh

logger = logging.getLogger(__name__)


def materialize(x, device):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bool:
            return torch.rand(x.size(), dtype=torch.float, device=device) > 0.5
        elif torch.is_floating_point(x):
            return torch.rand(x.size(), dtype=x.dtype, device=device)
        else:
            return torch.randint(high=8, size=x.size(), dtype=x.dtype, device=device)
    return x


def redist_tensor_func(input_tensor, specs):
    if isinstance(input_tensor, DTensor) and input_tensor.size() != torch.Size([0]):
        device_mesh = get_device_mesh()
        if specs != input_tensor._spec.placements:
            return input_tensor.redistribute(device_mesh, specs).contiguous()
    return input_tensor.contiguous()


def insert_spmd_head(body):
    return ["from torch.distributed._tensor import Shard, Replicate\n", *body]


@torch.no_grad()
def to_dtensor(input_tensor, placements=None, global_size=None):
    if isinstance(input_tensor, torch.Tensor) and not isinstance(input_tensor, DTensor):
        device_mesh = get_device_mesh()
        if placements is None:
            placements = [Replicate()] * device_mesh.ndim
        if global_size is None:
            return DTensor.from_local(input_tensor, device_mesh, placements)
        else:
            return DTensor(input_tensor, device_mesh, placements, size=global_size)
    return input_tensor


def fix_in_gragh_tensor(fx_module: torch.fx.GraphModule, sharding_strategy):

    for node in fx_module.graph.nodes:
        if node.op == 'get_attr':
            with fx_module.graph.inserting_after(node):
                to_dtensor_node = fx_module.graph.call_function(to_dtensor, args=(node, ))

                node.replace_all_uses_with(to_dtensor_node)

                to_dtensor_node.update_arg(0, node)

        create_ops = [
            "torch.ops.aten.scalar_tensor.", "torch.ops.aten.ones.", "torch.ops.aten.empty.",
            "torch.ops.aten.arange.", "torch.ops.aten.zeros."
        ]

        if node.op == 'call_function':
            for op in create_ops:
                if op in _get_qualified_name(node.target):
                    strategy, global_shape = None, None
                    if node.name in sharding_strategy and len(sharding_strategy[node.name]) > 0:
                        strategy = sharding_strategy[node.name][0]
                        # modify the arg of creat ops
                        # (CAUTION!!) device_mesh maybe mock now,
                        # but the local shape need to be consistent of runtime
                        global_shape = node.args[0]
                        local_shape = list(global_shape)
                        device_mesh = get_device_mesh()
                        if isinstance(device_mesh, MockDeviceMesh):
                            logger.warning("maybe wrong shape in MockDeviceMesh for create ops.")
                        for idx, placement in enumerate(strategy):
                            if placement.is_shard():
                                shard_dim = placement.dim
                                split_size, pad_idx = divmod(global_shape[shard_dim],
                                                             device_mesh.size(idx))
                                local_shape[shard_dim] = split_size
                                if device_mesh_rank(device_mesh, idx) < pad_idx:
                                    local_shape[shard_dim] = split_size + 1

                        node.update_arg(0, local_shape)

                    with fx_module.graph.inserting_after(node):
                        to_dtensor_node = fx_module.graph.call_function(to_dtensor,
                                                                        args=(node, strategy,
                                                                              global_shape))

                        node.replace_all_uses_with(to_dtensor_node)

                        to_dtensor_node.update_arg(0, node)

    fx_module.recompile()

    return fx_module


def replace_subsequence_use(node, arg_, redist_node):
    users_node = list(node.users.keys())
    node_next = node.next

    def maybe_replace_node(n: Node) -> Node:
        if n == arg_:
            return redist_node
        else:
            return n

    while node_next.name != "":
        if node_next in users_node:
            new_args = map_arg(node_next.args, maybe_replace_node)
            new_kwargs = map_arg(node_next.kwargs, maybe_replace_node)
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            node_next.args = new_args
            node_next.kwargs = new_kwargs
        node_next = node_next.next


def sharding_transform(fx_module: torch.fx.GraphModule, sharding_strategy):
    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if node.name in sharding_strategy:
                node_strategy = sharding_strategy[node.name]

                # skip for create ops
                ops_name = _get_qualified_name(node.target)
                if ops_name in [
                        "torch.ops.aten.empty.memory_format", "torch.ops.aten.zeros.default"
                ]:
                    continue

                node_args_flatten = pytree.tree_flatten(node.args)[0]
                invars = [arg for arg in node_args_flatten if isinstance(arg, Node)]
                assert (len(invars) == len(node_strategy))

                for arg_, arg_strategy in zip(invars, node_strategy):
                    with fx_module.graph.inserting_before(node):
                        redist_node = fx_module.graph.call_function(redist_tensor_func,
                                                                    args=(arg_, arg_strategy))

                        # FIXME: may introduce redundancy communication, update_arg for all subsequence use
                        node.replace_input_with(arg_, redist_node)
                        #replace_subsequence_use(node, arg_, redist_node)

    fx_module.graph.on_generate_code(lambda _: insert_spmd_head)

    fx_module.recompile()

    # (fix) %_tensor_constant0 : [#users=1] = get_attr[target=_tensor_constant0]
    fx_module = fix_in_gragh_tensor(fx_module, sharding_strategy)

    return fx_module
