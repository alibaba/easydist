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

import functools
import operator
import torch
import torch.fx as fx

from easydist.torch.preset_propagation import VIEW_OPS


def _fix_view_node(input_shape, output_shape):
    if -1 in output_shape:
        numel = functools.reduce(operator.mul, input_shape)
        dim_size = -1 * numel // functools.reduce(operator.mul, output_shape)
        output_shape[output_shape.index(-1)] = dim_size

    intermediate_shape = []
    i = j = 0
    while i < len(input_shape) and j < len(output_shape):
        accu_i = input_shape[i]
        accu_j = output_shape[j]
        while accu_i != accu_j:
            if accu_i < accu_j:
                i += 1
                accu_i *= input_shape[i]
            else:
                j += 1
                accu_j *= output_shape[j]
        intermediate_shape.append(accu_i)
        i += 1
        j += 1
    while i < len(input_shape):
        intermediate_shape.append(input_shape[i])
        i += 1
    while j < len(output_shape):
        intermediate_shape.append(output_shape[j])
        j += 1
    assert i == len(input_shape) and j == len(output_shape)
    return intermediate_shape


def decouple_view(fx_module: fx.GraphModule):
    for node in fx_module.graph.nodes:
        if node.op == 'call_function' and node.target in VIEW_OPS:
            target_op = node.target
            input_shape = list(node.args[0].meta['val'].shape)
            output_shape = list(node.args[1])
            intermediate_shape = _fix_view_node(input_shape, output_shape)
            if input_shape != intermediate_shape and output_shape != intermediate_shape:
                node.args = (node.args[0], intermediate_shape)
                fake_mode = node.meta['val'].fake_mode
                node.meta['val'] = fake_mode.from_tensor(
                    torch.zeros(intermediate_shape,
                                dtype=node.meta['val'].dtype))
                with fx_module.graph.inserting_after(node):
                    intermediate_view = fx_module.graph.call_function(
                        target_op, args=(node, output_shape))
                    intermediate_view.meta['val'] = fake_mode.from_tensor(
                        torch.zeros(output_shape,
                                    dtype=node.meta['val'].dtype))
                node.replace_all_uses_with(intermediate_view, delete_user_cb=lambda x: x != intermediate_view)

    fx_module.recompile()
    return fx_module

# def fix_sharded_view(fx_module: fx.GraphModule):
#     for node in fx_module.graph.nodes:
#         if (node.op == 'call_function' and node.target in VIEW_OPS and
#                 node.args[0].op == 'call_function' and node.args[0].target == scatter_wrapper):
#             input_val = node.args[0].meta['val']
#             output_shape = list(node.args[1])
#             try:
#                 _ = input_val.view(output_shape)
#             except Exception:

#             # global [768 (shard), 768] => view as [3, 256, 768]
#             # 0 [384, 768] => view as [1, 256, 768]
#             # 1 [384, 768] => view as [2, 256, 768]