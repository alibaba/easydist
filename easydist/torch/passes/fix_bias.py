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
from torch.fx.node import _get_qualified_name


def fix_addmm_bias(fx_module: torch.fx.GraphModule):

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if "torch.ops.aten.addmm.default" in _get_qualified_name(node.target):
                node.target = torch.ops.aten.mm.default
                bias = node.args[0]
                node.args = (node.args[1], node.args[2])

                with fx_module.graph.inserting_after(node):
                    add_bias_node = fx_module.graph.call_function(torch.ops.aten.add.Tensor,
                                                                  args=(node, bias))

                    node.replace_all_uses_with(add_bias_node)

                    add_bias_node.update_arg(0, node)

    fx_module.recompile()

    return fx_module


def fix_convoluation_bias(fx_module: torch.fx.GraphModule):

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if "torch.ops.aten.convolution.default" in _get_qualified_name(node.target):
                if node.args[2] is not None:
                    node.target = torch.ops.aten.convolution.default
                    bias = node.args[2]
                    node.args = (node.args[0], node.args[1], None, *node.args[3:])

                    with fx_module.graph.inserting_after(node):
                        bias_new = fx_module.graph.call_function(torch.ops.aten.view.default,
                                                                 args=(bias, [1, -1, 1, 1]))

                    with fx_module.graph.inserting_after(bias_new):
                        add_bias_node = fx_module.graph.call_function(torch.ops.aten.add.Tensor,
                                                                      args=(node, bias_new))

                        node.replace_all_uses_with(add_bias_node)

                        add_bias_node.update_arg(0, node)

    fx_module.recompile()

    return fx_module
