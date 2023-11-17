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

import torch.fx as fx
from torch.fx.node import _get_qualified_name


def eliminate_detach(fx_graph: fx.GraphModule):
    for node in fx_graph.graph.nodes:
        if node.op == 'call_function':
            if _get_qualified_name(node.target) == 'torch.ops.aten.detach.default':
                node.replace_all_uses_with(node.args[0])

    fx_graph.graph.eliminate_dead_code()

    return fx_graph
