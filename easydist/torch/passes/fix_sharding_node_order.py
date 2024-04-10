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
from easydist.torch.passes.comm_optimize import _link_nodes
from easydist.torch.passes.sharding import CUSTOM_FUNCS

def fix_sharding_node_order(fx_module: fx.GraphModule):
    nodes = list(fx_module.graph.nodes)
    for i in range(len(nodes)):
        node = nodes[i]
        if node.target in CUSTOM_FUNCS:
            ancessor_id = max(map(lambda n: nodes.index(n), node.all_input_nodes))
            if nodes[ancessor_id].op == 'placeholder':
                continue
            nodes.pop(i)
            nodes.insert(ancessor_id + 1, node)

    assert set(nodes) == set(fx_module.graph.nodes)
    _link_nodes(fx_module, nodes)
    fx_module.recompile()
    return fx_module
