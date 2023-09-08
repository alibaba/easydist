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


def md_embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if int(torch.max(indices).item()) >= weight.shape[0]:
        raise RuntimeError("embedding indice overflow")
    return torch.ops.aten.embedding.default(weight, indices, padding_idx, scale_grad_by_freq,
                                            sparse)


def fix_embedding(fx_module: torch.fx.GraphModule, recover=False):

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if "torch.ops.aten.embedding.default" in _get_qualified_name(node.target):
                node.target = md_embedding

            if recover and "md_embedding" in _get_qualified_name(node.target):
                node.target = torch.ops.aten.embedding.default

    fx_module.recompile()

    return fx_module
