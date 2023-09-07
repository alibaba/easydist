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

import os
import copy

import torch

import metadist.config as mdconfig


def fix_meta_device(fx_module: torch.fx.GraphModule):

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if "device" in node.kwargs:
                new_kwargs = dict(copy.deepcopy(node.kwargs))
                device = mdconfig.metadist_device
                new_kwargs["device"] = torch.device(device=device)
                assert isinstance(new_kwargs, dict)
                node.kwargs = new_kwargs

    fx_module.recompile()

    return fx_module
