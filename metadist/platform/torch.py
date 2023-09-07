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

from functools import partial

import torch
import torch.utils._pytree as pytree

add = torch.Tensor.add
equal = torch.equal
zeros_like = torch.zeros_like
min = torch.min
max = torch.max
allclose = partial(torch.allclose, rtol=1e-3, atol=1e-07)
concatenate = torch.concatenate
chunk = torch.chunk
narrow = torch.narrow

Tensor = torch.Tensor

tree_flatten = pytree.tree_flatten
tree_unflatten = pytree.tree_unflatten

clone = torch.clone
from_numpy = torch.from_numpy
