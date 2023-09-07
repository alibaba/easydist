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

from typing import List

from metadist import platform


class HaloInfo:

    def __init__(self, halowidth: int, dim: int) -> None:
        self.halowidth = halowidth
        self.dim = dim

    def __str__(self) -> str:
        return self.halowidth.__str__()

    def __repr__(self) -> str:
        return self.__str__()


def halo_padding(tensor_list_: List[platform.Tensor], haloinfo: HaloInfo) -> List[platform.Tensor]:
    """add halo padding to tensor_list_"""

    if haloinfo is None or len(tensor_list_) < 2:
        return tensor_list_

    halo = haloinfo.halowidth
    dim = haloinfo.dim

    padded_tensor_list = []
    for idx in range(len(tensor_list_)):
        to_concatenate = [tensor_list_[idx]]
        if idx >= 1:
            dim_size = tensor_list_[idx - 1].shape[dim]
            if dim_size < halo:
                raise RuntimeError("Cannot halo padding for this sharded_tensor")
            to_concatenate.insert(
                0, platform.narrow(tensor_list_[idx - 1], dim, dim_size - halo, halo))
        if idx <= len(tensor_list_) - 2:
            to_concatenate.append(platform.narrow(tensor_list_[idx + 1], dim, 0, halo))
        padded_tensor_list.append(platform.concatenate(to_concatenate, dim=dim))

    return padded_tensor_list
