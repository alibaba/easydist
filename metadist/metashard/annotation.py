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

import copy
from typing import List

from metadist import platform
from metadist.metashard.halo import HaloInfo


class ShardDim:
    """
    ``ShardDim`` is used to store sharding information for ``ShardAnnotation``.

    Within ``ShardAnnotation``, each dimension of the tensor list is assigned a ``ShardDim``. 
    A value of ``ShardDim(0)`` means that the dimension cannot be sharded. 
    All dimensions with a value of ``ShardDim(i)`` can be sharded together.

    ShardDim can also carry information about halo and chunk.

    - ``halo`` means that padding is needed for each shard.
    - ``chunk`` means that we need firstly chunk the data along the dimension,
      then shard for each chunk to get the sharded data. similar to block-cyclic.

      [1, 1, 2, 2] -(shard)-> [1, 1] | [2, 2] when chunk = 1
      [1, 1, 2, 2] -(chunk)-> [1, 1] and [2, 2] -(shard)-> [1, 2] | [1, 2]

    """

    def __init__(self, shard_dim_id: int, chunk: int = 1) -> None:
        self.shard_dim_id = shard_dim_id
        self.chunk = chunk

        self.halo: HaloInfo = None

    def __str__(self) -> str:
        if self.shard_dim_id == 0:
            return "NoShardDim"
        else:
            content = str(self.shard_dim_id)
            if self.chunk > 1:
                content += f", chunk={self.chunk})"
            if self.halo:
                content += f", halo={self.halo})"
            return f"ShardDim({content})"

    def set_halo(self, halo_: HaloInfo):
        self.halo = halo_

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def get_noshard_dim():
        return ShardDim(0)

    @staticmethod
    def get_shard_dim(shard_dim_id):
        return ShardDim(shard_dim_id)


class ShardAnnotation:
    """
    ``ShardAnnotation`` used to describe the sharding strategy space for a ``MetaOp``.
    
    For example:
    - matmul: ShardAnnotation([[ShardDim(1), ShardDim(2)], [ShardDim(2), ShardDim(3)]])
    - relu: ShardAnnotation([[ShardDim(1), ShardDim(2)], [ShardDim(1), ShardDim(2)]])
    - layernorm: ShardAnnotation([[ShardDim(1), ShardDim(2), NoShardDim]])
    """

    def __init__(self, annotation: List[List[ShardDim]]) -> None:
        self.annotation = annotation

    @staticmethod
    def init_from_input_args(input_args):
        return ShardAnnotation([[ShardDim.get_noshard_dim()] * len(t.shape) for t in input_args
                                if isinstance(t, platform.Tensor)])

    def inject_haloinfo(self, haloinfo: HaloInfo, shard_dim_idx: int):
        """set haloinfo for each ShardDim(shard_dim_idx) in ShardAnnotation"""
        if haloinfo is None:
            return
        for anno in self.annotation:
            for shard_dim in anno:
                if shard_dim.shard_dim_id == shard_dim_idx:
                    shard_dim.set_halo(haloinfo)

    def get_max_shard_dim_id(self) -> int:
        max_shard_dim_id = 0
        for anno in self.annotation:
            for dim_anno in anno:
                max_shard_dim_id = max(max_shard_dim_id, dim_anno.shard_dim_id)
        return max_shard_dim_id

    def clear_shard_dim(self, max_shard_dim_id):
        new_anno = copy.deepcopy(self)
        for anno in new_anno.annotation:
            for dim_idx in range(len(anno)):
                if anno[dim_idx].shard_dim_id > max_shard_dim_id:
                    anno[dim_idx] = ShardDim.get_noshard_dim()
        return new_anno

    def __str__(self) -> str:
        return f"ShardAnnotation({self.annotation.__str__()})"

    def __repr__(self) -> str:
        return f"ShardAnnotation({self.annotation.__repr__()})"

    def __getitem__(self, idx):
        return self.annotation[idx]

    def __setitem__(self, idx, shard_dim):
        self.annotation[idx] = shard_dim

    def __len__(self) -> int:
        return self.annotation.__len__()

    def __add__(self, other):
        return ShardAnnotation(self.annotation + other.annotation)
