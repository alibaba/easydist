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

from metadist.metashard.annotation import ShardAnnotation, ShardDim
from metadist.metashard.combination import CombinationFunc

EXTEND_VIEW = False


def get_next_non_one(shape_, idx_):
    if idx_ < len(shape_):
        while shape_[idx_] == 1:
            idx_ += 1
            if idx_ >= len(shape_):
                break
    return idx_


def view_propagation(input_shape, output_shape, world_size=1):

    if -1 in output_shape:
        numel = functools.reduce(operator.mul, input_shape)
        dim_size = -1 * numel // functools.reduce(operator.mul, output_shape)
        output_shape[output_shape.index(-1)] = dim_size

    sharding_ann = ShardAnnotation([[ShardDim(0)] * len(input_shape)])
    combination_ann = {}

    input_idx = get_next_non_one(input_shape, 0)
    output_idx = get_next_non_one(output_shape, 0)

    shard_dim = 1

    while input_idx < len(input_shape):
        if input_shape[input_idx] == output_shape[output_idx]:
            # [**, A, **] -> [**, A, **]
            if input_shape[input_idx] >= world_size:
                sharding_ann[0][input_idx] = ShardDim(shard_dim)
                combination_ann[shard_dim] = functools.partial(CombinationFunc.gather,
                                                               dim=output_idx)
                shard_dim += 1
            input_idx = get_next_non_one(input_shape, input_idx + 1)
            output_idx = get_next_non_one(output_shape, output_idx + 1)
        elif input_shape[input_idx] > output_shape[output_idx]:
            # [**, A, **] -> [**, a1, a2, **]
            leftmost_idx = output_idx
            accum_shape_ = output_shape[output_idx]
            for o_idx in range(output_idx + 1, len(output_shape)):
                accum_shape_ *= output_shape[o_idx]
                if accum_shape_ == input_shape[input_idx]:
                    if output_shape[leftmost_idx] >= world_size:
                        sharding_ann[0][input_idx] = ShardDim(shard_dim)
                        combination_ann[shard_dim] = functools.partial(CombinationFunc.gather,
                                                                       dim=leftmost_idx)
                        shard_dim += 1
                    output_idx = get_next_non_one(output_shape, o_idx + 1)
                    input_idx = get_next_non_one(input_shape, input_idx + 1)
                    break
        else:
            # [**, a1, a2, **] -> [**, A, **]
            leftmost_idx = input_idx
            accum_shape_ = input_shape[input_idx]
            for i_idx in range(input_idx + 1, len(input_shape)):
                accum_shape_ *= input_shape[i_idx]
                if accum_shape_ == output_shape[output_idx]:
                    if EXTEND_VIEW:
                        chunk_size_ = 1
                        for sub_idx in range(input_idx, i_idx + 1):
                            sharding_ann[0][sub_idx] = ShardDim(shard_dim)
                            combination_ann[shard_dim] = functools.partial(CombinationFunc.gather,
                                                                           dim=output_idx,
                                                                           chunk=chunk_size_)
                            chunk_size_ *= input_shape[sub_idx]
                            shard_dim += 1
                    else:
                        if input_shape[input_idx] >= world_size:
                            sharding_ann[0][input_idx] = ShardDim(shard_dim)
                            combination_ann[shard_dim] = functools.partial(CombinationFunc.gather,
                                                                           dim=output_idx)
                            shard_dim += 1

                    output_idx = get_next_non_one(output_shape, output_idx + 1)
                    input_idx = get_next_non_one(input_shape, i_idx + 1)
                    break

    return {'sharding_ann': sharding_ann, 'combination_ann': combination_ann}


def view_propagation_preset(input_shape, output_shape, preset_anno):
    accum_size = 1
    for idx, ann in enumerate(preset_anno[0]):
        if ann.shard_dim_id != 0:
            break
        accum_size *= input_shape[idx]

    chunk = preset_anno[0][idx].chunk

    out_accum_size = 1
    out_idx = 0
    while out_accum_size < accum_size:
        out_accum_size *= output_shape[out_idx]
        out_idx += 1

    if out_accum_size == accum_size:
        accum_chunk = 1
        if chunk == accum_chunk:
            return functools.partial(CombinationFunc.gather, dim=out_idx)
        for o_idx in range(out_idx, len(output_shape)):
            if chunk == accum_chunk:
                return functools.partial(CombinationFunc.gather, dim=o_idx)
            accum_chunk *= output_shape[o_idx]
