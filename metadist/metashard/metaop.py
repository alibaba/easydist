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
import logging

import numpy as np

from metadist import platform
from metadist.metashard.halo import HaloInfo, halo_padding
from metadist.metashard.annotation import ShardDim, ShardAnnotation
from metadist.metashard.combination import try_combination, HaloHint

logger = logging.getLogger(__name__)


def get_shard_size(dim_to_shard, shard_size):
    gcd = np.gcd.reduce(dim_to_shard)
    if gcd < shard_size:
        return gcd
    # elif gcd % shard_size == 0:
    #     return shard_size
    # return -1
    return shard_size


def get_shard_data(input_args, shard_size, dim_idx, haloinfo, chunk):
    chunk_data = platform.chunk(platform.clone(input_args), chunks=chunk, dim=dim_idx)
    chunk_shard_data = [
        platform.chunk(chunk_, chunks=shard_size, dim=dim_idx) for chunk_ in chunk_data
    ]
    shard_data = [
        platform.concatenate([t[i] for t in chunk_shard_data], dim_idx) for i in range(shard_size)
    ]

    return halo_padding(shard_data, haloinfo=haloinfo)


def check_prompt(flat_input_args, prompt_annotation):
    tensor_args = [i for i in flat_input_args if isinstance(flat_input_args, platform.Tensor)]
    if len(tensor_args) == len(prompt_annotation):
        for idx in range(len(tensor_args)):
            if tensor_args.ndim != len(prompt_annotation[idx]):
                return False
        return True
    return False


class MetaOp:

    def __init__(self, func, input_args, shard_size=2, name=None) -> None:
        self.func = func
        self.name = name
        if self.name is None:
            self.name = self.func.__name__
        self.input_args = input_args
        self.shard_size = shard_size
        self.flat_input_args, self.input_pytree_spec = platform.tree_flatten(input_args)

    def exec_platform(self, flat_input_args):
        if platform.get_backend() in ["torch", "tvm"]:
            args, kwargs = platform.tree_unflatten(flat_input_args, self.input_pytree_spec)
            return self.func(*args, **kwargs)
        elif platform.get_backend() == "jax":
            # (NOTE) disable_jit here for jax >= 4.7, the pjit primitive doesn't accept chunked input
            import jax
            with jax.disable_jit():
                flat_input_args = platform.tree_unflatten(flat_input_args, self.input_pytree_spec)
                if len(flat_input_args) == 3 and isinstance(flat_input_args[2], dict):
                    subfuns, invars, bind_params = flat_input_args
                    return self.func(*subfuns, *invars, **bind_params)
                return self.func(*flat_input_args)

    def exec(self, shard_annotation=None, priority_shard_dim_id=None, haloinfo=None):
        local_flat_input_args = self.flat_input_args

        tensor_list = [i for i in local_flat_input_args if isinstance(i, platform.Tensor)]

        if shard_annotation:
            dim_to_shard = []
            for tensor_ann_idx, tensor_ann in enumerate(shard_annotation):
                for dim_idx, dim in enumerate(tensor_ann):
                    if dim.shard_dim_id == priority_shard_dim_id:
                        dim_to_shard.append(tensor_list[tensor_ann_idx].shape[dim_idx])
                        break
            if len(dim_to_shard) == 0:
                raise RuntimeError(f"can not find priority_shard_dim_id {priority_shard_dim_id}")
            shard_size = get_shard_size(dim_to_shard, self.shard_size)
            if shard_size < self.shard_size:
                raise RuntimeError("can not find suitable shard_size")
            else:
                sharded_out = []
                for shard_idx in range(shard_size):
                    sharded_flat_input_args = []
                    tensor_arg = 0
                    for input_args in local_flat_input_args:
                        if isinstance(input_args, platform.Tensor):
                            tensor_ann = shard_annotation[tensor_arg]
                            no_shard_here = True
                            for dim_idx, dim in enumerate(tensor_ann):
                                if dim.shard_dim_id == priority_shard_dim_id:
                                    shard_data = get_shard_data(input_args, shard_size, dim_idx,
                                                                haloinfo, dim.chunk)[shard_idx]
                                    sharded_flat_input_args.append(shard_data)
                                    no_shard_here = False
                                    break
                            if no_shard_here:
                                sharded_flat_input_args.append(platform.clone(input_args))
                            tensor_arg += 1
                        else:
                            sharded_flat_input_args.append(input_args)

                    sharded_out.append(self.exec_platform(sharded_flat_input_args))
                return sharded_out

        else:
            return self.exec_platform(local_flat_input_args)

    def _try_sharding(self, fixed_annotation, subsequence_annotation, global_output, start_dim=0):
        shard_dim_id_flag = self.__shard_dim_id_flag

        if self.__find_new_strategy:
            return

        if len(subsequence_annotation) == 0:
            try:
                sharded_output = self.exec(shard_annotation=fixed_annotation,
                                           priority_shard_dim_id=shard_dim_id_flag)
            except RuntimeError as e:
                logger.debug(f"[{fixed_annotation}] {e}")
                return
            except:
                logger.debug(f"[{fixed_annotation}] run op.exec failed")
                return

            haloinfo = None
            combination_func = try_combination(sharded_output, global_output)
            if isinstance(combination_func, HaloHint):
                ecpt = combination_func
                if ecpt.halo <= 0:
                    ecpt.halo = 1
                if isinstance(sharded_output[0], tuple) or isinstance(sharded_output[0], list):
                    try_max_halo = sharded_output[0][ecpt.out_idx].shape[ecpt.dim] // 2
                else:
                    try_max_halo = sharded_output[0].shape[ecpt.dim] // 2
                for halo in range(ecpt.halo, try_max_halo):
                    sharded_output = self.exec(shard_annotation=fixed_annotation,
                                               priority_shard_dim_id=shard_dim_id_flag,
                                               haloinfo=HaloInfo(halo, ecpt.dim))
                    combination_func = try_combination(sharded_output, global_output)
                    if isinstance(combination_func, HaloHint):
                        combination_func = None
                    if combination_func:
                        haloinfo = HaloInfo(halo, ecpt.dim)
                        break

            if combination_func is not None and not isinstance(combination_func, HaloHint):
                self.__combination_ann[shard_dim_id_flag] = combination_func
                # inject haloinfo
                fixed_annotation.inject_haloinfo(haloinfo, shard_dim_id_flag)

                self.__sharding_annotion = copy.deepcopy(fixed_annotation)
                self.__find_new_strategy = True
            else:
                logger.debug(f"[{fixed_annotation}] combination failed")
            return

        for dim in range(start_dim, len(subsequence_annotation[0])):
            if subsequence_annotation[0][dim].shard_dim_id != 0:
                continue
            try_annotation = copy.deepcopy(subsequence_annotation[0])
            try_annotation[dim] = ShardDim.get_shard_dim(shard_dim_id_flag)
            self._try_sharding(fixed_annotation + ShardAnnotation([try_annotation]),
                               subsequence_annotation[1:], global_output)

        self._try_sharding(fixed_annotation + ShardAnnotation([subsequence_annotation[0]]),
                           subsequence_annotation[1:], global_output)

    def sharding_discovery(self, prompt_annotation=None):

        self.__combination_ann = {}
        # validate the prompt_annotation
        if prompt_annotation is not None and check_prompt(self.flat_input_args, prompt_annotation):
            max_shard_dim_id = prompt_annotation.get_max_shard_dim_id()
            for shard_dim_id in range(1, max_shard_dim_id + 1):
                combination_func = self.sharding_discovery_with_preset(
                    prompt_annotation, shard_dim_id)

                if combination_func is not None:
                    self.__combination_ann[shard_dim_id] = combination_func
                else:
                    break

        validate_max_shard_dim_id = len(self.__combination_ann)

        if validate_max_shard_dim_id >= 1:
            self.__sharding_annotion = prompt_annotation.clear_shard_dim(validate_max_shard_dim_id)
        else:
            # init the annotation with all NoShardDim
            self.__sharding_annotion = ShardAnnotation.init_from_input_args(self.flat_input_args)

        global_output = self.exec()

        self.__shard_dim_id_flag = validate_max_shard_dim_id + 1
        self.__find_new_strategy = False

        if len(self.__sharding_annotion) == 0:
            return self.__sharding_annotion, self.__combination_ann

        fixed_annotation = ShardAnnotation([])
        subsequence_annotation = self.__sharding_annotion
        start_dim = 0
        while True:
            self._try_sharding(fixed_annotation, subsequence_annotation, global_output, start_dim)
            if self.__find_new_strategy:

                # if shard_dim_id_flag not in sharding_annotion, then break
                max_shard_dim_id_now = 0
                for ann in self.__sharding_annotion:
                    for dim in ann:
                        max_shard_dim_id_now = max(max_shard_dim_id_now, dim.shard_dim_id)
                if max_shard_dim_id_now != self.__shard_dim_id_flag:
                    break

                self.__find_new_strategy = False
                # find the first dim for shard_dim_id_flag now
                # and update fixed_annotation, subsequence_annotation, start_dim
                fixed_annotation = ShardAnnotation([])
                finded = False
                for ann_idx, ann in enumerate(self.__sharding_annotion):
                    for idx, dim in enumerate(ann):
                        if self.__shard_dim_id_flag == dim.shard_dim_id:
                            finded = True
                            start_dim = idx + 1
                            break
                    if finded:
                        break
                    fixed_annotation += ShardAnnotation([ann])
                subsequence_annotation = self.__sharding_annotion[ann_idx:]
                if ann_idx == len(self.__sharding_annotion) - 1 and start_dim == len(
                        self.__sharding_annotion[-1]):
                    break
                self.__shard_dim_id_flag += 1
            else:
                break

        logger.debug(f"sharding_annotion of {self.name}: {self.__sharding_annotion}")

        return self.__sharding_annotion, self.__combination_ann

    def sharding_discovery_with_preset(self, sharding_annotion, priority_shard_dim_id=1):
        global_output = self.exec()

        try:
            sharded_output = self.exec(shard_annotation=sharding_annotion,
                                       priority_shard_dim_id=priority_shard_dim_id)
        except RuntimeError as e:
            logger.debug(f"[{sharding_annotion}] {e}")
            return
        except:
            logger.debug(f"[{sharding_annotion}] run op.exec failed")
            return

        combination_func = try_combination(sharded_output, global_output)
        if combination_func is not None and not isinstance(combination_func, HaloHint):
            return combination_func
