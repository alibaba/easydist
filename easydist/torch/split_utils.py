# Copyright (c) 2024, Alibaba Group;
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

from typing import List, Union, Tuple, Any, Dict, Callable

import torch


_before_split: Dict[type, Callable[[Any], List[torch.Tensor]]] = {}
_after_split: Dict[type, Callable[[List[torch.Tensor]], Any]] = {}

def before_split_register(*classes):

    def _register(func: Callable):
        for cls in classes:
            assert cls not in _before_split, f"split function for {cls} already registered"
            _before_split[cls] = func
        return func

    return _register


def after_split_register(*classes):

    def _register(func: Callable):
        for cls in classes:
            assert cls not in _after_split, f"split function for {cls} already registered"
            _after_split[cls] = func
        return func

    return _register


@before_split_register(torch.Tensor)
def tensor_before_split(ctx: dict, input: torch.Tensor) -> List[torch.Tensor]:
    return [input]


@after_split_register(torch.Tensor)
def tensor_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> torch.Tensor:
    return output[0]


@before_split_register(list)
def list_before_split(ctx: dict, input: List[Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
    ctx['is_tensor'] = []
    ctx['non_tensor_vals'] = []
    tup = []
    for x in input:
        ctx['is_tensor'].append(isinstance(x, torch.Tensor))
        if ctx['is_tensor'][-1]:
            tup.append(x)
        else:
            ctx['non_tensor_vals'].append(x)

    return tup


@after_split_register(list)
def list_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> List[Union[torch.Tensor, Any]]:
    ret = []
    output = list(output)
    for is_tensor in ctx['is_tensor']:
        if is_tensor:
            ret.append(output.pop(0))
        else:
            ret.append(ctx['non_tensor_vals'].pop(0))
    return ret


@before_split_register(tuple)
def tuple_before_split(ctx: dict, input: Tuple[Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
    return list_before_split(ctx, list(input))


@after_split_register(tuple)
def tuple_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> Tuple[Union[torch.Tensor, Any]]:
    return tuple(list_after_split(ctx, output))


