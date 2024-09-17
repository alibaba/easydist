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

from copy import deepcopy
import logging
from typing import Dict, Any, List
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Placement
import torch.fx as fx
import operator
from torch.fx.passes.graph_drawer import FxGraphDrawer
import easydist.config as mdconfig
from easydist.torch.device_mesh import get_device_mesh


def ordered_gi_users(node: fx.node):
    assert all(user.op == 'call_function' and user.target == operator.getitem
               for user in node.users), "All users of the node must be getitem"
    ret = [None for _ in range(len(node.users))]
    for user in node.users:
        ret[user.args[1]] = user
    return ret


def save_graphviz_dot(gm, name):
    if mdconfig.log_level <= logging.DEBUG:
        with open(f"./log/{name}.dot", "w") as f:
            f.write(str(FxGraphDrawer(gm, name).get_dot_graph()))


def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )


class OneToOneMap:
    def __init__(self):
        self._map: Dict[Any, Any]= {}
        self._inv: Dict[Any, Any] = {}

    def get(self, key: Any) -> Any:
        return self._map[key]

    def inv(self, key: Any) -> Any:
        return self._inv[key]

    def add(self, key: Any, value: Any) -> Any:
        if key in self._map or value in self._map:
            raise RuntimeError(f"{key}: {value} is not one to one mapping, found {key in self._map} {value in self._inv}")
        self._map[key] = value
        self._inv[value] = key

    def items(self):
        return self._map.items()

    def keys(self):
        return self._map.keys()

    def inv_items(self):
        return self._inv.items()

    def inv_keys(self):
        return self._inv.keys()

    def apply(self, other: "OneToOneMap") -> "OneToOneMap":
        mapping = OneToOneMap()
        for k, v in self.items():
            mapping.add(k, other.get(v))
        return mapping

    def map_dict_key(self, dictt: Dict) -> Dict:
        ret = {}
        for k, v in dictt.items():
            ret[self._map[k]] = v
        return ret

    def inverse(self) -> "OneToOneMap":
        inversed = deepcopy(self)
        inversed._map, inversed._inv = inversed._inv, inversed._map
        return inversed

    def __repr__(self):
        return f"{self._map=}\n{self._inv=}"

    def intersect(self, other: "OneToOneMap") -> "OneToOneMap":
        mapping = OneToOneMap()
        for k, v in self.items():
            if k in other._map:
                mapping.add(k, v)
        return mapping

    @staticmethod
    def from_dict(dict: Dict) -> "OneToOneMap":
        mapping = OneToOneMap()
        for k, v in dict.items():
            mapping.add(k, v)
        return mapping


def do_spmd_comm(tensor, src_specs: List[Placement], tgt_specs: List[Placement]):
    device_mesh = get_device_mesh('spmd')
    dtensor = DTensor.from_local(tensor, device_mesh, src_specs)
    return dtensor.redistribute(device_mesh, tgt_specs).to_local()
