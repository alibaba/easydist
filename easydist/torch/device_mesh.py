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

import logging
from copy import deepcopy
from functools import cache
from typing import Dict, Optional, Sequence, Union

import torch
from torch.distributed._tensor import DeviceMesh
from functools import lru_cache

if torch.__version__ >= (2, 3):
    from torch.distributed._tensor.device_mesh import _mesh_resources as mesh_resources
else:
    from torch.distributed._tensor import mesh_resources

logger = logging.getLogger(__name__)

class NDDeviceMesh(DeviceMesh):
    _device_mesh: DeviceMesh
    _binding: Dict[str, Sequence[str]]

    def __init__(self, torch_mesh: DeviceMesh):
        self._device_mesh = torch_mesh
        self._binding = {}

        for name, dim_sz in zip(torch_mesh.mesh_dim_names, torch_mesh.mesh.shape):
            if dim_sz <= 1:
                raise RuntimeError(f"{name} dimenstion size must be greater than 1")

        if self._device_mesh.mesh_dim_names is None:
            raise RuntimeError("mesh_dim_names is required")

    def __check_valid_names(self, dim_names: Sequence[str]):
        for name in dim_names:
            if name not in self._device_mesh.mesh_dim_names:
                raise ValueError(f"Invalid dim name: {name}")

    def __map_binding(self, dim_names: Union[str, Sequence[str]]) -> Sequence[str]:
        if isinstance(dim_names, str):
            dim_names = [dim_names]

        if len(dim_names) == 1:
            name = dim_names[0]
            if name in self._binding:
                return self._binding[name]

        self.__check_valid_names(dim_names)

        return dim_names

    def __get_dims(self, dim_names: Sequence[str]) -> Sequence[int]:
        return [self._device_mesh.mesh_dim_names.index(name) for name in dim_names]

    @cache
    def __getitem__(self, names: Union[str, Sequence[str]]) -> 'NDDeviceMesh':
        names = self.__map_binding(names)
        # self coordinates
        coord_cp = deepcopy(self._device_mesh.get_coordinate())
        target_dims = self.__get_dims(names)
        for dim in target_dims:
            coord_cp[dim] = slice(None)
        submesh_mesh = self._device_mesh.mesh[coord_cp]

        if torch.__version__ < (2, 3):
            submesh = DeviceMesh(device_type=self._device_mesh.device_type, mesh=submesh_mesh, mesh_dim_names=names, _init_process_groups=False)
        elif torch.__version__ == (2, 3, 0):
            submesh = DeviceMesh(device_type=self._device_mesh.device_type, mesh=submesh_mesh, mesh_dim_names=names)
        else:
            submesh = DeviceMesh(device_type=self._device_mesh.device_type, mesh=submesh_mesh, mesh_dim_names=names, _init_backend=False)

        submesh._dim_group_infos = [self._device_mesh._dim_group_infos[i] for i in target_dims]
        if torch.__version__ >= (2, 4):
            mesh_resources.child_to_root_mapping[self._device_mesh] = submesh
        else:
            mesh_resources.child_to_parent_mapping[self._device_mesh] = submesh

        return NDDeviceMesh(submesh)

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._device_mesh

    # DeviceMesh.size(dim: Optional[int]) ->  DeviceMesh.size(mesh_dim: Optional[int]) since torch 2.2.0
    if torch.__version__ < (2, 2):
        def size(self, dim: Optional[int] = None) -> int:
            return self._device_mesh.size(dim)
    else:
        def size(self, mesh_dim: Optional[int] = None) -> int:
            return self._device_mesh.size(mesh_dim)

    def __getattr__(self, name: str):
        return getattr(self._device_mesh, name)

    def __repr__(self) -> str:
        return f"NDDeviceMesh({self._device_mesh})"

    def bind(self, name: str, names: Sequence[str]):
        self.__check_valid_names(names)
        assert name not in self._binding, f"Name {name} is already bound"
        self._binding[name] = names

def __bind_spmd(mesh: NDDeviceMesh):
    names = []
    for name in mesh.mesh_dim_names:
        if "spmd" in name:
            names.append(name)
    if len(names) > 0:
        mesh.bind("spmd", names)

__DEFAULT_BINDINGS = [
    __bind_spmd
]

__GLOBAL_ND_DEVICEMESH: Optional[NDDeviceMesh] = None

def set_device_mesh(torch_mesh: DeviceMesh, default_binding: bool=True):
    global __GLOBAL_ND_DEVICEMESH
    __GLOBAL_ND_DEVICEMESH = NDDeviceMesh(torch_mesh)

    if default_binding:
        for bind_func in __DEFAULT_BINDINGS:
            bind_func(__GLOBAL_ND_DEVICEMESH)

    # TODO @botbw: better implementation for mesh initializtion
    _ = get_device_mesh('spmd')

    logger.info(f"set_device_mesh: {torch_mesh}")

@lru_cache(maxsize=1024)
def get_device_mesh(*dim_names) -> NDDeviceMesh:
    if __GLOBAL_ND_DEVICEMESH is None:
        raise RuntimeError("Device mesh hasn't been set, please set a Torch device mesh first.")

    if len(dim_names) > 0:
        return __GLOBAL_ND_DEVICEMESH[dim_names]

    return __GLOBAL_ND_DEVICEMESH

if __name__ == "__main__":
    import os
    rank = int(os.environ.get("RANK"))

    set_device_mesh(DeviceMesh("cuda", [
        [
            [0, 1], # spmd1 ->
            [2, 3]
        ],
        [
            [4, 5],
            [6, 7]
        ]
    ], mesh_dim_names=["pp", "spmd0", "spmd1"]))

    mesh = get_device_mesh()

    if rank == 5:
        print(mesh['spmd0', 'spmd1'])
        print(mesh['spmd0', 'spmd1'].get_coordinate())
        print(mesh['spmd1'])
        print(mesh['pp'])
        print(mesh['pp'].get_coordinate())
        print(mesh['spmd'])
        print(get_device_mesh('spmd'))
