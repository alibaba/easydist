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
import functools
import operator
from contextlib import contextmanager

import numpy
import torch
from torch.distributed._tensor import DeviceMesh, mesh_resources

from easydist.metashard import metair
from easydist.utils.testing import TorchMockDeviceMesh

logger = logging.getLogger(__name__)

TORCH_DEVICE_MESH = None


def set_device_mesh(device_mesh):
    global TORCH_DEVICE_MESH
    TORCH_DEVICE_MESH = device_mesh

    if device_mesh.size(0) == 1:
        metair.DEVICE_MESH_1D = 0
    elif device_mesh.size(1) == 1:
        metair.DEVICE_MESH_1D = 1

    logger.info(f"set_device_mesh: {device_mesh}")

def device_mesh_world_size(device_mesh=None):
    if device_mesh is None:
        device_mesh = get_device_mesh()

    if device_mesh is None:
        return None

    if isinstance(device_mesh, TorchMockDeviceMesh):
        device_mesh_shape = get_device_mesh().shape
    elif isinstance(device_mesh, DeviceMesh):
        device_mesh_shape = tuple(get_device_mesh().mesh.shape)

    return functools.reduce(operator.mul, device_mesh_shape)


def device_mesh_rank(device_mesh, dim):
    if device_mesh is None:
        device_mesh = get_device_mesh()

    if isinstance(device_mesh, TorchMockDeviceMesh):
        global_rank = torch.distributed.distributed_c10d.get_rank()
        device_mesh_shape = get_device_mesh().shape
        world_size = functools.reduce(operator.mul, device_mesh_shape)
        rank_coords = (numpy.arange(world_size).reshape(
            *device_mesh_shape) == global_rank).nonzero()
        rank = rank_coords[dim].item()
    elif isinstance(device_mesh, DeviceMesh):
        rank = device_mesh.get_coordinate_on_dim(dim)
    else:
        raise RuntimeError("DeviceMesh not support or not initialize")

    return rank

# [spmd0, spmd1, pp]
__with_spmd_device_mesh = None  # TODO @botbw: remove this function tag

@contextmanager
def spmd_device_mesh(pp_rank=None):
    global __with_spmd_device_mesh
    ori_with_spmd_device_mesh = __with_spmd_device_mesh
    if pp_rank is None:
        __with_spmd_device_mesh = TORCH_DEVICE_MESH.get_coordinate_on_dim(2)
    else:
        __with_spmd_device_mesh = pp_rank
    try:
        yield
    finally:
        __with_spmd_device_mesh = ori_with_spmd_device_mesh

def get_device_mesh():
    global TORCH_DEVICE_MESH
    if __with_spmd_device_mesh is not None:
        return __get_spmd_device_mesh(TORCH_DEVICE_MESH)
    return TORCH_DEVICE_MESH

def get_spmd_size():
    global TORCH_DEVICE_MESH
    return TORCH_DEVICE_MESH.size(0), TORCH_DEVICE_MESH.size(1)

def get_spmd_rank():
    global TORCH_DEVICE_MESH
    return TORCH_DEVICE_MESH.get_coordinate_on_dim(0), TORCH_DEVICE_MESH.get_coordinate_on_dim(1)

def get_pp_size():
    global TORCH_DEVICE_MESH
    return TORCH_DEVICE_MESH.size(2)

def get_pp_rank():
    global TORCH_DEVICE_MESH
    return TORCH_DEVICE_MESH.get_coordinate_on_dim(2)

def get_pp_group():
    global TORCH_DEVICE_MESH
    return TORCH_DEVICE_MESH.get_dim_groups(2)

def __get_spmd_device_mesh(device_mesh):
    assert __with_spmd_device_mesh is not None 

    if device_mesh is None:
        return None
    
    ranks = device_mesh.mesh[:, :, __with_spmd_device_mesh]
    spmd_mesh = DeviceMesh(
        device_mesh.device_type, ranks, _init_process_groups=False
    )
    spmd_mesh._dim_group_infos = device_mesh._dim_group_infos[:2]
    mesh_resources.child_to_parent_mapping[device_mesh] = spmd_mesh
    return spmd_mesh
