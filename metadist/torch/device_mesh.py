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

import numpy
import torch
from torch.distributed._tensor import DeviceMesh

from metadist.metashard import metair
from metadist.utils.testing import TorchMockDeviceMesh

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


def get_device_mesh():
    global TORCH_DEVICE_MESH
    return TORCH_DEVICE_MESH


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
