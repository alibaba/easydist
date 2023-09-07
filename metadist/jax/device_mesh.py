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

from metadist.metashard import metair

JAX_DEVICE_MESH = None

def set_device_mesh(device_mesh):
    global JAX_DEVICE_MESH
    JAX_DEVICE_MESH = device_mesh

    mesh_shape = device_mesh.device_ids.shape
    if mesh_shape[0] == 1:
        metair.DEVICE_MESH_1D = 0
    elif mesh_shape[1] == 1:
        metair.DEVICE_MESH_1D = 1


def get_device_mesh():
    global JAX_DEVICE_MESH
    return JAX_DEVICE_MESH


def device_mesh_world_size(device_mesh=None):
    if device_mesh is None:
        device_mesh = get_device_mesh()

    if device_mesh is None:
        return None

    device_mesh_shape = device_mesh.device_ids.shape

    return functools.reduce(operator.mul, device_mesh_shape)
