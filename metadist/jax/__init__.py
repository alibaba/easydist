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

import os
import logging

import jax
from mpi4py import MPI

from .api import metadist_shard, get_opt_strategy, set_device_mesh
from .sharding_interpreter import MDJaxShardingAnn
from .bridge import jax2md_bridge

__all__ = [
    "metadist_shard", "get_opt_strategy", "set_device_mesh", "MDJaxShardingAnn", "jax2md_bridge"
]

logger = logging.getLogger(__name__)


def is_jax_distributed_initialized():
    return jax._src.distributed.global_state.client is not None


def metadist_setup_jax(device, allow_tf32):
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1" if allow_tf32 else "0"
    jax.config.update('jax_platforms', device)

    # setup distributed
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()

    if not is_jax_distributed_initialized():

        jax.distributed.initialize(coordinator_address="localhost:19705",
                                   num_processes=size,
                                   process_id=rank,
                                   local_device_ids=rank)

        logging.info(
            f"[Rank {rank}], Global Devices: {jax.device_count()}, Local Devices: {jax.local_device_count()}"
        )
