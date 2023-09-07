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

from . import platform
import metadist.config as mdconfig


def metadist_setup(backend, device="cpu", allow_tf32=True):
    mdconfig.metadist_device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        level=mdconfig.log_level)

    if backend == "jax":
        from .jax import metadist_setup_jax
        metadist_setup_jax(device, allow_tf32)

        logging.getLogger("jax._src").setLevel(logging.INFO)
    elif backend == "torch":
        from .torch import metadist_setup_torch
        metadist_setup_torch(device, allow_tf32)

        logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.INFO)

    platform.init_backend(backend)
