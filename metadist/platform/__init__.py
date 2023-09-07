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
import importlib

logger = logging.getLogger(__name__)

METADIST_BACKEND = None

__all__ = [
    "add", "equal", "zeros_like", "min", "max", "allclose", "concatenate", "chunk", "narrow",
    "Tensor", "tree_flatten", "tree_unflatten", "clone", "from_numpy"
]


def backend_valid(_backend):
    return _backend in {"torch", "jax", "tvm"}


def init_backend(backend="torch"):
    assert backend_valid(backend)
    global METADIST_BACKEND
    METADIST_BACKEND = backend
    modules = importlib.import_module("." + backend, __name__)
    for val in __all__:
        exec("globals()['%s'] = modules.%s" % (val, val))
    logger.info(f"========= MetaDist init with backend {backend}. =========")


def get_backend():
    global METADIST_BACKEND
    return METADIST_BACKEND


for val in __all__:
    exec("globals()['%s'] = None" % val)
