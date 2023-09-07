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
from contextlib import contextmanager


@contextmanager
def _sharding_ann_env():

    ori_tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE", None)

    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

    try:
        yield
    finally:
        if ori_tf32_override is None:
            os.environ.pop("NVIDIA_TF32_OVERRIDE")
        os.environ["NVIDIA_TF32_OVERRIDE"] = ori_tf32_override
