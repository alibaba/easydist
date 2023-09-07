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

log_level_str = os.environ.get("METADIST_LOGLEVEL", 'INFO').upper()
log_level = logging.getLevelName(log_level_str)

dump_dir = os.environ.get("METADIST_DUMP_PATH", './md_dump')
enable_compile_cache = os.environ.get("ENABLE_COMPILE_CACHE", "False").upper() in ["1", "TRUE"]
compile_cache_dir = os.environ.get("METADIST_COMPILE_CACHE_PATH", './md_compiled')

metadist_device = os.environ.get("METADIST_DEVICE", "cuda")

# MetaSPMD Annotation

use_hint = os.environ.get("METADIST_USE_HINT") == "1"
extend_space = False

# Solver

liveness_only_input = False
max_seconds_same_incumbent = float('inf')

all_to_all_punish_factor = 3.

enable_graph_coarsen = os.environ.get("ENABLE_GRAPH_COARSEN", "True").upper() in ["1", "TRUE"]
coarsen_level = int(os.environ.get("COARSEN_LEVEL", "1"))

# runtime

use_contiguous_buffer = False
