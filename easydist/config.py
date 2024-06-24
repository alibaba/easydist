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
from pathlib import Path


def _get_env_or_raise(env_var: str) -> str:
    env_val = os.environ.get(env_var, None)
    if not env_val:
        raise ValueError(f"environment variable {env_var} expected, but not set")
    else:
        return env_val


log_level_str = os.environ.get("EASYDIST_LOGLEVEL", 'INFO').upper()
log_level = logging.getLevelName(log_level_str)

dump_dir = os.environ.get("EASYDIST_DUMP_PATH", './md_dump')
enable_compile_cache = os.environ.get("ENABLE_COMPILE_CACHE", "False").upper() in ["1", "TRUE"]
compile_cache_dir = os.environ.get("EASYDIST_COMPILE_CACHE_PATH", './md_compiled')

easydist_device = os.environ.get("EASYDIST_DEVICE", "cuda")

available_mem = 40 * 1024 * 1024 # (KB)

forced_compile = os.environ.get("EASYDIST_FORCED_COMPILE", "False").upper() in ["1", "TRUE"]
use_dtensor = False

# Profiling

profile_trials = 5
profile_warmup_trials = 2

easydist_dir = os.path.join(Path.home(), ".easydist")
prof_db_path = os.path.join(easydist_dir, "perf.db")
dump_prof_db = False

dump_fx_graph = os.environ.get("DUMP_FX_GRAPH", "False").upper() in ["1", "TRUE"]

# MetaSPMD Annotation

use_hint = os.environ.get("EASYDIST_USE_HINT") == "1"
extend_space = False

# Solver

liveness_only_input = False
max_seconds_same_incumbent = float('inf')

all_to_all_punish_factor = 3.

enable_graph_coarsen = os.environ.get("ENABLE_GRAPH_COARSEN", "True").upper() in ["1", "TRUE"]
coarsen_level = int(os.environ.get("COARSEN_LEVEL", "0"))

# Master address and port
master_addr = os.environ.get("MASTER_ADDR", "localhost")
master_port = int(os.environ.get("MASTER_PORT", 12888))

# PyTorch

# Tile communication
enable_tile_comm = False
critical_context_length = 100
tile_context_length = 15
nvlink_processor_usage = 0.15

# Scheduling communication
comm_optimization = False
# 'general', 'odd_even'
rcpsp_method = 'general'
rcpsp_iter_round = 1 # odd_even rounds
override_dtensor_rule = False

# runtime
use_contiguous_buffer = False

# memory optimization
enable_memory_opt = os.environ.get("ENABLE_MEMORY_OPT", "False").upper() in ["1", "TRUE"]

# ignore memory reuse
ignore_memory_reuse = os.environ.get("IGNORE_MEMORY_REUSE", "False").upper() in ["1", "TRUE"]

# reschedule ops
enable_reschedule = os.environ.get("ENABLE_RESCHEDULE", "False").upper() in ["1", "TRUE"]

# memory usage optimization by solver
mem_opt_by_solver = os.environ.get("MEM_OPT_BY_SOLVER", "False").upper() in ["1", "TRUE"]

# dump memory usage graph
dump_mem_usage_graph = os.environ.get("DUMP_MEM_USAGE_GRAPH", "False").upper() in ["1", "TRUE"]

# runtime trace
enable_runtime_trace = os.environ.get("ENABLE_RUNTIME_TRACE", "False").upper() in ["1", "TRUE"]

