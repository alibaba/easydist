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
import pickle
import logging
from pathlib import Path

DEFAULT_HYPER_DIR = os.path.join(Path.home(), ".hyper")
DEFAULT_DB_PATH = os.path.join(DEFAULT_HYPER_DIR, "perf.db")

_logger = logging.getLogger(__name__)


class PerfDB:

    def __init__(self) -> None:
        self.db = dict()
        if os.path.exists(DEFAULT_DB_PATH):
            self.db = pickle.load(open(DEFAULT_DB_PATH, 'rb'))
            _logger.info(f"Load Perf DB from {DEFAULT_DB_PATH}")

    def get_op_perf(self, key):
        if key["ops_name"] in self.db:
            return self.db[key["ops_name"]].get(key["called_time"], None)
        return None

    def record_op_perf(self, key, value):
        if key["ops_name"] not in self.db:
            self.db[key["ops_name"]] = dict()    
        self.db[key["ops_name"]][key["called_time"]] = value

    def persistent(self):
        _logger.info(f"Persistent Perf DB to {DEFAULT_DB_PATH}")

        if not os.path.exists(DEFAULT_HYPER_DIR):
            os.makedirs(DEFAULT_HYPER_DIR, exist_ok=True)

        pickle.dump(self.db, open(DEFAULT_DB_PATH, 'wb'))
