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

import easydist.config as mdconfig


_logger = logging.getLogger(__name__)


class PerfDB:

    def __init__(self) -> None:
        self.db = dict()
        if os.path.exists(mdconfig.prof_db_path):
            self.db = pickle.load(open(mdconfig.prof_db_path, 'rb'))
            _logger.info(f"Load Perf DB from {mdconfig.prof_db_path}")

    def get_op_perf(self, key_l1, key_l2):
        if key_l1 in self.db:
            return self.db[key_l1].get(key_l2, None)
        return None

    def record_op_perf(self, key_l1, key_l2, value):
        if key_l1 not in self.db:
            self.db[key_l1] = dict()    
        self.db[key_l1][key_l2] = value

    def persistent(self):
        _logger.info(f"Persistent Perf DB to {mdconfig.prof_db_path}")

        if not os.path.exists(mdconfig.easydist_dir):
            os.makedirs(mdconfig.easydist_dir, exist_ok=True)

        pickle.dump(self.db, open(mdconfig.prof_db_path, 'wb'))
