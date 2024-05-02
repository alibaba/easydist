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
import torch.fx as fx
import operator
from torch.fx.passes.graph_drawer import FxGraphDrawer
import easydist.config as mdconfig


def ordered_gi_users(node: fx.node):
    assert all(user.op == 'call_function' and user.target == operator.getitem
               for user in node.users), "All users of the node must be getitem"
    ret = [None for _ in range(len(node.users))]
    for user in node.users:
        ret[user.args[1]] = user
    return ret


def save_graphviz_dot(gm, name):
    if mdconfig.log_level <= logging.DEBUG:
        with open(f"./log/{name}.dot", "w") as f:
            f.write(str(FxGraphDrawer(gm, name).get_dot_graph()))


def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )
