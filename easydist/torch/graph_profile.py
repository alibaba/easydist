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
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.utils._pytree as pytree
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter
from torch.fx.node import Target, Argument, _get_qualified_name
from torch.distributed._tensor.api import DTensor

from easydist.torch.graph_profile_db import PerfDB


def to_meta(_node_output):
    if type(_node_output) is torch.Tensor:
        return _node_output.to(device="meta").contiguous()
    elif type(_node_output) is torch.nn.parameter.Parameter:
        return _node_output.data.to(device="meta").contiguous()
    elif type(_node_output) is DTensor:
        _node_output._local_tensor = _node_output._local_tensor.to("meta")
        return _node_output
    else:
        return _node_output


def get_shape_info(_node_output):
    if type(_node_output) in [torch.Tensor, torch.nn.parameter.Parameter]:
        return {"shape": _node_output.shape, "dtype": _node_output.dtype}
    else:
        return _node_output


class HyperPerfMeasure(Interpreter):

    def __init__(self,
                 module: GraphModule,
                 trials=2,
                 warmup_trials=1,
                 garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.trials = trials
        self.warmup_trials = warmup_trials
        self.shape_info = {}

        self.pass_by_ops = [
            "_operator.getitem", "torch.ops.aten.view", "torch.ops.aten._unsafe_view"
        ]

        self.sum_time = 0
        self.perf_db = PerfDB()

    def node_runtime(self):
        assert(self.sum_time > 0)
        called_time = {}
        node_to_time = {}
        for node in self.module.graph.nodes:
            if node.op == 'call_function':
                ops_name = _get_qualified_name(node.target)
                if ops_name in self.pass_by_ops:
                    continue
                if called_time.get(ops_name) is None:
                    called_time[ops_name] = 0
                node_to_time[node.name] = self.perf_db.get_op_perf(ops_name, called_time[ops_name])['time']
        return node_to_time

    def run(self, *args) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = {}
        self.node_cnt = {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        self.args_iter: Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:

            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            _node_output = self.run_node(node)
            self.env[node] = pytree.tree_map(to_meta, _node_output)
            self.shape_info[node.name] = pytree.tree_map(get_shape_info, self.env[node])

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == 'output':
                output_val = self.env[node]
                self.env = {}
                self.node_cnt = {}
                return output_val, self.shape_info

    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str,
                                                                                       Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        ops_name = _get_qualified_name(target)

        if ops_name in self.pass_by_ops:
            real_out = target(*args, **kwargs)
            return real_out

        args_meta = pytree.tree_map(to_meta, args)
        kwargs_meta = pytree.tree_map(to_meta, kwargs)

        if self.node_cnt.get(ops_name) is None:
            self.node_cnt[ops_name] = 0
        op_perf_key = {"ops_name": ops_name, "called_time": self.node_cnt[ops_name]}
        self.node_cnt[ops_name] += 1

        db_record = self.perf_db.get_op_perf(op_perf_key["ops_name"], op_perf_key["called_time"])

        if db_record is None:

            output_device = "cuda"

            # materialize the input of ops, execute the function and return the result
            args = pytree.tree_map(
                lambda x: torch.ones_like(x, device=output_device)
                if isinstance(x, torch.Tensor) else x, args)
            kwargs = pytree.tree_map(
                lambda x: torch.ones_like(x, device=output_device)
                if isinstance(x, torch.Tensor) else x, kwargs)

            start_evt_ = []
            end_evt_ = []
            for _ in range(0, self.trials):
                start_evt_.append(torch.cuda.Event(enable_timing=True))
                end_evt_.append(torch.cuda.Event(enable_timing=True))

            for trial_idx_ in range(0, self.trials + self.warmup_trials):
                evt_idx = trial_idx_ - self.warmup_trials

                if evt_idx >= 0:
                    start_evt_[evt_idx].record()

                real_out = target(*args, **kwargs)

                if evt_idx >= 0:
                    end_evt_[evt_idx].record()

            torch.cuda.synchronize()
            ops_elapsed_time_ = 0
            for evt_idx in range(0, self.trials):
                # time elapsed in **milliseconds**
                ops_elapsed_time_ += start_evt_[evt_idx].elapsed_time(end_evt_[evt_idx])
            ops_elapsed_time_ = ops_elapsed_time_ / self.trials

            db_record = {
                "output_meta": pytree.tree_map(to_meta, real_out),
                "time": ops_elapsed_time_
            }
            if torch.distributed.get_rank() == 0:
                print(ops_name)
                print(db_record["time"])

            self.perf_db.record_op_perf(op_perf_key["ops_name"], op_perf_key["called_time"], db_record)

        self.sum_time += db_record["time"]
        return db_record["output_meta"]