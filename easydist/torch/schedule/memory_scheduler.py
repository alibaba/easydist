from collections import defaultdict, OrderedDict

import sys
import logging
import time
import numpy as np
import torch
from enum import Enum
from typing import Tuple, Set

import easydist.config as mdconfig
from easydist.torch.mem_allocation_info import GraphMemInfo

from mip import (
    Model,
    xsum,
    BINARY,
    INTEGER,
    minimize,
    SearchEmphasis,
    OptimizationStatus,
)

logger = logging.getLogger(__name__)

class OverlapType(Enum):
    OVERLAP_FORCE_SHARE = 1  # lifetimes definitely overlap: inplace or getitem operator
    OVERLAP_NOT_SHARE = 2    # lifetimes definitely overlap: neither inplace nor getitem operator
    POSSIBLE_OVERLAP = 3     # not sure if lifetimes overlap before scheduling
    NEVER_OVERLAP = 4        # lifetimes definitely not overlap: no requirement for share

class MemSharedTensorGroup:
    def __init__(self):
        self.tensors = set()
        self.sink_nodes_cache = set()

    def add_tensor(self, tensor: Tuple[torch.fx.Node, int]):
        self.tensors.add(tensor)

    def has_mult_tensors(self):
        return len(self.tensors) > 1

    def source_asap_max(self, depend_info) -> int:
        asap_max = 0
        for tensor in self.tensors:
            node_asap = depend_info.asap(tensor[0])
            if asap_max < node_asap:
                asap_max = node_asap
        return asap_max

    def source_asap_min(self, depend_info ) -> int:
        asap_min = sys.maxsize
        for tensor in self.tensors:
            node_asap = depend_info.asap(tensor[0])
            if asap_min > node_asap:
                asap_min = node_asap
        return asap_min

    def source_alap_max(self, depend_info) -> int:
        alap_max = 0
        for tensor in self.tensors:
            node_alap = depend_info.alap(tensor[0])
            if alap_max < node_alap:
                alap_max = node_alap
        return alap_max

    def source_alap_min(self, depend_info) -> int:
        alap_min = sys.maxsize
        for tensor in self.tensors:
            node_alap = depend_info.alap(tensor[0])
            if alap_min > node_alap:
                alap_min = node_alap
        return alap_min

    def sink_nodes(self) -> Set[torch.fx.Node]:
        if not self.sink_nodes_cache:
            nodes_in_group = set()
            for tensor in self.tensors:
                nodes_in_group.add(tensor[0])

            for node in nodes_in_group:
                for user in node.users:
                    #if user in nodes_in_group:
                    #    continue
                    self.sink_nodes_cache.add(user)
        return self.sink_nodes_cache

    def sink_asap_max(self, depend_info) -> int:
        sink_nodes_ = self.sink_nodes()
        asap_max = 0
        for sink in sink_nodes_:
            asap = depend_info.asap(sink)
            if asap_max < asap:
                asap_max = asap
        return asap_max

    def sink_alap_max(self, depend_info) -> int:
        sink_nodes_ = self.sink_nodes()
        alap_max = 0
        for sink in sink_nodes_:
            alap = depend_info.alap(sink)
            if alap_max < alap:
                alap_max = alap
        return alap_max


class DependencyInfo:
    def __init__(self,
                 nodes_to_schedule,     # list of nodes to be scheduled
                 graph_mem_info,        # GraphMemInfo
                 user_sched_constraints,
                 one_step_one_op):     # user's schedule
        self.nodes_to_schedule = nodes_to_schedule
        self.graph_mem_info = graph_mem_info
        self.node_set = set(nodes_to_schedule)
        self.cache_prev_nodes = {}
        self.cache_post_nodes = {}
        self.user_sched_constraints = user_sched_constraints
        self.one_step_one_op = one_step_one_op

        # map: node -> (asap, alap)
        self.node_makespans = {}

        # map: edge(node out) -> (asap, alap)
        self.edge_makespans = {}

        self.depths = {}
        self.max_depth = 0
        for node in self.nodes_to_schedule:
            if node in self.depths:
                continue
            max_arg_depth = 0
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg not in self.node_set:
                        continue
                    assert arg in self.depths
                    max_arg_depth = max(max_arg_depth, self.depths[arg])
            depth = max_arg_depth+1
            self.depths[node] = depth
            self.max_depth = max(self.max_depth, depth)

        if one_step_one_op:
            self.num_steps = len(self.nodes_to_schedule)
        else:
            self.num_steps = self.max_depth + 1

        # build memory share groups
        # map: (node, tensor_idx) -> the set of nodes which share the same memory
        self.mem_share_info = defaultdict(lambda: MemSharedTensorGroup())
        # NOTE: It is more efficient to traverse all nodes by topological order
        for node in self.nodes_to_schedule:
            node_mem_info = graph_mem_info.get_node_mem_info(node.name)
            for out_tensor_info in node_mem_info.out_tensor_infos:
                out_tensor_key = (node, out_tensor_info.out_index)
                if out_tensor_info.is_reference:
                    arg_idx = out_tensor_info.arg_index
                    tensor_idx = out_tensor_info.tensor_index
                    pre_node = node.args[arg_idx]

                    pre_shared_group = self.mem_share_info[(pre_node, tensor_idx)]
                    if out_tensor_key in self.mem_share_info:
                        out_shared_group = self.mem_share_info[out_tensor_key]
                        if id(pre_shared_group) != id(out_shared_group):
                            # merge two groups
                            for out_set_item in out_shared_group:
                                pre_shared_group.add_tensor(out_set_item)
                                self.mem_share_info[out_set_item] = pre_shared_group
                    else:
                        pre_shared_group.add_tensor(out_tensor_key)
                        self.mem_share_info[out_tensor_key] = pre_shared_group
                else:
                    out_shared_group = self.mem_share_info[out_tensor_key]
                    out_shared_group.add_tensor(out_tensor_key)

        ## debug: dump memory-shared tensor group
        #for share_key, tensor_group in self.mem_share_info.items():
        #    print(("node: {}, out idx: {}, share set id: {}").format(
        #                      share_key[0].name, share_key[1], id(tensor_group)))

        #dumped_group_set = set()
        #for tensor_group in self.mem_share_info.values():
        #    if id(tensor_group) in dumped_group_set:
        #        continue
        #    print(("share set id: {}").format(id(tensor_group)))
        #    dumped_group_set.add(id(tensor_group))
        #    for tensor in tensor_group.tensors:
        #        print(("  node: {}, out idx: {}").format(tensor[0].name, tensor[1]))

    def get_overlap_type(self, node1: torch.fx.Node, out_idx1: int,
                         node2: torch.fx.Node, out_idx2: int) -> OverlapType:
        # lifetime relationship between two variables:
        # 1. var1 memory and var2 memory never overlap
        #    1.1. all users of one var are predecessors of another var's source
        #         set type OverlapType.NEVER_OVERLAP
        #    1.2. spans of two variables don't overlap
        #         set type OverlapType.NEVER_OVERLAP
        # 2. var1 memory and var2 memory definitely overlap sometime
        #    2.1. var1 and var2 are in the same shared memory group
        #         set type OverlapType.OVERLAP_FORCE_SHARE
        #    2.2. var1's memory and var2's memory are used by the same node.
        #         such as
        #             1). var1 and var2 are connected by a node
        #             2). one var in var1's memory-shared group and another var
        #                 in var2's memory-shared group are connected by a node
        #         set type OverlapType.OVERLAP_NOT_SHARE
        #    2.3. determine share type by analyzing variable span:
        #         define range1 as below:
        #         [min(alap(sources(var1_mem))), max(asap(sinks(var1_mem)))]
        #         where var1's memory is kept in use during range1.
        #         define range2 as below:
        #         [min(alap(sources(var2_mem))), max(asap(sinks(var2_mem)))]
        #         where var2's memory is kept in use during range2.
        #         2.3.1. If any variable that resides in var2_mem is definitely
        #                created in range1, that means var1's memory is
        #                overlapped with var2's memory
        #                set type OverlapType.OVERLAP_NOT_SHARE
        #         2.3.2. If any variable that resides in var1_mem is definitely
        #                created in range2, that means var2's memory is
        #                overlapped with var1's memory
        #                set type OverlapType.OVERLAP_NOT_SHARE
        # 3. others:
        #         var1 memory and var2 memory may overlap or not, it depends on
        #         schedule:
        #         set type OverlapType.POSSIBLE_OVERLAP

        tensor1_key = (node1, out_idx1)
        tensor1_group = self.mem_share_info[tensor1_key]
        tensor2_key = (node2, out_idx2)
        tensor2_group = self.mem_share_info[tensor2_key]
        if id(tensor1_group) == id(tensor2_group):
            return OverlapType.OVERLAP_FORCE_SHARE

        # two tensors never overlap?
        src_asap_min1, snk_alap_max1 = self.get_possible_live_span(tensor1_group)
        src_asap_min2, snk_alap_max2 = self.get_possible_live_span(tensor2_group)

        if snk_alap_max1 < src_asap_min2 or src_asap_min1 > snk_alap_max2:
            # never overlap
            return OverlapType.NEVER_OVERLAP

        if (
            (not tensor1_group.has_mult_tensors())
            and (not tensor2_group.has_mult_tensors())
        ):
            if self.is_user_before(node1, node2):
                # never overlap
                return OverlapType.NEVER_OVERLAP

            if self.is_user_before(node2, node1):
                # never overlap
                return OverlapType.NEVER_OVERLAP

        live_same_time = False
        if self.are_connected_by_node(node1, out_idx1, node2, out_idx2):
            live_same_time = True

        if not live_same_time:
            tensor1_live_span = self.get_definite_live_span(node1, out_idx1)
            if tensor1_live_span[0] <= tensor1_live_span[1]:
                if self.is_created_in_span(tensor2_group, tensor1_live_span):
                    live_same_time = True

        if not live_same_time:
            tensor2_live_span = self.get_definite_live_span(node2, out_idx2)
            if tensor2_live_span[0] <= tensor2_live_span[1]:
                if self.is_created_in_span(tensor1_group, tensor2_live_span):
                    live_same_time = True

        if live_same_time:
            return OverlapType.OVERLAP_NOT_SHARE
        else:
            return OverlapType.POSSIBLE_OVERLAP

    def is_created_in_span(self, tensor_group, span):
        for tensor in tensor_group.tensors:
            node_asap, node_alap = self.node_makespans[tensor[0]]
            if span[0] <= node_asap and span[1] >= node_alap:
                return True

        return False

    def get_possible_live_span(self, tensor_group):
        if tensor_group.has_mult_tensors():
            src_asap_min = tensor_group.source_asap_min(self)
            snk_alap_max = tensor_group.sink_alap_max(self)
        else:
            tensor_lst = list(tensor_group.tensors)
            node = tensor_lst[0][0]
            src_asap_min, snk_alap_max = self.edge_makespans[node]

        return (src_asap_min, snk_alap_max)

    def get_definite_live_span(self, node, out_idx):
        tensor_key = (node, out_idx)
        tensor_group = self.mem_share_info[tensor_key]
        if tensor_group.has_mult_tensors():
            src_alap_min = tensor_group.source_alap_min(self)
            snk_asap_max = tensor_group.sink_asap_max(self)
        else:
            tensor_lst = list(tensor_group.tensors)
            node = tensor_lst[0][0]
            src_alap_min, snk_asap_max = self.edge_makespans[node]

        return (src_alap_min, snk_asap_max)

    def are_connected_by_node(self, node1, out_idx1, node2, out_idx2):
        nodes_of_tensor1 = set()
        for user in node1.users.keys():
            nodes_of_tensor1.add(user)
        for arg in node1.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                nodes_of_tensor1.add(arg)

        nodes_of_tensor2 = set()
        for user in node2.users.keys():
            nodes_of_tensor2.add(user)
        for arg in node2.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                nodes_of_tensor2.add(arg)
        return not nodes_of_tensor1.isdisjoint(nodes_of_tensor2)

    def is_user_before(self, node1, node2): # if users of node1 are all before node2
        pre_nodes2 = self.find_all_prev_nodes(node2)
        if node1 in pre_nodes2:
            user_before_node2 = True
            for user in node1.users:
                if user not in pre_nodes2:
                    user_before_node2 = False
                    break
            if user_before_node2:
                return True

        return False

    def find_all_prev_nodes(self, node):
        if node in self.cache_prev_nodes.keys():
            return self.cache_prev_nodes[node]

        prev_nodes = set()
        if node not in self.node_set:
            return prev_nodes

        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                prev_nodes.add(arg)
                if arg not in self.cache_prev_nodes:
                    self.cache_prev_nodes[arg] = find_all_prev_nodes(arg)
                prev_nodes.update(self.cache_prev_nodes[arg])
            else:
                print(("ignore arg ").format(arg))

        self.cache_prev_nodes[node] = prev_nodes
        return prev_nodes

    def find_all_post_nodes(self, node):
        if node in self.cache_post_nodes.keys():
            return self.cache_post_nodes[node]

        post_nodes = set()
        if node not in self.node_set:
            return post_nodes

        for user in node.users:
            if user not in self.node_set:
                continue
            post_nodes.add(user)
            if user not in self.cache_post_nodes:
                self.cache_post_nodes[user] = find_all_post_nodes(user)
            post_nodes.update(self.cache_post_nodes[user])

        self.cache_post_nodes[node] = post_nodes
        return post_nodes

    def asap(self, node) -> int:
        assert self.node_makespans
        _asap,_ = self.node_makespans[node]
        return _asap

    def alap(self, node) -> int:
        assert self.node_makespans
        _, _alap = self.node_makespans[node]
        return _alap

    def compute_asap_by_depth(self):
        asaps = {}
        for node in self.nodes_to_schedule:
            self._compute_asap_by_depth(node, asaps)
        return asaps

    def _compute_asap_by_depth(self, node, asaps):
        if node in asaps:
            return asaps[node]

        asap = 0
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg not in self.node_set:
                    continue
                arg_asap = self._compute_asap_by_depth(arg, asaps)
                asap = max(arg_asap+1, asap)
        if self.user_sched_constraints and node in self.user_sched_constraints:
            if asap > self.user_sched_constraints[node]:
                raise ValueError(
                    "Infeasible user schedule constraint for node %s: user specified at %d, min feasible %d"
                    % (node.name, self.user_sched_constraints[node], asap))
            asap = self.user_sched_constraints[node]

        asaps[node] = asap
        return asap

    def compute_alap_by_depth(self):
        alaps = {}
        for node in self.nodes_to_schedule:
            self._compute_alap_by_depth(node, alaps)
        return alaps

    def _compute_alap_by_depth(self, node, alaps):
        if node in alaps:
            return alaps[node]

        alap = self.max_depth
        for user in node.users:
            if user not in self.node_set:
                continue
            user_alap = self._compute_alap_by_depth(user, alaps)
            alap = min(user_alap-1, alap)

        if self.user_sched_constraints and node in self.user_sched_constraints:
            if alap < self.user_sched_constraints[node]:
                raise ValueError(
                    "Infeasible user schedule constraint for node %s: user specified at %d, max feasible %d"
                    % (node.name, self.user_sched_constraints[node], alap))
            alap = self.user_sched_constraints[node]

        alaps[node] = alap
        return alap

    def compute_node_makespans_by_depth(self):
        asaps = self.compute_asap_by_depth()
        alaps = self.compute_alap_by_depth()
        for node in self.nodes_to_schedule:
            asap = asaps[node]
            alap = alaps[node]
            self.node_makespans[node] = (asap, alap)
        return self.node_makespans

    def compute_node_makespans_by_cone(self):
        # refresh post cache for post nodes to avoid calling deep incursive call
        for node in reversed(self.nodes_to_schedule):
            self.find_all_post_nodes(node)

        for node in self.nodes_to_schedule:
            prev_node_num = len(self.find_all_prev_nodes(node))
            post_node_num = len(self.find_all_post_nodes(node))

            # compute the asap and alap
            self.node_makespans[node] = (prev_node_num, node_num - post_node_num - 1)
        return self.node_makespans

    def compute_node_makespans(self):
        assert not self.node_makespans
        node_num = len(self.nodes_to_schedule)
        if self.user_sched_constraints:
            for node,step in self.user_sched_constraints.items():
                self.node_makespans[node] = (step, step)

        if self.node_makespans:
            assert len(self.node_makespans) == node_num
            return self.node_makespans

        if self.one_step_one_op:
            return self.compute_node_makespans_by_cone()
        else:
            return self.compute_node_makespans_by_depth()

    def compute_edge_makespans(self):
        # must compute node makespans(asap/alap) before computing edge makespans
        self.compute_node_makespans()

        for node in self.nodes_to_schedule:
            lb, ub = self.node_makespans[node]
            user_ub_max = ub
            for user in node.users:
                if user not in self.node_set:
                    continue
                _, user_ub = self.node_makespans[user]
                user_ub_max = max(user_ub_max, user_ub)

            # node represents its output edge
            self.edge_makespans[node] = (lb, user_ub_max)

        return self.edge_makespans

class MemoryScheduler:
    """
    Memory scheduler is inspired by [MODel](https://proceedings.mlr.press/v202/steiner23a/steiner23a.pdf).
    We build the python-mip model on fx graph IR. A customized memory allocator
    will be created which follow the memory allocation plan generated by memory
    scheduler.
    """
    def __init__(
        self,
        fx_module,      # torch.fx.GraphModule
        graph_mem_info, # GraphMemInfo
        timeout_s=None,
        rel_stop=0.01,
        abs_stop=8,
    ):
        self.fx_module = fx_module
        self.graph_mem_info = graph_mem_info
        self.timeout = timeout_s
        self.rel_stop = rel_stop
        self.abs_stop = abs_stop

        self.nodes_to_schedule = []
        self.args = []
        self.outputs = []

        self.num_steps = 0
        self.gcd = 1

        for node in fx_module.graph.nodes:
            if node.op == 'placeholder' or node.op == 'get_attr':
                self.args.append(node)
            elif node.op == 'output':
                self.outputs.append(node)
            else:
                self.nodes_to_schedule.append(node)

        self.node_set = set(self.nodes_to_schedule)

        # try run fx graph to determine output tensor size of each node
        # key: node, value: (mem size, mem alloc index)
        self.node_out_mem = {}

    def total_tensor_size(self, align_scale: int):
        total_size = 0
        total_align_scaled_size = 0
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            for var in out_vars:
                re_num = var.size()%align_scale
                if re_num > 0:
                    align_size = var.size()+align_scale-re_num
                else:
                    align_size = var.size()
                total_size += align_size
                total_align_scaled_size += (align_size//align_scale)

        return (total_size, total_align_scaled_size)

    def build_ilp_model(
        self,
        mem_limit=sys.maxsize,
        user_sched_constraints=None,
        one_step_one_op=False,
    ) -> Tuple[Model, OrderedDict]:

        depend_info = DependencyInfo(self.nodes_to_schedule,
                                     self.graph_mem_info,
                                     user_sched_constraints,
                                     one_step_one_op)
        edge_makespans = depend_info.compute_edge_makespans()
        print("asap and alap information:")
        print(depend_info.node_makespans)

        self.num_steps = depend_info.num_steps

        tensor_sizes = set()
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                tensor_sizes.add(out_var.size())

        print(edge_makespans)

        print("origin tensor sizes:")
        print(tensor_sizes)

        align_scale = 1024*128
        print("align_scale: %d" % align_scale)

        align_scaled_sizes = [(ten_size+align_scale-1)//align_scale for ten_size in tensor_sizes]
        print("align scaled sizes:")
        print(align_scaled_sizes)

        self.gcd = np.gcd.reduce(list(align_scaled_sizes))
        self.gcd *= align_scale
        print("gcd: %d" % self.gcd)

        total_size, total_align_scaled_size = self.total_tensor_size(align_scale)
        print("total_size: %d, total_align_scaled_size: %d" % (total_size, total_align_scaled_size))
        max_address = (min(total_size, mem_limit)+self.gcd-1) // self.gcd
        print("max_address: %d" % max_address)

        int_feas_tol = min(1e-5, 1.0 / max_address)
        int_feas_tol = max(1e-9, int_feas_tol)

        model = Model("mem_scheduler")
        model.max_mip_gap = self.rel_stop
        model.max_mip_gap_abs = self.abs_stop
        model.integer_tol = int_feas_tol              # set integer_tol
        model.emphasis = SearchEmphasis.FEASIBILITY   # mean "MIPFocus": 1
        #model.emphasis = SearchEmphasis.OPTIMALITY    # mean "MIPFocus": 2   # Lansong(TODO)
        model.threads = 64

        # Create two variables for each tensor and timestep:
        create_vars = defaultdict(lambda: defaultdict(lambda: {}))
        preserve_vars = defaultdict(lambda: defaultdict(lambda: {}))

        create_vars_per_step = [[] for _ in range(self.num_steps)]
        for node in self.nodes_to_schedule:
            lb, ub = edge_makespans[node]
            assert lb < self.num_steps
            assert ub < self.num_steps
            #print(("node name: {}, node op: {}, span: ({},{})").format(node,node.op,lb,ub))
            out_vars = self.graph_mem_info.get_out_vars(node)
            #print(("out_vars: {}").format(out_vars))
            for step in range(lb, ub+1):
                for idx,out_var in enumerate(out_vars):
                    out_idx = out_var.out_index
                    out_name = node.name + "[" + str(out_idx) + "]"

                    var_name = out_name + "_create_step" + str(step)
                    v = model.add_var(var_type=BINARY, name=var_name)
                    #print(("create/preserve: node: {}, out_idx: {}, step: {}").format(node, out_idx, step))
                    create_vars[node][out_idx][step] = v
                    if one_step_one_op and (idx == 0):
                        create_vars_per_step[step].append(v)

                    var_name = out_name + "_preserve_step" + str(step)
                    v = model.add_var(var_type=BINARY, name=var_name)
                    preserve_vars[node][out_idx][step] = v

            if user_sched_constraints:
                if node in user_sched_constraints:
                    # node is scheduled by user
                    user_step = user_sched_constraints[node]
                    assert user_step >= lb
                    assert user_step <= ub
                    for step in range(lb, ub+1):
                        if step != user_step:
                            for out_var in out_vars:
                                out_idx = out_var.out_index
                                model += create_vars[node][out_idx][step] == 0
                        else:
                            for out_var in out_vars:
                                out_idx = out_var.out_index
                                model += create_vars[node][out_idx][step] == 1
                for snk in node.users:
                    if snk in user_sched_constraints:
                        snk_step = user_sched_constraints[snk]
                        assert snk_step >= lb
                        assert snk_step <= ub
                        for out_var in out_vars:
                            out_idx = out_var.out_index
                            model += preserve_vars[node][out_idx][snk_step] == 1

        # 1. One and only one node is scheduled to each step
        if one_step_one_op:
            for one_step_vars in create_vars_per_step:
                model += xsum(var for var in one_step_vars) == 1

        # 2. A tensor can either be created or preserved at a step, but not both
        for node in self.nodes_to_schedule:
            lb, ub = edge_makespans[node]
            for step in range(lb, ub+1):
                out_vars = self.graph_mem_info.get_out_vars(node)
                for out_var in out_vars:
                    out_idx = out_var.out_index
                    model += preserve_vars[node][out_idx][step] + \
                             create_vars[node][out_idx][step] <= 1

        # 3. A tensor can be preserved at a step if and only if it was created or
        #    preserved at the previous step
        for node in self.nodes_to_schedule:
            lb, ub = edge_makespans[node]
            for step in range(lb+1, ub+1):
                out_vars = self.graph_mem_info.get_out_vars(node)
                for out_var in out_vars:
                    out_idx = out_var.out_index
                    model += preserve_vars[node][out_idx][step] <= \
                             preserve_vars[node][out_idx][step-1] + \
                             create_vars[node][out_idx][step-1]

        # 4. Force every tensor to be created once
        for node in self.nodes_to_schedule:
            lb, ub = edge_makespans[node]
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                out_idx = out_var.out_index
                model += xsum(create_vars[node][out_idx][step] for step in range(lb, ub+1)) == 1

        # 5. all inputs must be present in memory when evaluating a node
        for node in self.nodes_to_schedule:
            node_lb, node_ub = depend_info.node_makespans[node]
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg not in self.node_set:
                        continue
                    for step in range(node_lb, node_ub+1):
                        #print(("node: {}, arg: {}, step: {}").format(node, arg, step))
                        model += create_vars[node][0][step] <= \
                                 preserve_vars[arg][0][step]

        # 6. all outputs of a node must be created at the same time
        for node in self.nodes_to_schedule:
            out_vars = self.graph_mem_info.get_out_vars(node)
            if len(out_vars) > 1:
                node_lb, node_ub = depend_info.node_makespans[node]
                for var_idx in range(1, len(out_vars)):
                    out_idx = out_vars[var_idx].out_index
                    for step in range(node_lb, node_ub+1):
                        model += create_vars[node][0][step] == \
                                 create_vars[node][out_idx][step]

        # Auxiliary constraints which are not MUST but helpful to speedup solver
        #   Auxiliary constraint 1:
        for node in self.nodes_to_schedule:
            lb, _ = edge_makespans[node]
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                out_idx = out_var.out_index
                model += preserve_vars[node][out_idx][lb] == 0

        #   Auxiliary constraint 2:
        for node in self.nodes_to_schedule:
            lb, ub = edge_makespans[node]
            _, src_node_alap = depend_info.node_makespans[node]
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                out_idx = out_var.out_index
                for step in range(lb, ub+1):
                    if step > src_node_alap:
                        model += create_vars[node][out_idx][step] == 0

        # Do not need the constraint that all inputs are alive.
        # because following is always true:
        # at least one timestep during which all the inputs are live at the same time

        #   Auxiliary constraint 3:
        for node in self.nodes_to_schedule:
            _, node_ub = depend_info.node_makespans[node]
            latest_preserve_step = node_ub + 1
            last_read_step = 0
            for user in node.users:
                if user not in self.node_set:
                    continue
                user_lb, _ = depend_info.node_makespans[user]
                last_read_step = max(last_read_step, user_lb)
            if latest_preserve_step > last_read_step:
                continue
            edge_lb, edge_ub = edge_makespans[node]
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                out_idx = out_var.out_index
                for step in range(edge_lb, edge_ub+1):
                    if step < latest_preserve_step or step > last_read_step:
                        continue
                    model += preserve_vars[node][out_idx][step] == 1

        # memory usage at each step
        mem_per_step = defaultdict(lambda: 0)
        for step in range(self.num_steps):
            for node in self.nodes_to_schedule:
                out_vars = self.graph_mem_info.get_out_vars(node)
                for out_var in out_vars:
                    out_idx = out_var.out_index
                    out_size = out_var.size()
                    create_step_map = create_vars[node][out_idx]
                    if step in create_step_map.keys():
                        mem_per_step[step] = mem_per_step[step] + create_step_map[step] * ((out_size+self.gcd-1) // self.gcd)
                    preserve_step_map = preserve_vars[node][out_idx]
                    if step in preserve_step_map.keys():
                        mem_per_step[step] = mem_per_step[step] + preserve_step_map[step] * ((out_size+self.gcd-1) // self.gcd)

        #   Auxiliary constraint 4:
        # memory limit constraints
        liveness = defaultdict(lambda: [])
        for node in self.nodes_to_schedule:
            edge_lb, edge_ub = edge_makespans[node]
            out_vars = self.graph_mem_info.get_out_vars(node)
            for out_var in out_vars:
                for step in range(edge_lb, edge_ub+1):
                    liveness[step].append(out_var)

        for step, mem_usage in mem_per_step.items():
            max_mem = 0
            for var in liveness[step]:
                max_mem += var.size()
            if max_mem < mem_limit:
                continue
            model += mem_usage <= (mem_limit+self.gcd-1) // self.gcd

        # generate addresses
        gen_allocation = False
        if user_sched_constraints:
            for node in self.nodes_to_schedule:
                assert node in user_sched_constraints
            gen_allocation = True

        addresses = OrderedDict()
        if gen_allocation:
            # encode tensor locations
            for node in self.nodes_to_schedule:
                out_vars = self.graph_mem_info.get_out_vars(node)
                node_out_addr_vars = [None]*len(out_vars)
                for out_var in out_vars:
                    out_idx = out_var.out_index
                    out_size = out_var.size()
                    assert out_idx < len(out_vars)
                    assert out_size > 0
                    addr_var = model.add_var(var_type=INTEGER,
                                             name="%s[%d]" % (node.name, out_idx),
                                             lb=0,
                                             ub=max_address-((out_size+self.gcd-1)//self.gcd))
                    node_out_addr_vars[out_idx] = addr_var
                addresses[node] = node_out_addr_vars

            processed = set()
            for nd1, span1 in edge_makespans.items():
                for nd2, span2 in edge_makespans.items():
                    if nd1 is nd2:
                        continue
                    if (nd2, nd1) in processed:
                        continue
                    processed.add((nd1, nd2))
                    overlap_live_start = max(span1[0], span2[0])
                    overlap_live_stop = min(span1[1], span2[1])

                    assert ((span1[1] < span2[0] or span1[0] > span2[1]) ==
                           (overlap_live_stop < overlap_live_start))
                    out_vars1 = self.graph_mem_info.get_out_vars(nd1)
                    out_vars2 = self.graph_mem_info.get_out_vars(nd2)
                    for var1 in out_vars1:
                        idx1 = var1.out_index
                        for var2 in out_vars2:
                            idx2 = var2.out_index
                            overlap_type = depend_info.get_overlap_type(
                                                              nd1, idx1, nd2, idx2)
                            if overlap_type == OverlapType.NEVER_OVERLAP:
                                # need not add constraint for two memory addresses
                                continue
                            elif overlap_type == OverlapType.OVERLAP_FORCE_SHARE:
                                model += addresses[nd1][idx1] == addresses[nd2][idx2]
                            elif overlap_type == OverlapType.OVERLAP_NOT_SHARE:
                                # create a binary var to represent var1 memory is
                                # below var2 memory
                                v_name = nd1.name + "[" + str(idx1) + "]_below_" + \
                                         nd2.name + "[" + str(idx2) + "]"
                                v1_below_v2 = model.add_var(var_type=BINARY, name=v_name)
                                model += addresses[nd1][idx1] + ((var1.size()+self.gcd-1)//self.gcd) \
                                         - addresses[nd2][idx2] <= \
                                         (1-v1_below_v2)*max_address
                                model += addresses[nd2][idx2] + ((var2.size()+self.gcd-1)//self.gcd) \
                                         - addresses[nd1][idx1] <= \
                                         v1_below_v2*max_address
                            else:
                                assert overlap_type == OverlapType.POSSIBLE_OVERLAP
                                # It is possible that two tensor's memory using
                                # spans overlap.
                                v_base_name = nd1.name + "[" + str(idx1) + "]_" + \
                                              nd2.name + "[" + str(idx2) + "]"
                                v1_name = v_base_name + "_v1"
                                v1 = model.add_var(var_type=BINARY, name=v1_name)
                                v2_name = v_base_name + "_v2"
                                v2 = model.add_var(var_type=BINARY, name=v2_name)
                                model += v1 + v2 <= 1

                                for step in range(overlap_live_start, overlap_live_stop+1):
                                    live1 = create_vars[nd1][idx1][step] + \
                                            preserve_vars[nd1][idx1][step]
                                    live2 = create_vars[nd2][idx2][step] + \
                                            preserve_vars[nd2][idx2][step]
                                    overlap_at_step = live1 + live2 - 1
                                    model += v1 + v2 >= overlap_at_step

                                model += addresses[nd1][idx1] + ((var1.size()+self.gcd-1)//self.gcd) - \
                                         addresses[nd2][idx2] <= (1-v1)*max_address
                                model += addresses[nd1][idx1] - addresses[nd2][idx2] \
                                         - ((var2.size()+self.gcd-1)//self.gcd) >= (v2-1)*max_address

        # construct optimization objective
        min_peak_mem = 0  # Lansong(TODO)
        peak_mem_usage = model.add_var(var_type=INTEGER, name="peak_mem_usage",
                                       lb=min_peak_mem, ub=max_address)

        if addresses:
            # constraint for peak memory usage
            for nd, addr_lst in addresses.items():
                out_vars = self.graph_mem_info.get_out_vars(nd)
                assert len(out_vars) == len(addr_lst)
                for out_var in out_vars:
                    out_idx = out_var.out_index
                    out_size = out_var.size()
                    addr = addr_lst[out_idx]
                    model += peak_mem_usage >= addr + ((out_size+self.gcd-1)//self.gcd)

        for step, mem_usage in mem_per_step.items():
            model += peak_mem_usage >= mem_usage

        obj = peak_mem_usage

        # set optimization objective
        model.objective = minimize(obj)

        return model, addresses, create_vars, preserve_vars, peak_mem_usage

    def create_min_mem_schedule(self):
        model, addresses, create_vars, preserve_vars, peak_mem_usage = \
                                                        self.build_ilp_model()

        start_time = time.time()
        logger.info("Start ILP solver for minimize memory usage")
        if self.timeout:
            model.optimize(max_seconds=self.timeout)
        else:
            model.optimize()
        logger.info(f"ILP solver time: {time.time()-start_time} seconds")

        print(("model.status: {}").format(model.status))
        print(("model.num_solutions: {}").format(model.num_solutions))
        print(("model.objective_value: {}").format(model.objective_value))
        if model.num_solutions:
            if model.status == OptimizationStatus.OPTIMAL:
                logger.info(f"Optimal solution was found with objective value: {model.objective_value}")
            else:
                assert model.status == OptimizationStatus.FEASIBLE
                logger.info(f"Feasible solution was found with objective value: {model.objective_value}")
        else:
            logger.info("No solution was found")
        
        # extract schedules for return
        schedules = defaultdict(lambda: 0)
        for node in self.nodes_to_schedule:
            if node not in self.node_set:
                continue
            nd_out_create_step = -1
            print(("type of create_vars[node][0]: {}").format(type(create_vars[node][0])))
            print(("len of create_vars[node][0]: {}").format(len(create_vars[node][0])))
            for step,val in create_vars[node][0].items():
                print(("step: {}, val: {}, val.x: {}").format(step, val, val.x))
                if val.x >= 0.99:
                    nd_out_create_step = step
                    break
            assert nd_out_create_step >= 0
            schedules[node] = nd_out_create_step

        steped_nodes = [[] for _ in range(self.num_steps)]
        for nd,step in schedules.items():
            assert step < self.num_steps
            steped_nodes[step].append(nd)

        ordered_schedules = []
        for step, nodes_in_step in enumerate(steped_nodes):
            print("step: %d" % step)
            for nd in nodes_in_step:
                print(("  node: {}").format(nd))
                ordered_schedules.append(nd)
        assert len(ordered_schedules) == len(self.nodes_to_schedule)

        mem_locations = defaultdict(lambda: [])
        for node,addrs in addresses.items():
            mem_addrs = []
            out_vars = self.graph_mem_info.get_out_vars(node)
            for idx,addr in enumerate(addrs):
                mem_size = 0
                if out_vars[idx].out_index == idx:
                    mem_size = out_vars[idx].size()
                else:
                    # in case out_vars are out of order
                    for out_var in out_vars:
                        if out_var.out_index == idx:
                            mem_size = out_var.size()
                            break
                assert mem_size > 0
                mem_addr = int(addr.x + 0.5)*self.gcd
                mem_addrs.append((mem_addr, mem_size))
            mem_locations[node] = mem_addrs

        required_memory = int(peak_mem_usage.x + 0.5)*self.gcd

        # dump peak memory
        logger.info(f"required memory: {required_memory/1024/1024/1024}GB")

        # dump memory addresses
        graph_mem_addr_str = "graph memory addresses:\n"
        for node,mem_addrs in mem_locations.items():
            node_mem_str = node.name + ": "
            for mem_addr in mem_addrs:
                node_mem_str += "([" + str(mem_addr) + "~" + \
                                str(mem_addr+mem_size-1) + "], " + \
                                str(mem_size()) + "), "

            graph_mem_addr_str += node_mem_str + "\n"
        logger.info(graph_mem_addr_str)

        # dump ordered schedules
        ordered_nodes_str = "ordered nodes:\n"
        for idx,nd in enumerate(ordered_schedules):
            assert nd
            ordered_nodes_str += str(idx) + ": " + nd.name + "\n"
        logger.info(ordered_nodes_str)

        return (required_memory, schedules, ordered_schedules, mem_locations)

