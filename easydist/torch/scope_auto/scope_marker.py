# Copyright (c) 2024, Alibaba Group;
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

from typing import List, Sequence, Optional

import torch
import torch._custom_ops

from torch.nn.utils import stateless
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.experimental.proxy_tensor import make_fx
from easydist.torch.split_utils import list_before_split, list_after_split, tuple_before_split, tuple_after_split, _before_split, _after_split


op_param_type = Sequence[Optional[torch.Tensor]]
op_ret_type = List[torch.Tensor]

@torch._custom_ops.custom_op("easydist::fw_scope_start")
def fw_scope_start(args: op_param_type) -> op_ret_type:
    ...

@torch._custom_ops.impl_abstract("easydist::fw_scope_start")
def fw_scope_start_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.impl("easydist::fw_scope_start")
def fw_scope_start_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.custom_op("easydist::fw_scope_end")
def fw_scope_end(args: op_param_type) -> op_ret_type:
    ...

@torch._custom_ops.impl_abstract("easydist::fw_scope_end")
def fw_scope_end_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.impl("easydist::fw_scope_end")
def fw_scope_end_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.custom_op("easydist::bw_scope_start")
def bw_scope_start(args: op_param_type) -> op_ret_type:
    ...

@torch._custom_ops.impl_abstract("easydist::bw_scope_start")
def bw_scope_start_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.impl("easydist::bw_scope_start")
def bw_scope_start_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.custom_op("easydist::bw_scope_end")
def bw_scope_end(args: op_param_type) -> op_ret_type:
    ...

@torch._custom_ops.impl_abstract("easydist::bw_scope_end")
def bw_scope_end_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args

@torch._custom_ops.impl("easydist::bw_scope_end")
def bw_scope_end_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


class ScopeStartFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):
        tensor_list = list(tensor_tuple)  # custom op requires list

        # NOTE: return must be tensor of tuple of tensors
        return tuple(torch.ops.easydist.fw_scope_start(tensor_list))

    @staticmethod
    def backward(ctx, *grads):
        #print(f"start func's backward: grad type: {type(grads)}, value: {grads}")
        grads = list(grads)
        return tuple(torch.ops.easydist.bw_scope_end(grads))


class ScopeEndFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):
        tensor_list = list(tensor_tuple)  # custom op requires list

        # NOTE: return must be tensor of tuple of tensors
        return tuple(torch.ops.easydist.fw_scope_end(tensor_list))

    @staticmethod
    def backward(ctx, *grads):
        #print(f"end func's backward: grad type: {type(grads)}, value: {grads}")
        grads = list(grads)
        return tuple(torch.ops.easydist.bw_scope_start(grads))

def scope_start_func(tensor_list: op_param_type) -> op_ret_type:
    return ScopeStartFunc.apply(*tensor_list)

def scope_end_func(tensor_list: op_param_type) -> op_ret_type:
    return ScopeEndFunc.apply(*tensor_list)


def get_registered_by_mro(registered, cls_begin: type) -> type:
    for cls in cls_begin.mro():
        if cls in registered:
            return registered[cls]
    raise RuntimeError(f"no registered split function for {cls_begin}")


scope_marker_helper_vars = []

def split_with_helper_var(split_func, ret, helper_var):
    cls_ret = type(ret)
    ctx = {}
    tensor_tuple: List[torch.Tensor] = get_registered_by_mro(_before_split, cls_ret)(ctx, ret)
    tensor_tuple.append(helper_var)
    tensor_tuple_after_split: Tuple[torch.Tensor] = split_func(tensor_tuple)
    tensor_tuple_after_split = list(tensor_tuple_after_split)
    helper_var = tensor_tuple_after_split.pop()
    ret = get_registered_by_mro(_after_split, cls_ret)(ctx, tensor_tuple_after_split)
    return ret, helper_var


def scope_marker(marker_aux_vars: List[torch.Tensor]):
    def wrapper(func):
        def impl(*args):
            helper_var = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
            marker_aux_vars.append(helper_var)

            ctx_start = {}
            #print(f"args value: {args}")
            args, helper_var = split_with_helper_var(scope_start_func, args, helper_var)

            result = func(*args)
            result, helper_var = split_with_helper_var(scope_end_func, result, helper_var)

            return result

        return impl
    return wrapper


marker_aux_vars = []

class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = torch.nn.Linear(10, 50)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(50, 1)


    def forward(self, x):
        x = self.first_layer(x)
        x = self.fc2(x)
        return x

    @scope_marker(marker_aux_vars)
    def first_layer(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    model = SimpleMLP()

    input_tensor = torch.randn(32, 10, requires_grad=True)
    target_tensor = torch.randn(32, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    def train_step(input_tensor, target_tensor):
        model.train()
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        return loss

    loss = train_step(input_tensor, target_tensor)
    print(f"Initial loss: {loss.item()}")

    def forward_backward_pass(input_tensor, target_tensor):
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()

        return loss

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def stateless_func(params, buffers, input_tensor, target_tensor):
        with stateless._reparametrize_module(model, {**params, **buffers}):
            ret = forward_backward_pass(input_tensor, target_tensor)
            return ret

    traced_graph = make_fx(stateless_func)(params, buffers, input_tensor, target_tensor)

    print(traced_graph)
    print(f"python codes\n{traced_graph.code}")

    drawer = FxGraphDrawer(traced_graph, "traced_fx", ignore_getattr=True)
    dot_graphs = drawer.get_all_dot_graphs()
    for name, dot_graph in dot_graphs.items():
        dot_graph.write_jpg(f"./{name}.jpg")
        dot_graph.write_raw(f"./{name}.txt")

    for node in traced_graph.graph.nodes:
        print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Args: {node.args}")

