from typing import List, Union, Tuple, Any, Dict, Callable, Sequence, Optional

import torch
import torch._custom_ops

'''
The valid parameters types are: 
dict_keys([
    <class 'torch.Tensor'>, 
    typing.Optional[torch.Tensor], 
    typing.Sequence[torch.Tensor], 
    typing.Sequence[typing.Optional[torch.Tensor]],
    <class 'int'>, 
    typing.Optional[int], 
    typing.Sequence[int], 
    typing.Optional[typing.Sequence[int]], 
    <class 'float'>, 
    typing.Optional[float], 
    typing.Sequence[float], 
    typing.Optional[typing.Sequence[float]], 
    <class 'bool'>, 
    typing.Optional[bool], 
    typing.Sequence[bool], 
    typing.Optional[typing.Sequence[bool]], 
    <class 'str'>, 
    typing.Optional[str], 
    typing.Union[int, float, bool], 
    typing.Union[int, float, bool, NoneType], 
    typing.Sequence[typing.Union[int, float, bool]],
    <class 'torch.dtype'>, 
    typing.Optional[torch.dtype], 
    <class 'torch.device'>, 
    typing.Optional[torch.device]]
    )
'''
op_param_type = Sequence[Optional[torch.Tensor]]
op_ret_type = List[torch.Tensor]


@torch._custom_ops.custom_op("easydist::fw_bw_split")
def fw_bw_split(args: op_param_type) -> op_ret_type:
    ...


@torch._custom_ops.impl_abstract("easydist::fw_bw_split")
def fw_bw_split_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


@torch._custom_ops.impl("easydist::fw_bw_split")
def fw_bw_split_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class FWBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):  # tensors must be passed as args
        tensor_list = list(tensor_tuple)  # custom op requires list
        return tuple(torch.ops.easydist.fw_bw_split(
            tensor_list))  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, *grads):
        return tuple(torch.ops.easydist.fw_bw_split(grads))


def fw_bw_split_func(tensor_list: op_param_type) -> op_ret_type:
    assert not get_backward_flag() or all(
        t.requires_grad for t in tensor_list
    ), "Only split on tensors that need grad, otherwise backward pass won't be tracked"
    return FWBWSplitFunc.apply(*tensor_list)


class BeforeBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):  # tensors must be passed as args
        return torch.ops.easydist.fw_bw_split(
            [tensor])[0]  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, grad):
        return grad  # return must be tensor of tuple of tensors


def before_bw_split_func(tensor: torch.Tensor) -> torch.Tensor:
    assert not get_backward_flag(
    ) or tensor.requires_grad, "Only split on tensors that need grad, otherwise backward pass won't be tracked"
    ret = BeforeBWSplitFunc.apply(tensor)
    return ret


@torch._custom_ops.custom_op("easydist::step_split")
def step_split(args: op_param_type) -> op_ret_type:
    ...


@torch._custom_ops.impl_abstract("easydist::step_split")
def step_split_impl_abstract(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


@torch._custom_ops.impl("easydist::step_split")
def step_split_impl(args: op_param_type) -> op_ret_type:
    args = [arg.clone() for arg in args]
    return args


# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class StepSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_tuple):  # tensors must be passed as args
        tensor_list = list(tensor_tuple)
        return tuple(torch.ops.easydist.step_split(
            tensor_list))  # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, *grads):
        return grads  # return must be tensor of tuple of tensors


def step_split_func(tensor_list: op_param_type) -> op_ret_type:
    ret = StepSplit.apply(*tensor_list)
    return ret


ANNOTATION_OPS = [
    torch.ops.easydist.fw_bw_split.default,
    torch.ops.easydist.step_split.default
]

__updated_params = None
__backward_flag = False
__step_flag = False


def get_updated_params():
    global __updated_params
    return __updated_params


def set_updated_params(updated_params):
    global __updated_params
    __updated_params = updated_params


def get_backward_flag():
    global __backward_flag
    return __backward_flag


def set_backward_flag(flag):
    global __backward_flag
    __backward_flag = flag


def get_step_flag():
    global __step_flag
    return __step_flag


def set_step_flag(flag):
    global __step_flag
    __step_flag = flag


_before_split: Dict[type, Callable[[Any], Tuple[torch.Tensor]]] = {}
_after_split: Dict[type, Callable[[Tuple[torch.Tensor]], Any]] = {}


def before_split_register(*classes):

    def _register(func: Callable):
        for cls in classes:
            assert cls not in _before_split, f"split function for {cls} already registered"
            _before_split[cls] = func
        return func

    return _register


def after_split_register(*classes):

    def _register(func: Callable):
        for cls in classes:
            assert cls not in _after_split, f"split function for {cls} already registered"
            _after_split[cls] = func
        return func

    return _register


def get_registered_by_mro(registered, cls_begin: type) -> type:
    for cls in cls_begin.mro():
        if cls in registered:
            return registered[cls]
    raise RuntimeError(f"no registered split function for {cls_begin}")


def split(ret):
    cls_ret = type(ret)
    ctx = {}
    tensor_tuple: Tuple[torch.Tensor] = get_registered_by_mro(
        _before_split, cls_ret)(ctx, ret)
    tensor_tuple_after_split: Tuple[torch.Tensor] = fw_bw_split_func(
        tensor_tuple)
    ret = get_registered_by_mro(_after_split,
                                cls_ret)(ctx, tensor_tuple_after_split)
    return ret


@before_split_register(torch.Tensor)
def tensor_before_split(ctx: dict, input: torch.Tensor) -> Tuple[torch.Tensor]:
    return tuple([input])


@after_split_register(torch.Tensor)
def tensor_after_split(ctx: dict, output: Tuple[torch.Tensor]) -> torch.Tensor:
    return output[0]


@before_split_register(list)
def list_before_split(
        ctx: dict, input: List[Union[torch.Tensor,
                                     Any]]) -> Tuple[torch.Tensor]:
    ctx['is_tensor'] = []
    ctx['non_tensor_vals'] = []
    tup = []
    for x in input:
        ctx['is_tensor'].append(isinstance(x, torch.Tensor))
        if ctx['is_tensor'][-1]:
            tup.append(x)
        else:
            ctx['non_tensor_vals'].append(x)

    return tuple(tup)


@after_split_register(list)
def list_after_split(
        ctx: dict,
        output: Tuple[torch.Tensor]) -> List[Union[torch.Tensor, Any]]:
    ret = []
    output = list(output)
    for is_tensor in ctx['is_tensor']:
        if is_tensor:
            ret.append(output.pop(0))
        else:
            ret.append(ctx['non_tensor_vals'].pop(0))
    return ret


@before_split_register(tuple)
def tuple_before_split(
        ctx: dict, input: Tuple[Union[torch.Tensor,
                                      Any]]) -> Tuple[torch.Tensor]:
    return list_before_split(ctx, list(input))


@after_split_register(tuple)
def tuple_after_split(
        ctx: dict,
        output: Tuple[torch.Tensor]) -> Tuple[Union[torch.Tensor, Any]]:
    return tuple(list_after_split(ctx, output))

if __name__ == '__main__':
    a = torch.rand(10, 10, requires_grad=True)
    b = torch.rand(3, 10, 10, requires_grad=True)
    res = fw_bw_split_func([a, b])
    print(res[0].requires_grad, res[1].requires_grad)
    (res[0].mean() + res[1].mean()).backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = before_bw_split_func(a)
    print(res.requires_grad)
    res.mean().backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = step_split_func([a])[0]
    print(res.requires_grad)
    res.mean().backward()