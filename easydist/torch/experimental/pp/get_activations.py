'''
Adapted from https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Dict

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"


def get_activations(stage_output, outputs_with_grads_idxs, params: Dict[str, nn.Parameter]=None, buffers: Dict[str, torch.Tensor]=None):
    name_map = {}
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        name_map.update({id(v): k for k, v in params.items()})
    if buffers is not None:
        assert all(isinstance(p, Variable) for p in buffers.values())
        name_map.update({id(v): k for k, v in buffers.items()})

    activations = {}
    visited = set()

    unspecified_cnt = 0
    def get_var_name(var, name=None):
        nonlocal unspecified_cnt
        if id(var) not in name_map:
            if name is None:
                name_map[id(var)] = f"unspecified_{unspecified_cnt}"
                unspecified_cnt += 1
            else:
                name_map[id(var)] = name
        return name_map[id(var)]

    def add_nodes(fn):
        assert not torch.is_tensor(fn)

        if fn in visited:
            return
        visited.add(fn)
        
        fn_name = type(fn).__name__

        for attr in dir(fn):
            if not attr.startswith(SAVED_PREFIX):
                continue
            val = getattr(fn, attr)
            visited.add(val)
            if torch.is_tensor(val):
                activations[get_var_name(val, fn_name+'.'+attr)] = val

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            visited.add(var)
            activations[get_var_name(var)] = var

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                visited.add(t)
                activations[get_var_name(var)] = var

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    add_nodes(u[0])

    def add_base_tensor(var):
        if var in visited:
            return
        visited.add(var)
        activations[get_var_name(var)] = var
        if var.grad_fn:
            add_nodes(var.grad_fn)
        if var._is_view():
            add_base_tensor(var._base)

    # handle multiple outputs
    assert isinstance(stage_output, (tuple, list))
    for i in outputs_with_grads_idxs:
        v = stage_output[i]
        name_map[id(v)] = f'stage_output_{i}'
        add_base_tensor(v)

    return {k: v for k, v in activations.items() if k not in params and k not in buffers and "stage_output" not in k}
