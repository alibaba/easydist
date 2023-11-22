'''
Adapted from https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py
'''
import torch
from torch.autograd import Variable

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"


def get_activations(var, params):
    assert all(isinstance(p, Variable) for p in params.values())
    param_map = {id(v): k for k, v in params.items()}

    prop_values = {}
    seen = set()

    cnt = 0

    def get_var_name(var):
        nonlocal cnt
        if id(var) not in param_map:
            param_map[id(var)] = f"unspecified_{cnt}"
            cnt += 1
        return param_map[id(var)]

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)
        for attr in dir(fn):
            if not attr.startswith(SAVED_PREFIX):
                continue
            val = getattr(fn, attr)
            seen.add(val)
            if torch.is_tensor(val):
                if attr not in prop_values:
                    prop_values[attr] = val
                else:
                    prop_values[get_var_name(val)] = val

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            prop_values[get_var_name(var)] = var

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                prop_values[get_var_name(var)] = var

    def add_base_tensor(var):
        if var in seen:
            return
        seen.add(var)
        prop_values[get_var_name(var)] = var
        if var.grad_fn:
            add_nodes(var.grad_fn)
        if var._is_view():
            add_base_tensor(var._base)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    return seen, prop_values
