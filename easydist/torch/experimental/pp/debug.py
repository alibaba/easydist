import torch


def friendly_debug_info(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad})"
    else:
        return str(v)


def map_debug_info(a):
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
