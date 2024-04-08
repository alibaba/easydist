import torch.fx as fx
import operator
from torch.fx.passes.graph_drawer import FxGraphDrawer

def ordered_gi_users(node: fx.node):
    assert all(user.op == 'call_function' and user.target == operator.getitem for user in node.users), "All users of the node must be getitem"
    ret = [None for _ in range(len(node.users))]
    for user in node.users:
        ret[user.args[1]] = user
    return ret

def save_graphviz_dot(gm, name):
    with open(f"{name}.dot", "w") as f:
        f.write(str(FxGraphDrawer(gm, name).get_dot_graph()))

def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )
