from torch.fx.passes.graph_drawer import FxGraphDrawer


def save_graphviz_dot(gm, name):
    with open(f"{name}.dot", "w") as f:
        f.write(str(FxGraphDrawer(gm, name).get_dot_graph()))

def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, )
