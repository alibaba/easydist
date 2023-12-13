from torch.fx.passes.graph_drawer import FxGraphDrawer

__all__ = ['save_graphviz_dot']

def save_graphviz_dot(gm, name):
    with open(f"{name}.dot", "w") as f:
        f.write(str(FxGraphDrawer(gm, name).get_dot_graph()))
