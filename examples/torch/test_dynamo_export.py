from typing import List

import torch
import torch._dynamo


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    print(gm.graph)
    return gm.forward


model_exp = torch._dynamo.export(foo,
                                 torch.randn(10, 10),
                                 torch.randn(10, 10),
                                 aten_graph=True,
                                 tracing_mode="fake")
model_exp[0].print_readable()
# opt_foo1 = torch.compile(foo, backend=custom_backend, fullgraph=True)
a, b = torch.randn(10, 10), torch.randn(10, 10)
print(foo(a, b))

print(model_exp[0](a, b))
