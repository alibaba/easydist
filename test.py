import torch
from functorch.compile import aot_function, aot_module, \
    make_boxed_func, ts_compile


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(1024)
        self.linear = torch.nn.Linear(1024, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        norm = self.norm(x)
        linear = self.linear(norm)
        relu = self.relu(linear)
        return relu

model = Foo()
# def fn(a, b, c, d):
#     x = a + b + c + d
#     return x.cos().cos()

def run_func(func, *inputs):
    res = func(*inputs)
    print(res)


def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return make_boxed_func(fx_module.forward)

rand_x = torch.rand(16, 1024)

aot_print_fn = aot_module(model, fw_compiler=compiler_fn,
    bw_compiler=compiler_fn)

run_func(aot_print_fn, rand_x)