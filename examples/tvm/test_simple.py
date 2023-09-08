import logging

import rich
import tvm
import tvm.testing
from tvm import te
import numpy as np

import easydist as md

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.DEBUG)

md.platform.init_backend("tvm")

tgt = tvm.target.Target(target="llvm", host="llvm")

n = te.var("n")
A = te.placeholder((n, ), name="A")
B = te.placeholder((n, ), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

fadd = tvm.build(s, [A, B, C], tgt, name="myadd")


def fadd_wrapped(a, b):
    c = md.platform.zeros_like(a)
    assert a.shape == b.shape
    fadd(a, b, c)
    return c


dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)

meta_op = md.metashard.MetaOp(fadd_wrapped, ((a, b), {}))
sharding_annotion, combination_ann = meta_op.sharding_discovery()

rich.print(sharding_annotion)
rich.print(combination_ann)

c = fadd_wrapped(a, b)
print(c)
print(a.numpy() + b.numpy())
