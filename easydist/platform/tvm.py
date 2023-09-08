# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tvm
import numpy
import torch.utils._pytree as pytree


def add(a, b):
    return tvm.nd.array(numpy.add(a.numpy(), b.numpy()))


def equal(a, b):
    return numpy.array_equal(a.numpy(), b.numpy())


def zeros_like(input_):
    return tvm.nd.array(numpy.zeros_like(input_.numpy()))


def min(a, b):
    return tvm.nd.array(numpy.minimum(a.numpy(), b.numpy()))


def max(a, b):
    return tvm.nd.array(numpy.maximum(a.numpy(), b.numpy()))


def allclose(a, b):
    return numpy.allclose(a.numpy(), b.numpy(), rtol=5e-3, atol=5e-03)


def concatenate(tensors, dim=0):
    return tvm.nd.array(numpy.concatenate([t.numpy() for t in tensors], axis=dim))


def chunk(input, chunks, dim=0):
    return [tvm.nd.array(i) for i in numpy.array_split(input.numpy(), chunks, axis=dim)]


def narrow(input, dim, start, length):
    indices = numpy.asarray(range(start, start + length))
    return tvm.nd.array(numpy.take(input.numpy(), indices, axis=dim))


Tensor = tvm.nd.NDArray

tree_flatten = pytree.tree_flatten
tree_unflatten = pytree.tree_unflatten


def clone(input_):
    return tvm.nd.array(numpy.copy(input_.numpy()))


from_numpy = tvm.nd.array
