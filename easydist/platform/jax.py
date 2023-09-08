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

from functools import partial

import jax

add = jax.numpy.add
equal = jax.numpy.array_equal
zeros_like = jax.numpy.zeros_like
min = jax.numpy.minimum
max = jax.numpy.maximum
allclose = partial(jax.numpy.allclose, rtol=5e-3, atol=5e-03)


def concatenate(tensors, dim=0):
    return jax.numpy.concatenate(tensors, axis=dim)


def chunk(input, chunks, dim=0):
    return jax.numpy.array_split(input, chunks, axis=dim)


def narrow(input, dim, start, length):
    indices = jax.numpy.asarray(range(start, start + length))
    return jax.numpy.take(input, indices, axis=dim)


Tensor = jax.Array

tree_flatten = jax.tree_util.tree_flatten


def tree_unflatten(values, spec):
    return jax.tree_util.tree_unflatten(spec, values)


clone = jax.numpy.copy

from_numpy = jax.numpy.array
