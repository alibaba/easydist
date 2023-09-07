# mpirun -np 2 python ./examples/jax/simple_function.py

import logging

import jax
import jax.numpy as jnp

from metadist import metadist_setup, mdconfig
from metadist.jax.api import metadist_compile


@metadist_compile()
def foo_func(x, y):
    tanh = jnp.tanh(x)
    return jnp.exp(tanh) @ y + tanh


def main():
    mdconfig.log_level = logging.INFO
    metadist_setup(backend="jax", device="cuda")

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    randn_x = jax.random.normal(key, (10, 10))
    randn_y = jax.random.normal(subkey, (10, 10))

    jax_out = foo_func.original_func(randn_x, randn_y)
    md_out = foo_func(randn_x, randn_y)

    if not jax.numpy.allclose(jax_out, md_out):
        raise RuntimeError("simlpe function test failed!!")

    print("simlpe function example pass.")


if __name__ == '__main__':
    main()
