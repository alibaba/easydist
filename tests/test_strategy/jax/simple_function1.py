# mpirun -np 2 python ./examples/jax/simple_function.py

import logging

import jax
import jax.numpy as jnp

from easydist import easydist_setup, mdconfig
from easydist.jax.api import easydist_compile


@easydist_compile(compile_only=True)
def foo_func(x, y):
    tanh = jnp.tanh(x)
    return jnp.exp(tanh) @ y + tanh


def main():
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="jax", device="cuda")

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    randn_x = jax.random.normal(key, (10, 10))
    randn_y = jax.random.normal(subkey, (10, 10))

    foo_func(randn_x, randn_y)


if __name__ == '__main__':
    main()
