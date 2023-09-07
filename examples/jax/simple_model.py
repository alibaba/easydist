# mpirun -np 2 python ./examples/jax/simple_model.py --mode inference

import argparse
import logging

import jax
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random

from metadist import metadist_setup, mdconfig
from metadist.jax.api import metadist_compile


class Foo(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=6)(x)
        x = nn.relu(x)
        return x


def inference_example(module, params, input):

    @metadist_compile()
    def inference_step(params, input):
        out = module.apply({'params': params}, input)
        return out

    jax_out = inference_step.original_func(params, input)
    md_out = inference_step(params, input)

    if not jax.numpy.allclose(jax_out, md_out):
        raise RuntimeError("simlpe model test failed!!")

    print("simlpe model inference example pass.")


def train_example(module, params, input):

    tx = optax.adam(learning_rate=0.01)

    state = train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)

    @metadist_compile()
    def train_step(state, batch):
        """Train for a single step."""

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch)
            loss = logits.mean()
            return loss

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    jax_state = train_step.original_func(state, input)
    md_state = train_step(state, input)

    if not jax.tree_util.tree_all(jax.tree_map(jax.numpy.allclose, jax_state, md_state)):
        raise RuntimeError("simlpe model train example failed!!")

    print("simlpe model train example pass.")


def main():
    parser = argparse.ArgumentParser(description="Simple example of parallelize model.")

    parser.add_argument("--mode",
                        type=str,
                        default=None,
                        choices=["train", "inference"],
                        required=True)

    args = parser.parse_args()

    mdconfig.log_level = logging.INFO
    metadist_setup(backend="jax", device="cuda")

    model = Foo()

    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key = jax.random.split(key=root_key, num=2)
    rand_input = random.normal(main_key, (4, 6))
    variables = model.init(params_key, rand_input)
    params = variables['params']

    if args.mode == "train":
        train_example(model, params, rand_input)
    if args.mode == "inference":
        inference_example(model, params, rand_input)


if __name__ == "__main__":
    main()
