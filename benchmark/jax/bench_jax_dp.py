# python ./benchmark/bench_jax_dp.py

import os
import sys
import logging
from functools import partial

import jax

from easydist import easydist_setup
from easydist.utils.timer import EDTimer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark.bench_case import GPTCase, ResNetCase
from benchmark.jax.model.gpt import GPTSimple
from benchmark.jax.model.wresnet import resnet18


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)


def get_gpt_case():
    case = GPTCase()
    model = GPTSimple(case)

    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key = jax.random.split(key=root_key)
    input_ = jax.random.normal(
        main_key, (case.batch_size, case.seq_size, case.hidden_dim))  # Dummy input data
    variables = model.init(params_key, input_, deterministic=True)
    params = variables['params']

    @partial(jax.pmap, axis_name="batch")
    def train_step(params, input_):
        lr = 0.0001

        def loss_fn(params):
            dropout_key = jax.random.PRNGKey(seed=0)
            return model.apply({
                'params': params
            },
                               input_,
                               deterministic=False,
                               rngs={
                                   'dropout': dropout_key
                               }).mean()

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        params = jax.tree_map(lambda x, y: x - lr * y, params, grads)
        return params

    devices = jax.local_devices()
    params = jax.device_put_replicated(params, devices)

    def shard_batch(x):
        x = x.reshape((len(devices), -1) + x.shape[1:])
        return jax.device_put_sharded(list(x), devices)

    input_ = jax.tree_map(shard_batch, input_)

    return train_step, [params, input_]


def get_resnet_case():
    case = ResNetCase()
    model = resnet18()

    key1, key2 = jax.random.split(jax.random.PRNGKey(0), num=2)
    input_ = jax.random.normal(key1, (case.batch_size, 224, 224, 3))  # Dummy input data
    variables = model.init(key2, input_)  # Initialization call
    params, batch_stats = variables['params'], variables['batch_stats']

    @partial(jax.pmap, axis_name="batch")
    def train_step(params, batch_stats, input_):
        lr = 0.0001

        def loss_fn(params, batch_stats):
            out_, batch_stats = model.apply({
                'params': params,
                'batch_stats': batch_stats
            },
                                            input_,
                                            mutable=['batch_stats'])
            return out_.mean()

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, batch_stats)
        grads = jax.lax.pmean(grads, axis_name="batch")
        params = jax.tree_map(lambda x, y: x - lr * y, params, grads)
        return params

    devices = jax.local_devices()
    params = jax.device_put_replicated(params, devices)
    batch_stats = jax.device_put_replicated(batch_stats, devices)

    def shard_batch(x):
        x = x.reshape((len(devices), -1) + x.shape[1:])
        return jax.device_put_sharded(list(x), devices)

    input_ = jax.tree_map(shard_batch, input_)

    return train_step, [params, batch_stats, input_]


def bench_pmap_dp(func, args):

    def train_step():
        func(*args)

    timer = EDTimer(train_step, in_ms=False)

    elaps_time = timer.time()

    print(f"Time: {elaps_time}")


def main():
    # setup easydist
    easydist_setup(backend="jax", device="cuda")

    print(jax.devices())

    func, args = get_gpt_case()

    bench_pmap_dp(func, args)


if __name__ == '__main__':
    main()
