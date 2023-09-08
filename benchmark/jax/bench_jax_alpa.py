# python ./benchmark/bench_jax_alpa.py

import logging
import os
import sys

import alpa
import jax

os.environ["EASYDIST_DEVICE"] = "cuda"
os.environ["EASYDIST_BACKEND"] = "jax"

from easydist.utils.timer import EDTimer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark.jax.model.gpt import GPTSimple
from benchmark.jax.model.wresnet import resnet18
from benchmark.jax.model.gat import GATLayer
from benchmark.bench_case import GPTCase, ResNetCase, GATCase

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.DEBUG)


def get_gpt_case():
    case = GPTCase()
    model = GPTSimple(case)

    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key = jax.random.split(key=root_key)
    input_ = jax.random.normal(
        main_key, (case.batch_size, case.seq_size, case.hidden_dim))  # Dummy input data
    variables = model.init(params_key, input_, deterministic=True)
    params = variables['params']

    # DataParallel() Zero3Parallel() Zero2Parallel()
    @alpa.parallelize(method=alpa.ShardParallel())
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
        params = jax.tree_map(lambda x, y: x - lr * y, params, grads)
        return params

    return train_step, [params, input_]


def get_resnet_case():
    case = ResNetCase()
    model = resnet18()

    key1, key2 = jax.random.split(jax.random.PRNGKey(0), num=2)
    input_ = jax.random.normal(key1, (case.batch_size, 224, 224, 3))  # Dummy input data
    variables = model.init(key2, input_)  # Initialization call
    params, batch_stats = variables['params'], variables['batch_stats']

    @alpa.parallelize(method=alpa.ShardParallel())
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
        params = jax.tree_map(lambda x, y: x - lr * y, params, grads)
        return params

    return train_step, [params, batch_stats, input_]


def get_gat_case():

    case = GATCase()
    model = GATLayer(case.in_feature, case.out_feature)

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), num=3)
    h = jax.random.normal(key1, (case.num_node, case.in_feature))  # Dummy input data
    adj = jax.random.normal(key2, (case.num_node, case.num_node))  # Dummy input data
    variables = model.init(key3, h, adj)  # Initialization call
    params = variables['params']

    @alpa.parallelize(method=alpa.ShardParallel())
    def train_step(params, h, adj):
        lr = 0.0001

        def loss_fn(params):
            return model.apply({'params': params}, h, adj).mean()

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        params = jax.tree_map(lambda x, y: x - lr * y, params, grads)
        return params

    return train_step, [params, h, adj]


def bench_alpa(func, args):

    def train_step():
        func(*args)

    timer = EDTimer(train_step, in_ms=False)

    elaps_time = timer.time()

    print(f"Time: {elaps_time}")


def main():
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    print(jax.devices())

    func, args = get_gat_case()

    bench_alpa(func, args)


if __name__ == '__main__':
    main()
