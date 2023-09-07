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

import functools
import logging
import time
from typing import Any

import jax
import rich
from jax import core
from jax._src.distributed import global_state
from jax._src.util import safe_map
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import metadist.config as mdconfig
from metadist.autoflow import AutoFlowSolver
from metadist.metashard import metair

from .device_mesh import set_device_mesh, get_device_mesh
from .bridge import jax2md_bridge
from .sharding_interpreter import MDJaxShardingAnn
from .utils import _sharding_ann_env

logger = logging.getLogger(__name__)


INPUT_STRATEGY = None


def to_shape_array(x):
    if isinstance(x, jax.Array) and not jax.core.is_opaque_dtype(x.dtype):
        return core.ShapedArray(shape=x.shape, dtype=x.dtype)
    else:
        return x


def materialize(x):
    if isinstance(x, core.ShapedArray):
        key = jax.random.PRNGKey(seed=42)
        if x.dtype.name in ["float64", "float32", "float16"]:
            return jax.random.normal(key, shape=x.shape, dtype=x.dtype)
        elif x.dtype.name in ["int32", "unint32", "int64", "uint64", "uint8"]:
            return jax.random.randint(key, shape=x.shape, dtype=x.dtype, minval=1, maxval=8)
        elif x.dtype.name in ["bool"]:
            return jax.random.normal(key, shape=x.shape) > 1.
        else:
            return jax.numpy.empty(shape=x.shape, dtype=x.dtype)
    return x


def convert(strategy, mesh, val):
    axis_names = mesh.axis_names

    s1, s2 = strategy[0], strategy[1]
    ndim = len(val.shape)
    mesh_shape = [None] * ndim

    # we use strategy except val.shape -> ()
    if ndim > 0:
        for idx, s_ in enumerate([s1, s2]):
            if s_.state == metair.SPMD.SHARD:
                dim = s_.args["dim"]
                if mesh_shape[dim] == None:
                    mesh_shape[dim] = axis_names[idx]
                else:
                    mesh_shape[dim] = mesh.axis_names

    return NamedSharding(mesh, PartitionSpec(*mesh_shape))


def _get_local_array(array, specs):
    mesh = get_device_mesh()
    mesh_shape = mesh.device_ids.shape
    axis_names = mesh.axis_names
    for idx in range(len(specs)):
        for axis_idx, axis in enumerate(axis_names):
            if specs[idx] == axis:
                global_rank = global_state.process_id
                rank_coords = (mesh.device_ids == global_rank).nonzero()
                mesh_dim_rank = rank_coords[axis_idx].item()
                array = jax.numpy.array_split(array, mesh_shape[axis_idx], axis=idx)[mesh_dim_rank]

    return array


def shard_module(flatten_args):

    device_mesh = get_device_mesh()

    for i in range(len(flatten_args)):
        if isinstance(flatten_args[i], jax.Array) or isinstance(flatten_args[i], jax.ShapedArray):
            strategy = convert(INPUT_STRATEGY[i], device_mesh, flatten_args[i])
            flatten_args[i] = materialize(flatten_args[i])
            local_array = _get_local_array(flatten_args[i], strategy.spec)
            flatten_args[i] = multihost_utils.host_local_array_to_global_array(
                local_array, device_mesh, strategy.spec)

    return flatten_args


def add_sharding_jaxpr(jaxpr, consts, shard_strategy, args):
    env = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Args now correspond to Jaxpr outvars
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Looping backward
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)

        onelevel_key = eqn.primitive.__str__()
        if eqn.primitive.__str__() == "custom_jvp_call":
            onelevel_key += "[" + \
                subfuns[0].f.args[0].eqns[0].params['name'] + "]"
        if eqn.primitive.__str__() == "xla_call":
            onelevel_key += "[" + eqn.params['name'] + "]"

        invars = [var for var in eqn.invars if isinstance(var, jax.core.Var)]

        #  outvars are now invars
        invals = safe_map(read, eqn.invars)

        # NOTE, use last outvars for name of MetaNode in Jax
        unique_key = str(eqn.outvars[-1])

        mesh = get_device_mesh()

        if unique_key in shard_strategy:
            in_strtg_group = shard_strategy[unique_key]['strategy'].in_strtg_group

            sharding_idx = 0
            for idx in range(len(invals)):
                if isinstance(invals[idx], jax.interpreters.partial_eval.DynamicJaxprTracer):
                    strategy_ = convert(in_strtg_group.get_var_strtg(sharding_idx), mesh,
                                        invals[idx])
                    invals[idx] = jax.lax.with_sharding_constraint(invals[idx], strategy_)
                    sharding_idx += 1

        outval = eqn.primitive.bind(*subfuns, *invals, **bind_params)

        if isinstance(outval, jax.interpreters.partial_eval.DynamicJaxprTracer):
            outval = [outval]

        safe_map(write, eqn.outvars, outval)

    outvals = safe_map(read, jaxpr.outvars)

    return outvals


def get_opt_strategy(func, *args, **kwargs):

    global INPUT_STRATEGY

    closed_jaxpr = jax.make_jaxpr(func)(*args, **kwargs)

    # use tf32 during MetaSPMD Annotaion
    with _sharding_ann_env():

        start_t = time.perf_counter()
        sharding_interpreter = MDJaxShardingAnn(closed_jaxpr.jaxpr)
        sharding_info, shape_info = sharding_interpreter.run(closed_jaxpr.literals, *args,
                                                             **kwargs)
        logger.info(f"[MDJaxShardingAnn.run]: {time.perf_counter() - start_t} s.")

        if mdconfig.log_level <= logging.DEBUG and global_state.process_id == 0:
            rich.print("sharding_info:\n", sharding_info)
            rich.print("shape_info:\n", shape_info)

    meta_graph = jax2md_bridge(closed_jaxpr.jaxpr, sharding_info, shape_info)

    if mdconfig.log_level <= logging.DEBUG and global_state.process_id == 0:
        rich.print(meta_graph)

    device_mesh = get_device_mesh()
    solver = AutoFlowSolver(device_mesh.device_ids.shape)
    solver.add_graph(meta_graph)
    start_t = time.perf_counter()
    opt_strategy = solver.ilp_optimize()
    logger.info(f"[AutoFlowSolver.ilp_optimize]: {time.perf_counter() - start_t} s.")
    # start_t = time.perf_counter()
    # beam_search_strategy = solver.beam_search()
    # logger.info(f"[AutoFlowSolver.beam_search]: {time.perf_counter() - start_t} s.")

    INPUT_STRATEGY = meta_graph.get_input_strategy(opt_strategy)

    if mdconfig.log_level <= logging.DEBUG and global_state.process_id == 0:
        rich.print(opt_strategy)

    return opt_strategy


def metadist_shard(fun, shard_strategy={}):

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        # Since we assume unary functions, we won't worry about flattening and
        # unflattening arguments.
        closed_jaxpr, return_shape = jax.make_jaxpr(fun, return_shape=True)(*args, **kwargs)

        flatten_args, in_tree = jax.tree_util.tree_flatten(args)

        out = add_sharding_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, shard_strategy,
                                 flatten_args)

        flatten_shape, shape_tree = jax.tree_util.tree_flatten(return_shape)

        out_idx = 0
        for idx, shape in enumerate(flatten_shape):
            if isinstance(shape, jax.ShapeDtypeStruct):
                flatten_shape[idx] = out[out_idx]
                out_idx += 1

        out_tree = jax.tree_util.tree_unflatten(shape_tree, flatten_shape)

        return out_tree

    return wrapped


# =============== Experimetal API for Jax ==================


def _compile(func, args, kwargs):

    # setup the device mesh
    size = jax.device_count()
    devices = mesh_utils.create_device_mesh((1, size))
    mesh = Mesh(devices, axis_names=('a', 'b'))

    set_device_mesh(mesh)

    opt_strategy = get_opt_strategy(func, *args, **kwargs)

    shard_func = jax.jit(metadist_shard(func, opt_strategy))

    if mdconfig.log_level <= logging.DEBUG and global_state.process_id == 0:
        closed_jaxpr = jax.make_jaxpr(shard_func)(*args, **kwargs)
        print(closed_jaxpr.jaxpr)
        print(closed_jaxpr.literals)

    def compiled_func_return(*args, **kwargs):

        flatten_args, specs = jax.tree_util.tree_flatten((args, kwargs))
        flatten_args = shard_module(flatten_args)
        shard_args, shard_kwargs = jax.tree_util.tree_unflatten(specs, flatten_args)

        with jax.spmd_mode('allow_all'):
            out = shard_func(*shard_args, **shard_kwargs)

        return out

    return compiled_func_return


class CompiledFuncWrapper:

    def __init__(self, func) -> None:
        self.original_func = func
        self.compiled_func = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.compiled_func is None:

            self.compiled_func = _compile(self.original_func, args, kwargs)

        return self.compiled_func(*args, **kwargs)


def metadist_compile(func=None, max_solver_time=float("inf"), liveness_only_input=False):

    mdconfig.liveness_only_input = liveness_only_input
    mdconfig.max_seconds_same_incumbent = max_solver_time

    if func:
        return CompiledFuncWrapper(func)

    else:

        def wrapper(func):
            return CompiledFuncWrapper(func)

        return wrapper
