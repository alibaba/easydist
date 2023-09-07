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

import re
from typing import Any

import jax
from jax._src import source_info_util
from jax._src.util import safe_map
from jax._src.core import ShapedArray

from metadist.metashard import MetaOp, view_propagation
from .device_mesh import device_mesh_world_size


def generate_str(invals, bind_params):
    return str([(val.shape, val.dtype.name) if isinstance(val, jax.Array) else val
                for val in invals]) + str(bind_params)


def inject_view_propagation(sharding_info):

    view_op_list = ["reshape"]

    for op_name in sharding_info:
        if op_name in view_op_list:
            for args in sharding_info[op_name]:
                # example args for view:
                # "[((4, 256, 768), dtype('float32'))]{'new_sizes': (1, 4, 256, 768), 'dimensions': None}"
                re_matches = re.findall(r'\((\d+,|\d+(?:, \d+)*)\)', args)
                input_shape = [int(num) for num in re_matches[0].strip(',').split(',')]
                output_shape = [int(num) for num in re_matches[1].strip(',').split(',')]
                sharding_info[op_name][args] = view_propagation(input_shape, 
                                                                output_shape, 
                                                                world_size=device_mesh_world_size())

    return sharding_info


class MDJaxShardingAnn:

    def __init__(self, jaxpr, use_cache=True, seed=42) -> None:
        self.jaxpr = jaxpr
        self.use_cache = use_cache
        self.shape_info = {}
        self.sharding_info = {}

        self.key = jax.random.PRNGKey(seed)

    def run(self, consts, *args) -> Any:
        # Mapping from variable -> value
        env = {}

        def read(var):
            # Literals are values baked into the Jaxpr
            if type(var) is jax.core.Literal:
                return var.val
            if isinstance(env[var], ShapedArray):
                self.key, subkey = jax.random.split(self.key)
                if env[var].dtype.name in ["float64", "float32", "float16"]:
                    return jax.random.normal(subkey, shape=env[var].shape, dtype=env[var].dtype)
                elif env[var].dtype.name in ["int32", "unint32", "int64", "uint64", "uint8"]:
                    return jax.random.randint(subkey,
                                              shape=env[var].shape,
                                              dtype=env[var].dtype,
                                              minval=1,
                                              maxval=8)
                elif env[var].dtype.name in ["bool"]:
                    return jax.random.normal(subkey, shape=env[var].shape) > 1.
                else:
                    return jax.numpy.empty(shape=env[var].shape, dtype=env[var].dtype)
            return env[var]

        def write(var, val):
            if isinstance(val, int):
                env[var] = ShapedArray(shape=set(), dtype=jax.numpy.int32)
            elif isinstance(val, jax.Array) and not jax.core.is_opaque_dtype(val.dtype):
                env[var] = ShapedArray(shape=val.shape, dtype=val.dtype)
            else:
                env[var] = val

        def record_shape(var):
            if isinstance(env[var], jax.Array) or isinstance(env[var], ShapedArray):
                self.shape_info[var.__str__()] = {
                    'shape': env[var].shape,
                    'dtype': env[var].dtype.name
                }

        flat_args, _ = jax.tree_util.tree_flatten(args)

        # Bind args and consts to environment
        safe_map(write, self.jaxpr.invars, flat_args)
        safe_map(write, self.jaxpr.constvars, consts)

        safe_map(record_shape, self.jaxpr.invars)
        safe_map(record_shape, self.jaxpr.constvars)

        # add sharding_info for invars+constvars
        for invar in self.jaxpr.invars + self.jaxpr.constvars:
            invals = env[invar]
            onelevel_key = "placeholder"
            twolevel_key = str([(invals.shape, invals.dtype.name)])
            if onelevel_key not in self.sharding_info:
                self.sharding_info[onelevel_key] = {}
            self.sharding_info[onelevel_key][twolevel_key] = view_propagation(invals.shape, 
                                                                              invals.shape, 
                                                                              world_size=device_mesh_world_size())

        # Loop through equations and evaluate primitives using `bind`
        for eqn in self.jaxpr.eqns:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            onelevel_key = eqn.primitive.__str__()
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack

            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):

                # Read inputs to equation from environment
                invals = safe_map(read, eqn.invars)

                if eqn.primitive.__str__() == "custom_jvp_call":
                    pjit_eqn = subfuns[0].f.args[0].eqns[0]
                    onelevel_key += "[" + pjit_eqn.params['name'] + "]"
                elif eqn.primitive.__str__() in ["xla_call", "pjit"]:
                    onelevel_key += "[" + eqn.params['name'] + "]"

                # `bind` is how a primitive is called
                meta_op_ = MetaOp(func=eqn.primitive.bind,
                                  input_args=(subfuns, invals, bind_params),
                                  name=onelevel_key)
                outvals = meta_op_.exec()

            if onelevel_key not in self.sharding_info:
                self.sharding_info[onelevel_key] = {}

            twolevel_key = generate_str(invals, bind_params)

            if not self.use_cache or twolevel_key not in self.sharding_info[onelevel_key]:
                prompt_annotation = None
                if len(self.sharding_info[onelevel_key]) >= 1:
                    prompt_annotation = list(
                        self.sharding_info[onelevel_key].values())[0]["sharding_ann"]
                sharding_ann, combination_ann = meta_op_.sharding_discovery(
                    prompt_annotation=prompt_annotation)
                self.sharding_info[onelevel_key][twolevel_key] = {
                    "sharding_ann": sharding_ann,
                    "combination_ann": combination_ann
                }

            # outvals = eqn.primitive.bind(*invals, **eqn.params)
            # Primitives may return multiple outputs or not
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            # Write the results of the primitive into the environment
            safe_map(write, eqn.outvars, outvals)
            safe_map(record_shape, eqn.outvars)

        # Read the final result of the Jaxpr from the environment
        # return safe_map(read, self.jaxpr.outvars)
        return inject_view_propagation(self.sharding_info), self.shape_info
