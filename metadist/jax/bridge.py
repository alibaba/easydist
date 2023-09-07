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

import jax

from metadist.metashard.metair import MetaGraph, MetaNode, MetaVar


def jax2md_bridge(jaxpr: jax.core.Jaxpr, sharding_info, meta_info) -> MetaGraph:

    meta_graph = MetaGraph(jaxpr)

    meta_var_map = {}

    for in_var in jaxpr.invars + jaxpr.constvars:
        meta_var = MetaVar(name=in_var.__str__(),
                           shape=meta_info[in_var.__str__()]["shape"],
                           dtype=meta_info[in_var.__str__()]["dtype"])
        meta_var_map[in_var.__str__()] = meta_var

        node_sharding_info = None
        onelevel_key = "placeholder"
        twolevel_key = str([(meta_var.shape, meta_var.dtype)])
        if onelevel_key in sharding_info:
            node_sharding_info = sharding_info[onelevel_key].get(twolevel_key, None)

        meta_node = MetaNode(name=in_var.__str__(),
                             op_name="placeholder",
                             invars=[],
                             outvars=[meta_var],
                             sharding_info=node_sharding_info,
                             is_placeholder=True)
        meta_graph.add_node(meta_node)
        meta_graph.add_input(meta_node)

    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        onelevel_key = eqn.primitive.__str__()
        if eqn.primitive.__str__() == "custom_jvp_call":
            onelevel_key += "[" + subfuns[0].f.args[0].eqns[0].params['name'] + "]"
        if eqn.primitive.__str__() in ["xla_call", "pjit"]:
            onelevel_key += "[" + eqn.params['name'] + "]"

        node_sharding_info = None
        if onelevel_key in sharding_info:
            abstract_list = []
            for var in eqn.invars:
                if var.__str__() in meta_info:
                    abstract_list.append(
                        (meta_info[var.__str__()]["shape"], meta_info[var.__str__()]["dtype"]))
                elif type(var) is jax.core.Literal:
                    abstract_list.append(var.val)
                else:
                    abstract_list.append(var)
            twolevel_key = str(abstract_list) + str(bind_params)
            if twolevel_key in sharding_info[onelevel_key]:
                node_sharding_info = sharding_info[onelevel_key][twolevel_key]

        outvars_ = []
        for var in eqn.outvars:
            name = var.__str__()
            outvar = MetaVar(name=name,
                             shape=meta_info[name]["shape"],
                             dtype=meta_info[name]["dtype"])
            outvars_.append(outvar)
            meta_var_map[name] = outvar

        # NOTE, use last outvars for name of MetaNode in Jax
        meta_node = MetaNode(name=name,
                             op_name=onelevel_key,
                             invars=[
                                 meta_var_map[var.__str__()] for var in eqn.invars
                                 if isinstance(var, jax.core.Var)
                             ],
                             outvars=outvars_,
                             sharding_info=node_sharding_info)
        meta_graph.add_node(meta_node)

    for var in jaxpr.outvars:
        meta_graph.add_output(meta_var_map[var.__str__()])

    return meta_graph
