from typing import Any, List

from torch.fx._pytree import (_dict_flatten_spec, _list_flatten_spec, register_pytree_flatten_spec)
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils._pytree import TreeSpec

__pipeline_tracer_global = None


def get_pipeline_tracer():
    global __pipeline_tracer_global
    return __pipeline_tracer_global


def set_pipeline_tracer(tracer):
    global __pipeline_tracer_global
    __pipeline_tracer_global = tracer


# TODO @botbw: correct to do this?
# fix runtime error
def _immutable_dict_flatten_spec(d: immutable_dict, spec: TreeSpec) -> List[Any]:
    d = dict(d)
    return [d[k] for k in spec.context]


def _immutable_list_flatten_spec(d: immutable_list, spec: TreeSpec) -> List[Any]:
    d = list(d)
    return [d[i] for i in range(len(spec.children_specs))]


register_pytree_flatten_spec(immutable_dict, _dict_flatten_spec)
register_pytree_flatten_spec(immutable_list, _list_flatten_spec)
