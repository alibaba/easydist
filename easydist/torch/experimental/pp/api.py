from typing import Any

import torch
import torch.utils._pytree as pytree
import torch.distributed as dist

from easydist.torch.compile import ed_compile_func
from easydist.torch.experimental.pp.compile_pipeline import compile_pipeline
from easydist.torch.experimental.pp.microbatch import split_args_kwargs_into_chunks
from easydist.torch.experimental.pp.runtime import PipelineStage, ScheduleGPipe
from easydist.torch.experimental.pp.utils import save_graphviz_dot

def _compile_pp(func,
                tracing_mode,
                init_helper,
                input_signature,
                args,
                kwargs,
                schedule_cls=ScheduleGPipe,
                args_chunk_spec=None,
                kwargs_chunk_spec=None,
                outputs_chunk_spec=None,
                num_chunks=1,
                return_to_all_stages=True,
                accumulate_grads_inplace=True,
                strict=True,
                local=False,
                local_pp_stage_cnt=None) -> PipelineStage:
    args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, num_chunks,
                                                             args_chunk_spec, kwargs_chunk_spec)
    args, kwargs = args_split[0], kwargs_split[0]

    if local_pp_stage_cnt is not None:
        world_size = local_pp_stage_cnt
        rank = 0
    else:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    module, opt = None, None

    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, torch.nn.Module):
            assert module is None, "Only support single nn.Module in args now"
            module = arg
        if isinstance(arg, torch.optim.Optimizer):
            assert opt is None, "Only support single Optimizer in args now"
            opt = arg

    params, buffers, named_states, _, traced_stateless_func = ed_compile_func(func, tracing_mode, init_helper, args, kwargs, schedule_cls, module, opt)

    traced_stateless_func_node_metas = {
        node.name: node.meta
        for node in traced_stateless_func.graph.nodes
    }
    stateless_func_args = [params, buffers, named_states, args, kwargs]

    if local:
        if num_chunks != 1:
            raise RuntimeError(f"Num of chunks should be one to run pp locally, notice that local pp is only for debugging purposes")

        compiled_meta, compiled_stages, local_gm, _ = compile_pipeline(traced_stateless_func,
                                                                       local_pp_stage_cnt,
                                                                       stateless_func_args,
                                                                       strict=strict)

        class EDCompiledFunc:

            def __init__(self, graph) -> None:
                self.graph = graph

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                outputs = self.graph(*args, **kwargs)
                ret = pytree.tree_unflatten(outputs, compiled_meta.out_spec)[-1]
                return ret

            def run_with_graph(self, graph, *args: Any, **kwargs: Any) -> Any:
                outputs = self.graph(*args, **kwargs)
                ret = pytree.tree_unflatten(outputs, compiled_meta.out_spec)[-1]
                return ret

        return EDCompiledFunc(local_gm)

    compiled_meta, compiled_stages, local_gm, _ = compile_pipeline(traced_stateless_func,
                                                                   world_size,
                                                                   stateless_func_args,
                                                                   strict=strict)

    pipe = PipelineStage(schedule_cls=schedule_cls,
                         local_gm=local_gm,
                         compiled_meta=compiled_meta,
                         stage_idx=rank,
                         compiled_stage=compiled_stages[rank],
                         node_metas=traced_stateless_func_node_metas,
                         num_chunks=num_chunks,
                         args_chunk_spec=args_chunk_spec,
                         kwargs_chunk_spec=kwargs_chunk_spec,
                         returns_chunk_spec=outputs_chunk_spec,
                         pp_group=dist.GroupMember.WORLD,
                         device=torch.device(f"cuda:{rank}"),
                         sharded_graph=traced_stateless_func,
                         return_to_all_stages=return_to_all_stages,
                         accumulate_grads_inplace=accumulate_grads_inplace)

    return pipe
