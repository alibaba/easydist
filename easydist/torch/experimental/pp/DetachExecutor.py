import torch

from easydist.torch.experimental.pp.backward import stage_backward


def run_local_split_gm(split_gm, *args, **kwargs):
    executor = DetachExecutor(split_gm)
    executor_args = args
    # if len(kwargs) > 0:
    #     from inspect import Parameter, Signature

    # parameters = []
    # for node in split_gm.graph.nodes:
    #     if node.op == "placeholder":
    #         if node.args and len(node.args) > 0:
    #             parameters.append(
    #                 Parameter(
    #                     node.target,
    #                     Parameter.POSITIONAL_OR_KEYWORD,
    #                     default=node.args[0],
    #                 )
    #             )
    #         else:
    #             parameter_kind = Parameter.POSITIONAL_OR_KEYWORD
    #             param_name = node.target
    #             if node.target.startswith("**"):
    #                 parameter_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
    #                 param_name = param_name[2:]
    #             elif node.target.startswith("*"):
    #                 parameter_kind = Parameter.VAR_POSITIONAL  # type: ignore[assignment]
    #                 param_name = param_name[1:]
    #             parameters.append(Parameter(param_name, parameter_kind))
    # signature = Signature(parameters)
    # ba = signature.bind(*args, **kwargs)
    # ba.apply_defaults()
    # executor_args = ba.arguments.values()  # type: ignore[assignment]

    return executor.run(*executor_args)


class DetachExecutor(torch.fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """

    def __init__(self, module, garbage_collect_values=True):
        garbage_collect_values = False
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}

    def run(self, *args, initial_env=None):
        self.value_remap = {}
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):

        def detach_tensors(a):
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a

        args = torch.fx.node.map_aggregate(args, detach_tensors)
        kwargs = torch.fx.node.map_aggregate(kwargs, detach_tensors)

        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        # HACK to reroute saved input tensors to point to the detach()ed version
        if target == stage_backward:
            kwargs = dict(kwargs)
            kwargs["input_values"] = [self.value_remap.get(v, v) for v in kwargs["input_values"]]
        return super().call_function(target, args, kwargs)
