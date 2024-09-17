import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states

from easydist.utils import rgetattr, rsetattr

from benchmark.torch.model.gpt import GPT


def train_step(input, model, opt):
    out = model(input)
    loss = out.mean()
    loss.backward()
    if opt:
        opt.step()
        opt.zero_grad()
    return out


def fw_bw_step(input, model):
    out = model(input)
    loss = out.mean()
    loss.backward()
    return out


def train_step_chunked(input, model, opt, num_chunks):
    output = []
    for chunk in input.chunk(num_chunks):
        out = fw_bw_step(chunk, model)
        output.append(out)
    output = torch.concat(output)
    opt.step()
    opt.zero_grad()
    return output


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())

    return model


class Foo(GPT):
    def __init__(self):
        super().__init__(
                    depth=4,
                    dim=128,
                    num_heads=4,
                    mlp_ratio=4,
                    attention_dropout=0,
                    dropout=0.,
                    dtype=torch.float32)


def get_module_opt_states(module, opt):
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    named_states = {}
    # assign grad and warm up optimizer
    for name in dict(module.named_parameters()):
        with torch.no_grad():
            rsetattr(module, name + ".grad", torch.zeros_like(rgetattr(module, name).data))

    opt.step()
    opt.zero_grad(True)

    for n, p in params.items():
        if p in opt.state:
            named_states[n] = opt.state[p]  # type: ignore[index]
            # if step in state, reduce one for warmup step.
            if 'step' in named_states[n]:
                named_states[n]['step'] -= 1

    return params, buffers, named_states
