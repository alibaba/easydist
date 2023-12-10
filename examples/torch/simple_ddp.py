import copy
import logging
import os

import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import _sync_module_states
from torch.nn.parallel import DistributedDataParallel as DDP

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile


def broadcast_module(model):
    _sync_module_states(model,
                        _get_default_group(),
                        broadcast_bucket_size=int(250 * 1024 * 1024),
                        src=0,
                        params_and_buffers_to_ignore=set())

    return model


class Foo(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(1024)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x.relu()


def main():

    # setting up easydist and torch.distributed
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # when using cuda_graph, because of the warm-up and cuda graph capture,
    # the result of the first step is equivalent to the original result of the third step
    @easydist_compile(tracing_mode="fake", cuda_graph=False, parallel_mode="ddp")
    def train_step(input, model, opt):
        out = model(input).mean()
        out.backward()
        opt.step()
        opt.zero_grad(True)

        return out

    with torch.device('cuda'):
        model = Foo()
        randn_input = torch.randn(1024, 1024)

        model = broadcast_module(model)
        model_2 = copy.deepcopy(model)

        ddp_model = DDP(model)

        opt = torch.optim.Adam(ddp_model.parameters(), lr=0.001, fused=True, capturable=True)
        opt_2 = torch.optim.Adam(model_2.parameters(), lr=0.001, fused=True, capturable=True)

        torch_step_1_result = train_step.original_func(randn_input, ddp_model, opt)
        torch_step_2_result = train_step.original_func(randn_input, ddp_model, opt)

    md_step_1_result = train_step(randn_input, model_2, opt_2)
    md_step_2_result = train_step(randn_input, model_2, opt_2)

    assert torch.allclose(torch_step_1_result,
                          md_step_1_result), "simple model training test failed."
    assert torch.allclose(torch_step_2_result,
                          md_step_2_result), "simple model training test failed."

    print("simple ddp training example pass.")


if __name__ == "__main__":
    main()
