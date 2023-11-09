import os
import logging

import torch

from easydist import easydist_setup, mdconfig
from easydist.torch.api import easydist_compile


def main():
    mdconfig.log_level = logging.INFO
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    randn_x = torch.randn(10, 10, requires_grad=True).cuda()
    randn_y = torch.randn(10, 10, requires_grad=True).cuda()
    torch.distributed.broadcast(randn_x, src=0)
    torch.distributed.broadcast(randn_y, src=0)

    @easydist_compile()
    def foo_func(x, y):
        tanh = torch.tanh(x)
        return torch.mm(torch.exp(tanh), y) + tanh

    torch_out = foo_func.original_func(randn_x, randn_y)
    md_out = foo_func(randn_x, randn_y)

    if not torch.allclose(torch_out, md_out):
        raise RuntimeError("simlpe function test failed!!")

    print("simlpe function example pass.")


if __name__ == '__main__':
    main()
