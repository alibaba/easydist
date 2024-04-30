import os
import time

import torch
import torch.distributed as dist
from torch.distributed import batch_isend_irecv

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
assert world_size == 3
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(rank)

with torch.device('cuda'):
    if rank == 0:
        send_tensor = torch.zeros(2, dtype=torch.float32) + 111111
        send_op1 = dist.P2POp(dist.isend, send_tensor, 1)
        send_op2 = dist.P2POp(dist.isend, send_tensor, 2)
        reqs = batch_isend_irecv([send_op1, send_op2])
        for req in reqs:
            req.wait()
        print(f"rank 0 sent")
    elif rank == 1:
        recv_tensor = torch.zeros(2, dtype=torch.float32)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
        reqs = batch_isend_irecv([recv_op])
        for req in reqs:
            req.wait()
        # print(f"rank 1 recv_tensor: {recv_tensor}")
        # compute
        time.sleep(1)
        send_tensor = torch.zeros(2, dtype=torch.float32) + 222222
        send_op = dist.P2POp(dist.isend, send_tensor, 2)
        reqs = batch_isend_irecv([send_op])
        for req in reqs:
            req.wait()
        print(f"rank 1 sent")
    else:
        recv_tensor1 = torch.zeros(2, dtype=torch.float32)
        recv_tensor2 = torch.zeros(2, dtype=torch.float32)
        recv_op1 = dist.P2POp(dist.irecv, recv_tensor1, 0)
        recv_op2 = dist.P2POp(dist.irecv, recv_tensor2, 1)
        reqs = batch_isend_irecv([recv_op1, recv_op2])
        for req in reqs:
            req.wait()
        print(f"rank 2 recv_tensor1: {recv_tensor1}")
    