# torchrun --nproc_per_node 4 tests/test_torch/test_devicemesh/test_dtensor.py
import functools
import os

import torch
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor, DTensor

from easydist.torch.compile_auto import dtensor_to_tensor
from easydist.torch.device_mesh import get_device_mesh, set_device_mesh


if __name__ == "__main__":
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    dim_names = ["spmd0", "spmd1", "pp"]
    size = [2, 2, 1]
    matrix = torch.arange(world_size).reshape(size)

    set_device_mesh(DeviceMesh("cuda", matrix, mesh_dim_names=dim_names))

    spmd_mesh = get_device_mesh('spmd')

    big_tensor = torch.ones(3, 4)
    my_dtensor = distribute_tensor(big_tensor, spmd_mesh, [Replicate(), Replicate()])

    big_tensor2 = torch.ones(4, 3)
    my_dtensor2 = distribute_tensor(big_tensor2, spmd_mesh, [Shard(0), Replicate()])

    ans = my_dtensor @ my_dtensor2
    print(ans.shape, ans._local_tensor.shape, ans._spec)
    exit()
    my_dtensor = distribute_tensor(big_tensor, spmd_mesh, [Shard(0), Replicate()])
    assert my_dtensor._local_tensor.shape == (50000, 88, 78)
    
    my_dtensor = my_dtensor.redistribute(spmd_mesh, [Replicate(), Shard(1)])
    assert my_dtensor._local_tensor.shape == (100000, 44, 78)

    my_dtensor = my_dtensor.redistribute(spmd_mesh, [Shard(0), Shard(1)])
    assert my_dtensor._local_tensor.shape == (50000, 44, 78)

    my_dtensor = my_dtensor.redistribute(spmd_mesh, [Shard(1), Shard(2)])
    assert my_dtensor._local_tensor.shape == (100000, 44, 39)

    my_dtensor = my_dtensor.redistribute(spmd_mesh, [Shard(0), Shard(0)])
    assert my_dtensor._local_tensor.shape == (25000, 88, 78)

    print("Passed!")