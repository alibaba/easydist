# Examples

## PyTorch Examples

Please use `torchrun` to launch the pytorch examples. Take `simple_function.py` for example:

```shell
# for single-node environment (2 GPUs for examle)
torchrun --nproc_per_node 2 --master_port 9543 ./torch/test_simple.py

# for multi-node environment (2 nodes for example, 2 GPUs each node):
# Machine1: 
torchrun --nnodes 2 --node_rank 0 \
         --master_addr [Machine1 IP] --master_port 9543 \
         ./torch/test_simple.py
# Machine2: 
torchrun --nnodes 2 --node_rank 1 \
         --master_addr [Machine1 IP] --master_port 9543 \
         ./torch/test_simple.py
```

For more details of `torchrun` please refer [Torch Distributed Elastic](https://pytorch.org/docs/stable/elastic/run.html).


## Jax Examples

Please use `mpirun` to launch the pytorch examples. Take `simple_function.py` for example:

```shell
# for single-node environment (2 GPUs for examle)
mpirun -np 2 python ./examples/jax/simple_function.py
```

For multi-node environments, you may need to read docs [Multi Process in Jax](https://jax.readthedocs.io/en/latest/multi_process.html). Also, the function `metadist_setup_jax` in `metadist/jax/__init__.py` may need to be modified to launch process in a clustered environment such as SLURM.