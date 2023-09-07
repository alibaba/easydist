## Benchmark

```shell
torchrun --nproc_per_node 2 --master_port 26543 ./benchmark/torch/bench_torch.py
torchrun --nproc_per_node 2 --master_port 26543 ./benchmark/torch/bench_torch_tp.py
```
