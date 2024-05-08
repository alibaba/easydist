threshold microbatch size
for ((i=1 ; i <= 16 ; i += 1)); do
    torchrun --nproc_per_node 4 benchmark/torch/pp/gpt/speed/easydist_pipeline.py --micro-batch-size $i --num-chunks 1 --schedule gpipe
done

for ((i=1 ; i <= 16 ; i += 1)); do
    torchrun --nproc_per_node 4 benchmark/torch/pp/gpt/speed/easydist_pipeline.py --micro-batch-size $i --num-chunks 1 --schedule dapple
done

for ((i=1 ; i <= 16 ; i += 1)); do
    python benchmark/torch/pp/gpt/speed/torchgpipe_pipeline.py --micro-batch-size $i --num-chunks 1
done

for ((i=1 ; i <= 16 ; i += 1)); do
    python benchmark/torch/pp/gpt/speed/vanilla_torch.py --micro-batch-size $i --num-chunks 1
done


num chunks
for ((i=1; i <= 32 ; i *= 2)); do
    torchrun --nproc_per_node 4 benchmark/torch/pp/gpt/speed/easydist_pipeline.py --dataset-size 5000 --micro-batch-size 16 --num-chunks $i --schedule gpipe
done
torchrun --nproc_per_node 4 benchmark/torch/pp/gpt/speed/easydist_pipeline.py --dataset-size 5000 --micro-batch-size 16 --num-chunks 34 --schedule gpipe

for ((i=1; i <= 64 ; i *= 2)); do
    torchrun --nproc_per_node 4 benchmark/torch/pp/gpt/speed/easydist_pipeline.py --dataset-size 5000 --micro-batch-size 16 --num-chunks $i --schedule dapple
done
torchrun --nproc_per_node 4 benchmark/torch/pp/gpt/speed/easydist_pipeline.py --dataset-size 5000 --micro-batch-size 16 --num-chunks 98 --schedule dapple

for ((i=1; i <= 32 ; i *= 2)); do
    python benchmark/torch/pp/gpt/speed/torchgpipe_pipeline.py --dataset-size 5000 --micro-batch-size 16 --num-chunks $i
done
python benchmark/torch/pp/gpt/speed/torchgpipe_pipeline.py --dataset-size 5000 --micro-batch-size 16 --num-chunks 34

for ((i=1; i <= 256 ; i *= 2)); do
    python benchmark/torch/pp/gpt/speed/vanilla_torch.py --dataset-size 5000 --micro-batch-size 16 --num-chunks $i
done


