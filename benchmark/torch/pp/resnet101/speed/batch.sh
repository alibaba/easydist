rm *.txt
rm *.dot
for ((i=1; i <=256 ; i *= 2)); do
    torchrun --nproc_per_node 4 benchmark/torch/pp/resnet101/speed/easydist_pipeline.py --micro-batch-size 128 --num-chunks $i --schedule gpipe --do-profile
done

for ((i=1; i <=256 ; i *= 2)); do
    torchrun --nproc_per_node 4 benchmark/torch/pp/resnet101/speed/easydist_pipeline.py --micro-batch-size 128 --num-chunks $i --schedule dapple --do-profile
done


