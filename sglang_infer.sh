#!/bin/bash
set -x
set -f

CUDA_VISIBLE_DEVICES=0
GPUS_PER_NODE=1
RANK=0 # 多机情况下读取环境变量，保证每台的rank不一样
NNODES=1 # 机器数

echo "Connect ray cluster"
# 每一台机器都是头节点，不要设置互联
ray start --num-gpus=$GPUS_PER_NODE --head

sleep 10
echo "Ray cluster status:"
ray status

OPTS=$@
python sglang_infer.py \
    --ngpus $GPUS_PER_NODE \
    --node_rank $RANK \
    --nnodes $NNODES \
    $OPTS

ray stop
# bash sglang_infer.sh --model_path /data/models/Qwen2.5-0.5B-Instruct --tp_size 1 --input_path /data/data/test.json --output_path /data/data/test --batch_size 100 --max_new_tokens 128
# pip install outlines==0.0.44