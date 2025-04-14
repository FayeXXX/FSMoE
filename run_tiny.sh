#!/usr/bin/env bash

splitid=(0 1 2 3 4)
#mt=$2
gpuid=$1

for id in "${splitid[@]}" # split id
do
    python osr_lowlayer3.py  --dataset tinyimagenet --out_num 10 --loss Softmax --use_default_parameters False --num_workers 32 --split_idx ${id} \
    --batch_size 32 --LR 0.004 --MAX_EPOCH 60 --transform cocoop --backbone ViT-B-32 --gpu ${gpuid} --method lowlayer3 --prec fp16
done



