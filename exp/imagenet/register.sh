#!/bin/bash
export OMP_NUM_THREADS=2
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7

accelerate launch --multi-gpu --main_process_port 29402 --num_processes 2 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features

for register in 1 2 4 8; do
    accelerate launch --multi-gpu --main_process_port 29402 --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features --register $register
done

for register in 1 2 4 8; do
    for model in "DiT-S/8" "DiT-S/4" "DiT-S/2"; do
        accelerate launch --multi-gpu --main_process_port 29402 --num_processes 1 --mixed_precision fp16 train.py --model $model --feature-path features --register $register
    done
done