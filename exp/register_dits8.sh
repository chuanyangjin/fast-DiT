#!/bin/bash

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

accelerate launch --main_process_port 29402 --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features --load-checkpoint

for register in 1 2 4 8; do
    accelerate launch --main_process_port 29402 --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features --register $register --load-checkpoint
done