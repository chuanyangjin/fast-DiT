#!/bin/bash

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch --main_process_port 29402 --num_processes 3 --mixed_precision fp16 train.py --model DiT-B/4 --feature-path features --load-checkpoint --register 1

# for register in 1 2 4 8; do
#     accelerate launch --main_process_port 29401 --num_processes 1 --mixed_precision fp16 train.py --model DiT-B/4 --feature-path features --register $register --load-checkpoint
# done
