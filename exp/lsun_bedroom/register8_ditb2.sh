#!/bin/bash

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5,6,7

accelerate launch --main_process_port 29403 --num_processes 3 --mixed_precision fp16 train.py --model DiT-B/2 --feature-path features --register 8 --load-checkpoint --lsun --num-classes 1
