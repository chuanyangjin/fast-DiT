# NCCL_P2P_DISABLE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 torchrun --rdzv-endpoint=localhost:29401 --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-B/2 --data-path /media/dataset1/ImageNet2012 --features-path features

NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 accelerate launch --multi_gpu --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features

NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 accelerate launch  --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --mixed_precision fp16 train.py --model DiT-S/8 --feature-path features --register 2