
NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 accelerate launch  --main_process_port 29401 --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --model DiT-S/4 --feature-path features

NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 accelerate launch  --main_process_port 29401 --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --model DiT-S/4 --feature-path features --register 1

NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 accelerate launch  --main_process_port 29401 --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --model DiT-S/4 --feature-path features --register 2

NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 accelerate launch  --main_process_port 29401 --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --model DiT-S/4 --feature-path features --register 4

NCCL_P2P_DISABLE=1  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 accelerate launch  --main_process_port 29401 --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --model DiT-S/4 --feature-path features --register 8



