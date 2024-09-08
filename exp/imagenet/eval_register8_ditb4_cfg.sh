export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

for register in 8; do
    torchrun --rdzv-endpoint=localhost:29403 --nnodes=1 --nproc_per_node=4 sample_ddp.py --model DiT-B/4 --num-fid-samples 50000 --ckpt results/DiT-B-4_register$register/checkpoints/0400000.pt --register $register --cfg-scale 3.0
done