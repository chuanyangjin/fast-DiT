export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

for register in 1; do
    torchrun --nnodes=1 --nproc_per_node=5 sample_ddp.py --model DiT-B/4 --num-fid-samples 50000 --ckpt results/DiT-B-4_register$register/checkpoints/0100000.pt --register $register
done