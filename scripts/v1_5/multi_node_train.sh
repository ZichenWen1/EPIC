#!/usr/bin/env bash
source /mnt/petrelfs/fengpeilin/miniconda3/bin/activate vmba
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export GPUS_PER_NODE=8
export NNODES=2
export MASTER_PORT=10043
export CPUS_PER_TASK=16
export NAME=vmamba_nv2d
export EXP_NAME=$NAME
export NUM_TRAIN_EPOCHS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PARTITION=${1:-bigdata_sdd}
NODE_LIST=${2:-"SH-IDC1-10-140-24-43,SH-IDC1-10-140-24-63"}
srun -p bigdata_sdd -w $NODE_LIST \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    ACCELERATE_CPU_AFFINITY=1 python -m torch.distributed.launch --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} /mnt/petrelfs/fengpeilin/Mamba/VMamba_NV2D/classification/main.py \
        --cfg /mnt/petrelfs/fengpeilin/Mamba/VMamba_NV2D/classification/configs/vssm/vmambav2v_tiny_224.yaml \
        --batch-size 256 \
        --data-path /mnt/petrelfs/fengpeilin/huggingfacedata/imagenet_new \
        --output /mnt/petrelfs/fengpeilin/Mamba/VMamba_NV2D/classification/result \
        --pre_trained /mnt/petrelfs/fengpeilin/Mamba/VMamba_NV2D/classification/result/vssm1_tiny_0230s/20250331202906/ckpt_epoch_37.pth

    