#!/bin/bash
start_time=$(date +%s)

# # Runs the "1.5B" parameter model
# # export NCCL_SOCKET_IFNAME=ens10f0  #mlnx-0：40G
# export NCCL_SOCKET_IFNAME=ens10f1  # mlnx-1：40G
# export NCCL_IB_DISABLE=1         # 禁用 InfiniBand，走以太网
# export NCCL_P2P_DISABLE=1        # 禁用 GPU-P2P 通信（跨节点不稳定时可设）
# export NCCL_NET=Socket           # 强制 NCCL 使用 Socket 模式通信  
# # export NCCL_DEBUG=INFO 
# # export MEGATRON_LOG_LEVEL=INFO

export NCCL_SOCKET_IFNAME=ens10f1np1  #mlnx-0：40G
export NCCL_NET=IB         # 
export NCCL_IB_GID_INDEX=3       # 
echo "已切换到 RoCE 模式"

GPUS_PER_NODE=4
MASTER_ADDR=192.168.41.1 #
MASTER_PORT=6000
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=mydata_processed/meg-gpt2_text_document
CHECKPOINT_PATH=checkpoints/gpt2

# ========== 启动训练 ==========
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 48 \
       --hidden-size 1536 \
       --num-attention-heads 24 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file mydata_processed/gpt2-vocab.json \
       --merge-file mydata_processed/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 500 \
       --eval-interval 10 \
       --eval-iters 10 \
       --fp16
       
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "总训练耗时: $duration 秒"

# --load $CHECKPOINT_PATH \