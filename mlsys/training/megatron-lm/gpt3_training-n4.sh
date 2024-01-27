#!/bin/bash

#SBATCH -p batch --reservation=testing -t 1:00:00 --nodes=4 --exclusive --mem=0 --ntasks-per-node=8 --gpus-per-node=8 --job-name=gpt3-15b-8t-n4

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_SOCKET_IFNAME=ibp101s0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152

NAME="gpt3-15b-8t-n4"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

CHECKPOINT_DIR="${DIR}/../checkpoints/${NAME}"
mkdir -p ${CHECKPOINT_DIR}
# mkdir -p ${CHECKPOINT_DIR_LOAD}

NSYS_LOG_DIR=/home/${USER}/nsys/${SLURM_JOBID}/
mkdir -p $NSYS_LOG_DIR

BLEND_NAME="8t_blend"
DATACACHE_DIR="${DIR}/../data_cache/${BLEND_NAME}"

TENSORBOARD_DIR="$DIR/../tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /home/haowan/data/nvllm-3.5t/nvllm-3.5t-blend.sh

options=" \
    --distributed-timeout-minutes 60 \
    --use-mcore-models \
    --data-cache-path ${DATACACHE_DIR} \
    --sequence-parallel \
    --recompute-activations \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 460 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 768 \
    --train-samples 1953125000 \
    --lr-decay-samples 1855468750 \
    --lr-warmup-samples 976563 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval 10 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /home/haowan/data/nvllm-3.5t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 11 \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.008 \
    --log-num-zeros-in-grad \
    --log-throughput \
    --bf16 \
    --overlap-grad-reduce \
    --no-load-optim \
    --tp-comm-overlap \
    --no-create-attention-mask-in-dataloader \
    --manual-gc \
    --num-workers 1 \
    --timing-log-level 2 \
    --use-distributed-optimizer \
    --overlap-param-gather \
    --profile-step-start 10 \
    --profile-step-end 12 \
    --profile \
    --tensorboard-dir ${TENSORBOARD_DIR}"

run_cmd="nsys profile --output ${NSYS_LOG_DIR}/nsys_%q{SLURM_JOB_NAME}_%q{SLURM_JOBID}_%q{SLURM_NODEID}_%q{SLURM_PROCID} \
         -s none \
         --capture-range cudaProfilerApi \
         --capture-range-end stop \
         python -u ${DIR}/pretrain_gpt.py ${options}; \
         sleep 600"

srun -l \
     --container-image "/home/haowan/images/adlr+megatron-lm+pytorch+23.09-py3-pretrain-draco_cw_ub_tot-te-apex.sqsh" \
     --container-mounts "/home:/home" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

