#! /bin/bash

GPUS_PER_NODE=16
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=1
export DLWS_NUM_GPU_PER_WORKER=16

DATA_PATH=data/webtext/webtext_text_document
CHECKPOINT_PATH=/turing-nfs/users/samyamr/checkpoints/

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_zero_stage_2_config.json"

#Megatron Model Parallelism
mp_size=16

#ZeRO Configs
stage=2
reduce_scatter=true
contigious_gradients=false
rbs=200000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false


gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers 125 \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size 32 \
        --train-iters 500000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file data/gpt2-vocab.json \
        --merge-file data/gpt2-merges.txt \
        --data-impl mmap \
        --eval-iters 200 \
        --log-interval 10 \
        --save-interval 5000 \
        --eval-interval 5000 \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --fp16 \
        --hysteresis 2 \
        --num-workers 0
"

       #Some Old Parameters
       #  --load $CHECKPOINT_PATH \
       #  --split 900,9,1 \
       #  --hidden-dropout 0.1 \
       #  --attention-dropout 0.1 \
       #  --hysteresis 2 \
       #  --num-workers 0 \
       #  --cache-dir /data/bert/users/corosset/ConversationalUnderstanding/Bert/checkpoints/MegatronGPT2/cache_bing \
       #   --save /turing-nfs/users/samyamr/checkpoints/tests \
       #  --train-data webtext \
       #  --resume-dataloader \
       #  --lazy-loader \
       #  --tokenizer-type GPT2BPETokenizer \
       #  --cache-dir cache \
        
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "
#deepspeed_options=""

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

