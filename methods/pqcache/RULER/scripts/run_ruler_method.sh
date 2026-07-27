#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Root Directories
GPUS="1" # GPU size for tensor_parallel.
DATA_ROOT="../../../../benchmarks/ruler/benchmark_root"
RESULT_ROOT="benchmark_root"
BATCH_SIZE=1  # increase to improve GPU utilization

# PQ Compression Parameters (used when MODEL_FRAMEWORK='pq')
PQ_FIXBUDGET="true"   # Enable fixed budget mode
PQ_BUDGET=${PQ_BUDGET:?PQ_BUDGET is required}
PQ_COMPRESS_RATIO=${PQ_COMPRESS_RATIO:-0.1}
PQ_IMPORTANT_RATIO=${PQ_IMPORTANT_RATIO:-0.5}
PQ_RECENT_RATIO=${PQ_RECENT_RATIO:-0.5}
PQ_RECENT_SIZE=${PQ_RECENT_SIZE:?PQ_RECENT_SIZE is required}
PQ_SINK_SIZE=${PQ_SINK_SIZE:?PQ_SINK_SIZE is required}
PQ_COMPRESSOR=${PQ_COMPRESSOR:?PQ_COMPRESSOR is required}
PQ_N_SUBVEC=${PQ_N_SUBVEC:-2}
PQ_N_SUBBITS=${PQ_N_SUBBITS:-6}
PQ_TOPR=${PQ_TOPR:-1}
PQ_GQA=${PQ_GQA:-True}
PQ_MAX_SEQ_LEN=131072  # Maximum sequence length
PQ_CACHE_BLOCK_SIZE=128  # Block size for cache management
PQ_GLOBAL_CACHE_SIZE=4096  # Size of global cache
PQ_CACHE_TOPK=32     # Number of top-k tokens for cache retrieval
PQ_SCORE_FUNC=${PQ_SCORE_FUNC:-sum}
PQ_DROP_RATIO=${PQ_DROP_RATIO:-0}
PQ_MAX_ITER=${PQ_MAX_ITER:-0}
PQ_PRESERVE_LAYER=${PQ_PRESERVE_LAYER:-0}
PQ_FIXTHRESHOLD=${PQ_FIXTHRESHOLD:--1}

# System parameters
DEVICE=${DEVICE:-0}
MAX_CPU_IN_USE=${MAX_CPU_IN_USE:-12}
SEED=${SEED:-4321}


MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
TOKENIZER_PATH=${TOKENIZER_PATH:?TOKENIZER_PATH is required}
MODEL_TEMPLATE_TYPE=${MODEL_TEMPLATE_TYPE:?MODEL_TEMPLATE_TYPE is required}
EXP_NAME=${EXP_NAME:?EXP_NAME is required}
MODEL_FRAMEWORK=pq
TOKENIZER_TYPE=hf
TEMPERATURE=1.0
TOP_K=32
TOP_P=1.0
STOP_WORDS=""
BENCHMARK=synthetic
NUM_SAMPLES=50
SEQ_LENGTHS=(${SEQ_LENGTHS:?SEQ_LENGTHS is required})
TASKS=(${TASKS:?TASKS is required})

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"


# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    CURRENT_N_SUBVEC=${PQ_N_SUBVEC}
    CURRENT_N_SUBBITS=${PQ_N_SUBBITS}
    if [ "${MAX_SEQ_LENGTH}" -ge 65536 ]; then
        CURRENT_N_SUBVEC=${PQ_N_SUBVEC_64K:-${CURRENT_N_SUBVEC}}
        CURRENT_N_SUBBITS=${PQ_N_SUBBITS_64K:-${CURRENT_N_SUBBITS}}
    fi

    # Build model name with PQ suffix
    PQ_MODEL_NAME=""
    if [ "${MODEL_FRAMEWORK}" = "pq" ]; then
        if [[ "${PQ_COMPRESSOR}" == *"topp"* ]]; then
            PQ_MODEL_NAME="-${PQ_COMPRESSOR}_thr${PQ_FIXTHRESHOLD}"
        else
            PQ_MODEL_NAME="-${PQ_COMPRESSOR}_bud${PQ_BUDGET}"
        fi
    fi

    RESULTS_DIR="${RESULT_ROOT}/${MODEL_NAME}${PQ_MODEL_NAME}/synthetic/${MAX_SEQ_LENGTH}"
    DATA_DIR="${DATA_ROOT}/${MODEL_NAME}/synthetic/${MAX_SEQ_LENGTH}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        echo "Running ${EXP_NAME} - ${MODEL_NAME} - ${TASK} - ${MAX_SEQ_LENGTH}"
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
        
        start_time=$(date +%s)
        echo "Calling model API..."

        # PQ parameters (only used when MODEL_FRAMEWORK is 'pq')
        PQ_PARAMS=""
        if [ "${MODEL_FRAMEWORK}" = "pq" ]; then
            PQ_PARAMS="\
                --pq_fixbudget
                --pq_budget ${PQ_BUDGET} \
                --pq_compress_ratio ${PQ_COMPRESS_RATIO} \
                --pq_important_ratio ${PQ_IMPORTANT_RATIO} \
                --pq_recent_ratio ${PQ_RECENT_RATIO} \
                --pq_recent_size ${PQ_RECENT_SIZE} \
                --pq_sink_size ${PQ_SINK_SIZE} \
                --pq_compressor ${PQ_COMPRESSOR} \
                --pq_n_subvec_per_head ${CURRENT_N_SUBVEC} \
                --pq_n_subbits ${CURRENT_N_SUBBITS} \
                --pq_topr ${PQ_TOPR} \
                --pq_gqa ${PQ_GQA} \
                --pq_max_seq_len ${PQ_MAX_SEQ_LEN} \
                --pq_cache_block_size ${PQ_CACHE_BLOCK_SIZE} \
                --pq_global_cache_size ${PQ_GLOBAL_CACHE_SIZE} \
                --pq_cache_topk ${PQ_CACHE_TOPK} \
                --pq_score_func ${PQ_SCORE_FUNC} \
                --pq_drop_ratio ${PQ_DROP_RATIO} \
                --pq_max_iter ${PQ_MAX_ITER} \
                --pq_preserve_layer ${PQ_PRESERVE_LAYER} \
                --fixthreshold ${PQ_FIXTHRESHOLD}"
        fi

        export MAX_CPU_IN_USE
        RANDOM_SEED=${SEED}
        export RANDOM_SEED
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
        export PYTORCH_CUDA_ALLOC_CONF
        CUDA_VISIBLE_DEVICES=${DEVICE}
        export CUDA_VISIBLE_DEVICES
        TOKENIZERS_PARALLELISM=false
        export TOKENIZERS_PARALLELISM
        SUBVEC=${CURRENT_N_SUBVEC}
        export SUBVEC
        SUBBITS=${CURRENT_N_SUBBITS}
        export SUBBITS
        METRIC="euc"
        export METRIC

        python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            ${STOP_WORDS} \
            ${PQ_PARAMS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
        echo "Time spent on ${TASK}: ${time_diff} seconds"
    done
    
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
