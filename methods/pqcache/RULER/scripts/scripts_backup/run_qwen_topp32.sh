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

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name>"
    exit 1
fi


# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="../.." # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

# PQ Compression Parameters (used when MODEL_FRAMEWORK='pq')
PQ_FIXBUDGET="true"   # Enable fixed budget mode
PQ_BUDGET=1024         # Fixed budget size
PQ_COMPRESS_RATIO=0.1  # KV cache compression ratio
PQ_IMPORTANT_RATIO=0.5  # Ratio of important tokens to retrieve
PQ_RECENT_RATIO=0.5   # Ratio of recent tokens to preserve
PQ_RECENT_SIZE=32     # Number of recent tokens to keep
PQ_SINK_SIZE=16       # Number of most recent tokens to always keep
PQ_COMPRESSOR="no_drop_lb_topp32"  # Compression method: pq_search, sparq_f, infllm, h2o, original
PQ_N_SUBVEC=2         # Number of PQ subvectors per head
PQ_N_SUBBITS=6       # Bits per PQ subvector
PQ_TOPR=1          # Top-k tokens to retrieve during decoding
PQ_GQA="True"        # Whether to use grouped-query attention
PQ_MAX_SEQ_LEN=131072  # Maximum sequence length
PQ_CACHE_BLOCK_SIZE=128  # Block size for cache management
PQ_GLOBAL_CACHE_SIZE=4096  # Size of global cache
PQ_CACHE_TOPK=32     # Number of top-k tokens for cache retrieval
PQ_SCORE_FUNC="sum"  # Score function: sum or max
PQ_DROP_RATIO=0      # Drop ratio for tokens
PQ_MAX_ITER=0        # K-means iterations (0 for auto)
PQ_PRESERVE_LAYER=0  # Number of layers to preserve without compression
PQ_FIXTHRESHOLD=0.9 # Fixed threshold for topp attention

# System parameters
DEVICE=0              # GPU device ID
MAX_CPU_IN_USE=12    # Number of CPU cores for K-means
SEED=4321             # Random seed


# Model and Tokenizer
source config_models.sh

SEQ_LENGTHS=(
    4096
    8192
    16384
    32768
    65536
    # 131072
)
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks

source config_tasks.sh
NUM_SAMPLES=50
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi


# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        &

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        &

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer \
        &
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs

fi


# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    # Build model name with PQ suffix
    PQ_MODEL_NAME=""
    if [ "${MODEL_FRAMEWORK}" = "pq" ]; then
        if [[ "${PQ_COMPRESSOR}" == *"topp"* ]]; then
            PQ_MODEL_NAME="-${PQ_COMPRESSOR}_thr${PQ_FIXTHRESHOLD}"
        else
            PQ_MODEL_NAME="-${PQ_COMPRESSOR}_bud${PQ_BUDGET}"
        fi
    fi

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}${PQ_MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    # DATA_DIR="${RESULTS_DIR}/data"
    DATA_DIR="../../../../../benchmarks/ruler/benchmark_root////data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        echo "Running ${BENCHMARK} - ${TASK} - ${MAX_SEQ_LENGTH}"
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
                --pq_n_subvec_per_head ${PQ_N_SUBVEC} \
                --pq_n_subbits ${PQ_N_SUBBITS} \
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

        TOPP_SAVE_TOPK="qwen_topp32_09_0422_1"\
        MAX_CPU_IN_USE=${MAX_CPU_IN_USE} \
        RANDOM_SEED=${SEED} \
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
        CUDA_VISIBLE_DEVICES=${DEVICE} \
        TOKENIZERS_PARALLELISM=false \
        SUBVEC=${PQ_N_SUBVEC} \
        SUBBITS=${PQ_N_SUBBITS} \
        METRIC="euc" \
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
