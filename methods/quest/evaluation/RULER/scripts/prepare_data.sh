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


# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR=" " # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization
CHUNK_SIZE=16
TOKEN_BUDGET=1024
cd evaluation/RULER/scripts/
# Model and Tokenizer
source config_models.sh

MODEL_NAME="Llama-3.1-8B-Instruct"
#MODEL_NAME="Qwen2.5-7B-Instruct-1M"

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

echo ${TOKENIZER_PATH}
echo ${TOKENIZER_TYPE}
echo ${MODEL_TEMPLATE_TYPE}
echo ${MAX_SEQ_LENGTH}

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK="synthetic"
declare -n TASKS=$BENCHMARK

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
total_time=0
#for model in "Qwen2.5-7B-Instruct-1M";do
for model in "Llama-3.1-8B-Instruct";do
    #for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    for MAX_SEQ_LENGTH in 4096; do

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data-test"
    PRED_DIR="${RESULTS_DIR}/pred-vt02"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    #for MAX_SEQ_LENGTH in 4096 8192 16384 32768 65536; do    #
    
    TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"
    for TASK in $TASKS;do
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
    done
    
    done
done