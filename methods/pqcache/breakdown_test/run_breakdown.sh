#!/bin/bash

MODEL_PATH=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
INPUT_FILE=${2:-"./myinput.txt"}
# Ensure myinput.txt exists at INPUT_FILE; copy from benchmarks/ via relative path if missing
if [ ! -f "${INPUT_FILE}" ]; then
    _SRC="$(dirname "$0")/../../../benchmarks/myinput.txt"
    if [ -f "${_SRC}" ]; then
        case "${INPUT_FILE}" in
            /*) _DEST="${INPUT_FILE}" ;;
            *)  _DEST="$(dirname "$0")/${INPUT_FILE#./}" ;;
        esac
        mkdir -p "$(dirname "${_DEST}")"
        cp "${_SRC}" "${_DEST}"
    fi
fi
unset _SRC _DEST
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/breakdown_test_once.py" \
    --model "${MODEL_PATH}" \
    --input-file "${INPUT_FILE}" \
    --input-lens 4096 8192 16384 32768 65536 131072 \
    --max-new-tokens 32 \
    --budget 1024 \
    --warmup-rounds 3 \
    --measure-rounds 7 \
    --csv ./breakdown_test/log/breakdown_results.csv \
    --device cuda:0
