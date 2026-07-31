#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"


"$PYTHON_BIN" "$SCRIPT_DIR/evaluation_full.py" \
    --save_dir "$SCRIPT_DIR/res"
