#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/evaluation_full.py" \
    --save_dir "$SCRIPT_DIR/res717_2" \
    --token_budget 360
