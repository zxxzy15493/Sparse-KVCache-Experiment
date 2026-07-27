#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=0

if ! "${SCRIPT_DIR}/run_mem_pq_4k_to_128k.sh"; then
    FAILED=1
fi

if ! "${SCRIPT_DIR}/run_mem_full_4k_to_128k.sh"; then
    FAILED=1
fi

exit "${FAILED}"
