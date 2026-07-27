#!/bin/bash
# One-click build script for minference CUDA extension
# Usage: bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="/tmp/minference_cuda_build_$$"

echo "=== Building minference CUDA extension ==="
echo "Source dir: $SCRIPT_DIR"
echo "Project dir: $PROJECT_DIR"

# Clean up on exit
cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"

# Create build script
cat > "$BUILD_DIR/build.py" << 'PYEOF'
import sys
import shutil
from torch.utils import cpp_extension

sources = [
    sys.argv[1] + "/kernels.cpp",
    sys.argv[1] + "/vertical_slash_index.cu",
]

ext = cpp_extension.load(
    name="cuda",
    sources=sources,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
    verbose=True,
    build_directory=sys.argv[2],
    is_python_module=True,
)
PYEOF

# Compile
python "$BUILD_DIR/build.py" "$SCRIPT_DIR" "$BUILD_DIR"

# Copy to minference/
cp "$BUILD_DIR/cuda.so" "$PROJECT_DIR/minference/cuda.so"
echo "  -> Copied to minference/cuda.so"

# Copy to minference_time/ if exists
if [ -d "$PROJECT_DIR/minference_time" ]; then
    cp "$BUILD_DIR/cuda.so" "$PROJECT_DIR/minference_time/cuda.so"
    echo "  -> Copied to minference_time/cuda.so"
fi

echo "=== Done ==="
