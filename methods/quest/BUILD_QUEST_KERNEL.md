# Reproducible Quest Kernel Build

This guide describes how to build the Quest Python CUDA extension for experiments. It avoids machine-specific absolute paths and supports different NVIDIA GPU architectures.

The required output is a Python extension importable as:

```text
quest._kernels
```

The optional `methods/quest/kernels` benchmark/test executables are not required for normal Quest inference experiments.

## 1. Clone And Initialize Submodules

From the repository root:

```bash
git submodule update --init --recursive methods/quest
```

Check that the third-party folders are populated:

```bash
git submodule status --recursive methods/quest
```

No line should start with `-`. A leading `-` means that submodule has not been initialized.

## 2. Prepare Environment

Activate the conda environment you want to use:

```bash
conda activate <env-name>
```

The environment should provide:

- Python and headers
- PyTorch with CUDA support
- CUDA toolkit with `nvcc`
- CMake
- Ninja
- A C++17 compiler, such as GCC/G++ 11

Verify the core tools:

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)'
nvcc --version
cmake --version
ninja --version
```

## 3. Set Portable Build Variables

From the repository root:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CPM_SOURCE_CACHE="$REPO_ROOT/methods/quest/kernels/cpm_cache"
export MAX_JOBS=4
export MAXJOBS=4
```

Detect the visible GPU architecture with PyTorch:

```bash
eval "$(python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible. Set GPU architecture manually before building.")

major, minor = torch.cuda.get_device_capability(0)
print(f'export TORCH_CUDA_ARCH_LIST="{major}.{minor}"')
print(f'export CUDAARCHS="{major}{minor}"')
print(f'export CMAKE_CUDA_ARCHITECTURES="{major}{minor}"')
PY
)"

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CUDAARCHS=$CUDAARCHS"
echo "CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
```

Examples:

```text
A100:  TORCH_CUDA_ARCH_LIST=8.0,  CMAKE_CUDA_ARCHITECTURES=80
RTX 4090: TORCH_CUDA_ARCH_LIST=8.9,  CMAKE_CUDA_ARCHITECTURES=89
H100:  TORCH_CUDA_ARCH_LIST=9.0,  CMAKE_CUDA_ARCHITECTURES=90
Blackwell: TORCH_CUDA_ARCH_LIST=12.0, CMAKE_CUDA_ARCHITECTURES=120
```

If the build machine cannot see a GPU, set these variables manually according to the target GPU.

## 4. Prepare CPM

Quest/RAFT uses CPM to fetch matching third-party dependencies.

```bash
mkdir -p "$CPM_SOURCE_CACHE/cmake"

if [ ! -s "$CPM_SOURCE_CACHE/cmake/CPM_0.38.5.cmake" ]; then
  curl -fL --retry 3 --connect-timeout 20 \
    -o "$CPM_SOURCE_CACHE/cmake/CPM_0.38.5.cmake" \
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.5/CPM.cmake
fi
```

## 5. Build Quest Python Operators

The build should use the RAFT submodule and the matching CPM dependencies. Do not mix the RAFT submodule with unrelated `rmm`, `spdlog`, or `fmt` versions already installed in the environment.

```bash
cd "$REPO_ROOT/methods/quest/quest/ops"
mkdir -p build_repro
cd build_repro

cmake -GNinja \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCMAKE_DISABLE_FIND_PACKAGE_raft=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_rmm=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_spdlog=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_fmt=ON \
  -DCPM_raft_SOURCE="$REPO_ROOT/methods/quest/kernels/3rdparty/raft" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  -DCMAKE_CUDA_FLAGS="-diag-suppress=20281" \
  ..

ninja -j4
```

The `CMAKE_DISABLE_FIND_PACKAGE_*` flags are intentional. They force CMake to use versions compatible with the Quest RAFT submodule instead of accidentally picking incompatible packages from the active conda environment.

## 6. Link Extension Into The Package

The Python package imports `quest._kernels`, so link the compiled shared object into the `quest` package directory:

```bash
cd "$REPO_ROOT/methods/quest/quest/ops"

for file in ./build_repro/*.so; do
  ln -sfn "$(realpath "$file")" ../"$(basename "$file")"
done
```

## 7. Verify

```bash
cd "$REPO_ROOT/methods/quest"

PYTHONPATH="$REPO_ROOT/methods/quest" \
python -c 'import quest._kernels as k; print(k.__file__); print([x for x in dir(k) if not x.startswith("_")])'
```

The command should print the `_kernels` shared object path and exported functions such as:

```text
append_kv_cache_decode
append_kv_cache_prefill
apply_rope_in_place
estimate_attn_score
prefill_with_paged_kv_cache
rms_norm_forward
topk_filtering
```

Also check for missing dynamic libraries:

```bash
ldd "$REPO_ROOT/methods/quest/quest"/_kernels*.so | grep "not found" || true
```

No `not found` line should appear.

## Troubleshooting

- `CUDA_ARCHITECTURES is set to native, but no GPU was detected`: use explicit `CMAKE_CUDA_ARCHITECTURES` instead of `native`.
- Errors mentioning `sm_50`, `sm_60`, or old architectures: set `TORCH_CUDA_ARCH_LIST` before configuring CMake.
- Errors in `rmm`, `spdlog`, or `fmt`: make sure the `CMAKE_DISABLE_FIND_PACKAGE_*` flags are present so CPM uses compatible versions.
- CUDA warning `#20281` treated as an error: keep `-DCMAKE_CUDA_FLAGS="-diag-suppress=20281"`.
- Out of memory during compilation: lower `MAX_JOBS`, `MAXJOBS`, and `ninja -j`.
