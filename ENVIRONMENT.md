# Environment Setup

The environment described in this document has been successfully used to reproduce the experiments on an NVIDIA RTX PRO 6000 GPU.

## Base Environment

The code is recommended to run with `Python 3.12`, `CUDA 12.8`, and `PyTorch 2.8`.
We recommend using `conda` to manage the Python environment.

```bash
conda create -n kv python=3.12.13 -y
conda activate kv

conda install -y mkl
conda install -c conda-forge libstdcxx-ng -y

pip install torch==2.8.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install flashinfer-python

pip install minference==0.1.6.0
pip install -U "cmake==3.26.4" ninja

```

## Additional Method Setup

If you plan to run a specific method, install that method's additional kernels and dependencies as needed.

### XAttention Block-Sparse Attention

#### Install Block Sparse Streaming Attention

```bash
cd methods/x-attention
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
python setup.py install
cd ../../..
```

### Quest Kernel

#### Install CMake

Install CMake version 3.26.4 or later.

```bash
cd methods/quest
conda install cmake
```

#### Build libraft

```bash
cd kernels/3rdparty/raft
./build.sh libraft
```

#### Compile Kernel Benchmarks (Optional)

Configure the CUDA environment variables as described in the Quest tutorial before running the following commands.

```bash
cd kernels
mkdir build && cd build
cmake ..
make -j
```

#### Build End-to-End PyBind Operators

```bash
cd quest/ops
bash setup.sh
```

### PQCache LFU

```bash
conda activate kv
cd methods/pqcache/vq_method/retrieval_based/lfu
rm -rf build
mkdir build
cd build
cmake ..
make -j
```

### MInference

```bash
conda activate kv
cd methods/minference/csrc
bash build.sh
```

### AdaKV / HeadKV CUDA Kernel

AdaKV and HeadKV share the same `csrc/` CUDA kernel (each has an identical copy). The kernel provides `update_flatten_view` for efficient KV cache tensor manipulation, compiled via `torch.utils.cpp_extension.CUDAExtension`.

```bash
conda activate test
cd methods/Adakv/csrc
python build.py install

# HeadKV uses an identical kernel — compile it too if needed:
cd methods/HeadKV/csrc
python build.py install
```

The `build.py` uses PyTorch's `CUDAExtension` to compile `csrc/cuda_api.cu` into the `tiny_pkg` package, which exposes `tiny_api_cuda.update_flatten_view` at runtime. The compiled `.so` is installed into the current Python environment's `site-packages`.

---

The build script compiles the CUDA extension (`kernels.cpp` + `vertical_slash_index.cu`) and copies `cuda.so` to both `minference/` and `minference_time/` (if present).

For the A6000Pro setup, RetroInfer, MagicPIG, ClusterKV uses the following versions because of flashinfer-python (ClusterKV also can run in CUDA12.8 with flashattn):

- CUDA **12.9**
- flashinfer-python **0.6.12**

### MagicPIG

> **Note:** MagicPIG requires a CPU with AVX-512 support.

```bash
conda activate kv
cd methods/magicpig
bash install.sh
```

The install script performs the following steps:

1. Install `flashinfer-python==0.6.12` (CUDA 12.9, PyTorch 2.8); `flashinfer-python==0.2.4` (CUDA 12.4, PyTorch 2.5)
2. Build and install the `sparse_attention` library (`library/sparse_attention/`)
3. Build and install the `lsh` library (`library/lsh/`)

### RetroInfer

```bash
conda activate kv
cd methods/retroinfer
bash install.sh
```

The install script performs the following steps:

1. Install `flashinfer-python==0.6.12` (CUDA 12.9, PyTorch 2.8); `flashinfer-python==0.2.4` (CUDA 12.4, PyTorch 2.5)
2. Install a custom `flash-attention` from source (`git+https://github.com/Starmys/flash-attention.git@weighted`)
3. Clone NVIDIA CUTLASS (`library/cutlass/`)
4. Build and install the `retroinfer` library (`library/retroinfer/`)

### ClusterKV Kernel

#### Build libraft

```bash
cd ../../../../ClusterKV
export VIRTUAL_ENV="$CONDA_PREFIX"
cd 3rdparty/raft
INSTALL_PREFIX="$CONDA_PREFIX" ./build.sh libraft
```

#### Build the ClusterKV Kernel

```bash
cd ../../kernel
bash setup.sh
```

#### Test the ClusterKV Kernel

```bash
cd ..
python - <<'PY'
import clusterkv._clusterkv_knl
import clusterkv._quest_knl
print("ClusterKV kernels import OK")
PY
```
