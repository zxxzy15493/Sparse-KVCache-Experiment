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

pip install flashinfer-python==0.6.1


pip install -U "cmake==3.26.4" ninja

```

## Editable Install

The following methods use absolute imports (e.g., `from evaluation.mistral import ...`) in their scripts. Before running any of them, you must install the corresponding package in editable mode, otherwise Python will raise a `ModuleNotFoundError`.

### Prerequisites

```bash
# flexPrefill
pip install -e methods/flexPrefill

# X-Attention
pip install -e methods/x-attention

# Quest
pip install -e methods/quest

# SparQ
pip install -e methods/sparq
```

> **Note:** These commands should be run from the repository root directory. The `-e` (editable) flag means changes to the source code take effect immediately without reinstalling.



## Additional Method Setup

If you plan to run a specific method, install that method's additional kernels and dependencies as needed.

### XAttention Block-Sparse Attention

#### Install Block Sparse Streaming Attention

```bash
cd methods/x-attention
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
MAX_JOBS=4 python setup.py install
cd ../../..
```

### Quest Kernel

The Quest kernel build requires several CUDA/CMake dependency settings, and the full setup process is relatively long. For clarity, the detailed installation and troubleshooting instructions are provided in a separate guide.

see [BUILD_QUEST_KERNEL.md](methods/quest/BUILD_QUEST_KERNEL.md)


### PQCache LFU

```bash
pushd methods/pqcache/vq_method/retrieval_based/lfu
rm -rf build
mkdir build
cd build
cmake ..
make -j
popd
```

### ClusterKV Kernel

#### Build libraft


```bash
unset VIRTUAL_ENV
unset PYTHONHOME
pushd methods/ClusterKV/3rdparty/raft
rm -rf cpp/build
INSTALL_PREFIX="$CONDA_PREFIX" ./build.sh libraft
popd
```

#### Build the ClusterKV Kernel

```bash
pushd methods/ClusterKV/kernel
bash setup.sh
popd
```

#### Test the ClusterKV Kernel

```bash
pip install -e methods/ClusterKV

python - <<'PY'
import clusterkv._clusterkv_knl
import clusterkv._quest_knl
print("ClusterKV kernels import OK")
PY
```


### MInference

```bash
conda activate kv
cd methods/minference/csrc
bash build.sh
```

---

The build script compiles the CUDA extension (`kernels.cpp` + `vertical_slash_index.cu`) and copies `cuda.so` to both `minference/` and `minference_time/` (if present).

### AdaKV / HeadKV CUDA Kernel

AdaKV and HeadKV share the same `csrc/` CUDA kernel (each has an identical copy). The kernel provides `update_flatten_view` for efficient KV cache tensor manipulation, compiled via `torch.utils.cpp_extension.CUDAExtension`.

```bash
cd methods/Adakv/csrc
python build.py install

# HeadKV uses an identical kernel — compile it too if needed:
cd methods/HeadKV/csrc
python build.py install
```

The `build.py` uses PyTorch's `CUDAExtension` to compile `csrc/cuda_api.cu` into the `tiny_api_cuda` module (the pip package is named `tiny_pkg`). The compiled `.so` exports `update_flatten_view`, imported as `from tiny_api_cuda import update_flatten_view`, and is installed into `site-packages`.



### MagicPIG

> **Note:** MagicPIG requires a CPU with AVX-512 support.

```bash
conda activate kv
cd methods/magicpig
bash install.sh
```

The install script performs the following steps:

1. Build and install the `sparse_attention` library (`library/sparse_attention/`)
2. Build and install the `lsh` library (`library/lsh/`)

### RetroInfer

```bash
conda activate kv
cd methods/retroinfer
bash install.sh
pip install minference==0.1.6.0
```

The install script performs the following steps:

1. Install a custom `flash-attention` from source (`git+https://github.com/Starmys/flash-attention.git@weighted`)
2. Clone NVIDIA CUTLASS (`library/cutlass/`)
3. Build and install the `retroinfer` library (`library/retroinfer/`)

