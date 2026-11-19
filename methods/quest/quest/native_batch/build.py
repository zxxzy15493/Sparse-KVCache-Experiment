"""Offline CMake build for the native batch extension, using the active environment."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


QUEST_ROOT = Path(__file__).resolve().parents[2]
SOURCE = QUEST_ROOT / 'quest/ops/native_batch'


def build_directory():
    return Path(os.environ.get('QUEST_NATIVE_BATCH_BUILD_DIR', QUEST_ROOT / 'kernels/batchs/native_batch'))


def fingerprint():
    digest = hashlib.sha256()
    for path in sorted(SOURCE.rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(SOURCE)).encode())
            digest.update(path.read_bytes())
    digest.update(f'{torch.__version__}|{torch.version.cuda}|{sys.version_info[:2]}'.encode())
    return digest.hexdigest()


def _build(force=False):
    directory = build_directory()
    library = directory / '_kernels_native_batch.so'
    manifest = directory / 'build_manifest.json'
    signature = fingerprint()
    requested_arch = os.environ.get('QUEST_NATIVE_CUDA_ARCHS')
    if requested_arch is None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        requested_arch = f'{major}{minor}'
    if not force and library.exists() and manifest.exists():
        try:
            cached = json.loads(manifest.read_text())
        except (ValueError, OSError):
            cached = {}
        cached_archs = set(cached.get('architectures', '').split(';'))
        if cached.get('fingerprint') == signature and (not requested_arch or
                set(requested_arch.split(';')).issubset(cached_archs)):
            return library
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError('Set CUDA_HOME to a CUDA toolkit compatible with PyTorch')
    nvcc = Path(CUDA_HOME) / 'bin/nvcc'
    version = subprocess.check_output([str(nvcc), '--version'], text=True)
    if f'release {torch.version.cuda},' not in version:
        raise RuntimeError(f'Set CUDA_HOME to CUDA {torch.version.cuda}, matching the active PyTorch')
    arch = requested_arch
    if not arch:
        if not torch.cuda.is_available():
            raise RuntimeError('Set QUEST_NATIVE_CUDA_ARCHS when building without a visible GPU')
        major, minor = torch.cuda.get_device_capability()
        arch = f'{major}{minor}'
    directory.mkdir(parents=True, exist_ok=True)
    quest_root = os.environ.get('QUEST_NATIVE_SOURCE_ROOT', str(QUEST_ROOT))
    command = ['cmake', '-S', str(SOURCE), '-B', str(directory), '-GNinja',
               f'-DQUEST_ROOT={quest_root}', f'-DPython_EXECUTABLE={sys.executable}',
               f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
               f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCUDAToolkit_ROOT={CUDA_HOME}',
               f'-DCMAKE_CUDA_ARCHITECTURES={arch}']
    build_env = os.environ.copy()
    build_env['TORCH_CUDA_ARCH_LIST'] = ';'.join(f'{int(a) // 10}.{int(a) % 10}' for a in arch.split(';'))
    subprocess.run(command, check=True, env=build_env)
    subprocess.run(['cmake', '--build', str(directory), '--parallel', os.environ.get('MAX_JOBS', '2')], check=True)
    manifest.write_text(json.dumps({'fingerprint': signature, 'torch': torch.__version__,
                                    'cuda': torch.version.cuda, 'architectures': arch}, indent=2))
    return library


def build(force=False):
    directory = build_directory()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build(force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    print(build(force=args.force))


if __name__ == '__main__':
    main()
