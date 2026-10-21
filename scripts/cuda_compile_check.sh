#!/usr/bin/env bash
# Compile-only check of cuda/*.cu with clang (no nvcc / GPU needed): syntax + PTX generation for sm_75.
# Needs: apt clang, and in ~/kvenv: pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
# This does NOT run the kernel; the real test is cuda/test_paged_attention.cu on a GPU (see cuda/GPU_CHECKLIST.md).
set -e
S=$HOME/kvenv/lib/python3.12/site-packages/nvidia
R=$HOME/fakecuda; rm -rf $R; mkdir -p $R/include $R/nvvm $R/bin $R/lib64
cp -rs $S/cuda_runtime/include/* $R/include/ 2>/dev/null || true
cp -rs $S/cuda_nvcc/include/* $R/include/ 2>/dev/null || true
cp -rs $S/cuda_cccl/include/* $R/include/ 2>/dev/null || true
cp -rs $S/cuda_nvcc/nvvm/* $R/nvvm/
cp -s $S/cuda_nvcc/bin/ptxas $R/bin/
: > $R/include/curand_mtgp32_kernel.h   # clang wrapper wants it; unused by our code
echo "CUDA Version 12.3.0" > $R/version.txt
echo "{\"cuda\": {\"name\": \"CUDA SDK\", \"version\": \"12.3.0\"}}" > $R/version.json
printf "#include <cuda_runtime.h>
extern \"C\" cudaError_t cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = 0);
" > $R/stub.h
cd /mnt/e/AAAAProject
for f in cuda/paged_attention.cu cuda/test_paged_attention.cu; do
  echo "== $f"
  clang++ -x cuda --cuda-path=$R --cuda-gpu-arch=sm_75 -nocudalib -std=c++17 -Iinclude -Icuda \
      -include $R/stub.h -Wno-unknown-cuda-version -fsyntax-only $f 2>&1 | head -30
  clang++ -x cuda --cuda-path=$R --cuda-gpu-arch=sm_75 --cuda-device-only -S -O2 -std=c++17 -Iinclude -Icuda -include $R/stub.h -Wno-unknown-cuda-version $f -o /tmp/$(basename $f).ptx 2>&1 | head -20; ls -la /tmp/$(basename $f).ptx 2>&1 | tail -1
done
