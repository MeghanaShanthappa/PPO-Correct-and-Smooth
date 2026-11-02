#!/usr/bin/env bash
export DGLBACKEND=pytorch
export PYTHONUTF8=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/rlcs-mpl
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUPY_CACHE_DIR=/var/tmp/mshanthappa/rl-cs/cupy-cache
export CUDA_PATH=/var/tmp/mshanthappa/rl-cs/grad-venv/lib/python3.10/site-packages/nvidia/cuda_runtime
export LD_LIBRARY_PATH=/var/tmp/mshanthappa/rl-cs/grad-venv/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
for rlcs_lib in /var/tmp/mshanthappa/rl-cs/grad-venv/lib/python3.10/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$rlcs_lib:$LD_LIBRARY_PATH"
done
