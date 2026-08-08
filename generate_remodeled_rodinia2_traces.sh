#!/usr/bin/env bash

set -e

# Usage:
#   ./generate_remodeled_rodinia2_traces.sh -D 0
#   ./generate_remodeled_rodinia2_traces.sh 0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE=0

if [ "${1:-}" = "-D" ]; then
    DEVICE="${2:-0}"
elif [ -n "${1:-}" ]; then
    DEVICE="$1"
fi

# Use CUDA_INSTALL_PATH if it is already set; otherwise find it from nvcc.
if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
    NVCC="$(command -v nvcc || true)"
    if [ -z "$NVCC" ]; then
        echo "ERROR: nvcc was not found. Set CUDA_INSTALL_PATH first." >&2
        exit 1
    fi
    export CUDA_INSTALL_PATH="$(cd "$(dirname "$NVCC")/.." && pwd)"
fi

export PATH="$CUDA_INSTALL_PATH/bin:$PATH"

if [ ! -x "$CUDA_INSTALL_PATH/bin/nvcc" ]; then
    echo "ERROR: $CUDA_INSTALL_PATH/bin/nvcc does not exist." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | grep -Fxq "$DEVICE"; then
        echo "ERROR: GPU $DEVICE was not found." >&2
        exit 1
    fi
fi

cd "$SCRIPT_DIR/simulator-remodeled"

echo "CUDA_INSTALL_PATH=$CUDA_INSTALL_PATH"
echo "GPU device=$DEVICE"

# Build the remodeled NVBit tracer.
if [ ! -d ./util/tracer_nvbit/nvbit_release ]; then
    ./util/tracer_nvbit/install_nvbit.sh
fi
make -C ./util/tracer_nvbit/

# Build Rodinia 2.0 and download its input data.
source ./gpu-app-collection/src/setup_environment
make -j"$(nproc)" -C ./gpu-app-collection/src rodinia_2.0-ft
make -C ./gpu-app-collection/src data

# Generate remodeled traces on the selected physical GPU.
./util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft -D "$DEVICE"

echo
echo "Done. Traces are under:"
echo "$SCRIPT_DIR/simulator-remodeled/hw_run/traces/"
