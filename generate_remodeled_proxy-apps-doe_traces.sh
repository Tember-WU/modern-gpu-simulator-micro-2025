#!/usr/bin/env bash

# Do not enable set -u: gpu-app-collection's setup script reads optional
# environment variables before defining all of them.
set -eo pipefail

# Usage:
#   ./generate_remodeled_proxy-apps-doe_traces.sh -D 0
#   ./generate_remodeled_proxy-apps-doe_traces.sh 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE=0

if [ "${1:-}" = "-D" ]; then
    DEVICE="${2:-0}"
elif [ -n "${1:-}" ]; then
    DEVICE="$1"
fi

if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
    NVCC="$(command -v nvcc || true)"
    if [ -z "$NVCC" ]; then
        echo "ERROR: nvcc was not found." >&2
        echo "Run: source $SCRIPT_DIR/setup_a6000_trace_env.sh" >&2
        exit 1
    fi
    export CUDA_INSTALL_PATH="$(cd "$(dirname "$NVCC")/.." && pwd)"
fi

export CUDA_HOME="${CUDA_HOME:-$CUDA_INSTALL_PATH}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_INSTALL_PATH}"
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export ARCH="${ARCH:-sm_86}"
export INTERMEDIATE_EXTRA_FILES_PERSISTANCE=1

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

echo "Benchmark suite=proxy-apps-doe"
echo "Build target=proxy-apps"
echo "CUDA_INSTALL_PATH=$CUDA_INSTALL_PATH"
echo "CUDA version=$("$CUDA_INSTALL_PATH/bin/nvcc" --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
echo "GPU device=$DEVICE"

if [ ! -d ./util/tracer_nvbit/nvbit_release ]; then
    ./util/tracer_nvbit/install_nvbit.sh
fi
make -C ./util/tracer_nvbit/

source ./gpu-app-collection/src/setup_environment
make -j"$(nproc)" -C ./gpu-app-collection/src proxy-apps
make -C ./gpu-app-collection/src data

./util/tracer_nvbit/run_hw_trace.py -B proxy-apps-doe -D "$DEVICE"

echo
echo "Done. Traces are under:"
echo "$SCRIPT_DIR/simulator-remodeled/hw_run/traces/"
echo "SASS for each workload is under its traces/extra_info/sass/ directory."
