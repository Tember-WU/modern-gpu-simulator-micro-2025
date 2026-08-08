#!/usr/bin/env bash

# Run as root. To keep the exports in the current shell, use:
#   source ./setup_a6000_trace_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="/usr/local/cuda-12.8"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run this script as root." >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f /etc/os-release ]; then
    echo "ERROR: /etc/os-release was not found." >&2
    return 1 2>/dev/null || exit 1
fi

source /etc/os-release
case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:20.04|ubuntu:22.04|ubuntu:24.04) ;;
    *)
        echo "ERROR: README supports Ubuntu 20.04, 22.04, or 24.04." >&2
        echo "Detected: ${PRETTY_NAME:-unknown}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

if [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: CUDA 12.8 repository setup in this script requires x86_64." >&2
    return 1 2>/dev/null || exit 1
fi

echo "Installing build dependencies and GCC/G++ 11..."
apt-get update
apt-get install -y --no-install-recommends \
    gcc-11 g++-11 build-essential bc \
    wget ca-certificates gnupg \
    git xutils-dev bison flex zlib1g-dev \
    libglu1-mesa-dev libssl-dev libxml2-dev libboost-all-dev \
    protobuf-compiler libprotobuf-dev \
    python3 python3-pip python3-yaml python3-psutil

if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    echo "Installing CUDA Toolkit 12.8 (the NVIDIA driver is not installed)..."
    CUDA_REPO_OS="ubuntu${VERSION_ID//./}"
    CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_OS}/x86_64"
    CUDA_KEYRING_DEB="/tmp/cuda-keyring-$$.deb"

    wget -O "$CUDA_KEYRING_DEB" "$CUDA_REPO_URL/cuda-keyring_1.1-1_all.deb"
    dpkg -i "$CUDA_KEYRING_DEB"
    rm -f "$CUDA_KEYRING_DEB"

    apt-get update
    apt-get install -y cuda-toolkit-12-8
fi

# The application Makefiles invoke plain `gcc` and `g++`. These local links
# make those names resolve to version 11 without changing the whole system.
TOOLCHAIN_BIN="$SCRIPT_DIR/.toolchain/gcc-11/bin"
mkdir -p "$TOOLCHAIN_BIN"
ln -sfn /usr/bin/gcc-11 "$TOOLCHAIN_BIN/gcc"
ln -sfn /usr/bin/g++-11 "$TOOLCHAIN_BIN/g++"
ln -sfn /usr/bin/gcc-11 "$TOOLCHAIN_BIN/cc"
ln -sfn /usr/bin/g++-11 "$TOOLCHAIN_BIN/c++"

export CUDA_HOME="$CUDA_DIR"
export CUDA_INSTALL_PATH="$CUDA_DIR"
export CUDA_PATH="$CUDA_DIR"
export PATH="$TOOLCHAIN_BIN:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=gcc
export CXX=g++
export CUDAHOSTCXX="$TOOLCHAIN_BIN/g++"
export ARCH=sm_86

echo
echo "Environment check"
echo "-----------------"
echo "CUDA_INSTALL_PATH=$CUDA_INSTALL_PATH"
echo "ARCH=$ARCH"
echo "gcc=$(command -v gcc)"
echo "g++=$(command -v g++)"
echo "nvcc=$(command -v nvcc)"
gcc --version | head -n 1
g++ --version | head -n 1
nvcc --version | grep -E 'release|Build'

if ! nvcc --list-gpu-code | grep -q '^sm_86$'; then
    echo "ERROR: nvcc does not report SM 8.6 support." >&2
    return 1 2>/dev/null || exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "GPU information"
    echo "---------------"
    nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total \
        --format=csv,noheader || true
fi

echo
echo "CUDA 12.8 + GCC/G++ 11 environment is ready for the remodeled A6000 tracer."
echo "If this script was executed instead of sourced, activate the variables with:"
echo "  source $SCRIPT_DIR/setup_a6000_trace_env.sh"
