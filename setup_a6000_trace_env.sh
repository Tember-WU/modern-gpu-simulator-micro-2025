#!/usr/bin/env bash

# Prepare the CUDA 12.8/GCC 11 environment used by the remodeled A6000 tracer.
#
# Recommended usage (run this as the normal login user):
#   source ./setup_a6000_trace_env.sh
#
# System packages are installed through sudo when necessary. Sourcing is
# important because PATH and the other exports must remain in the current shell.

_a6000_is_sourced() {
    [ "${BASH_SOURCE[0]}" != "$0" ]
}

_a6000_error() {
    echo "ERROR: $*" >&2
    return 1
}

_a6000_run_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        _a6000_error "'$1' requires root privileges, but sudo is not installed."
    fi
}

_a6000_setup() {
    local script_dir cuda_dir cuda_repo_os cuda_repo_url cuda_keyring_deb
    local toolchain_bin gpu_info package
    local ID VERSION_ID PRETTY_NAME
    local -a packages missing_packages

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    cuda_dir="${CUDA_DIR:-/usr/local/cuda-12.8}"

    if [ ! -f /etc/os-release ]; then
        _a6000_error "/etc/os-release was not found."
        return 1
    fi

    # shellcheck disable=SC1091
    source /etc/os-release
    case "${ID:-}:${VERSION_ID:-}" in
        ubuntu:20.04|ubuntu:22.04|ubuntu:24.04) ;;
        *)
            _a6000_error "supported systems are Ubuntu 20.04, 22.04, and 24.04; detected ${PRETTY_NAME:-unknown}."
            return 1
            ;;
    esac

    if [ "$(uname -m)" != "x86_64" ]; then
        _a6000_error "the CUDA 12.8 repository configured here requires x86_64."
        return 1
    fi

    packages=(
        gcc-11 g++-11 build-essential bc \
        wget ca-certificates gnupg \
        git xutils-dev bison flex zlib1g-dev \
        libglu1-mesa-dev libssl-dev libxml2-dev libboost-all-dev \
        protobuf-compiler libprotobuf-dev \
        python3 python3-pip python3-yaml python3-psutil
    )
    missing_packages=()
    for package in "${packages[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
            missing_packages+=("$package")
        fi
    done

    if [ "${#missing_packages[@]}" -gt 0 ]; then
        echo "Installing missing build dependencies: ${missing_packages[*]}"
        _a6000_run_root env DEBIAN_FRONTEND=noninteractive apt-get update || return 1
        _a6000_run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            "${missing_packages[@]}" || return 1
    else
        echo "Build dependencies and GCC/G++ 11 are already installed."
    fi

    if [ ! -x "$cuda_dir/bin/nvcc" ]; then
        echo "Installing CUDA Toolkit 12.8 (the NVIDIA driver is left unchanged)..."
        cuda_repo_os="ubuntu${VERSION_ID//./}"
        cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo_os}/x86_64"
        cuda_keyring_deb="$(mktemp /tmp/cuda-keyring.XXXXXX.deb)" || return 1

        if ! wget -O "$cuda_keyring_deb" "$cuda_repo_url/cuda-keyring_1.1-1_all.deb"; then
            rm -f "$cuda_keyring_deb"
            return 1
        fi
        if ! _a6000_run_root dpkg -i "$cuda_keyring_deb"; then
            rm -f "$cuda_keyring_deb"
            return 1
        fi
        rm -f "$cuda_keyring_deb"

        _a6000_run_root env DEBIAN_FRONTEND=noninteractive apt-get update || return 1
        _a6000_run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8 || return 1
    fi

    if [ ! -x "$cuda_dir/bin/nvcc" ]; then
        _a6000_error "nvcc was not found at $cuda_dir/bin/nvcc after installation."
        return 1
    fi

    # Project Makefiles invoke plain gcc/g++. Repo-local links select GCC 11
    # without changing the machine-wide compiler alternatives.
    toolchain_bin="$script_dir/.toolchain/gcc-11/bin"
    mkdir -p "$toolchain_bin" || return 1
    ln -sfn /usr/bin/gcc-11 "$toolchain_bin/gcc" || return 1
    ln -sfn /usr/bin/g++-11 "$toolchain_bin/g++" || return 1
    ln -sfn /usr/bin/gcc-11 "$toolchain_bin/cc" || return 1
    ln -sfn /usr/bin/g++-11 "$toolchain_bin/c++" || return 1

    export CUDA_HOME="$cuda_dir"
    export CUDA_INSTALL_PATH="$cuda_dir"
    export CUDA_PATH="$cuda_dir"
    export PATH="$toolchain_bin:$cuda_dir/bin:$PATH"
    export LD_LIBRARY_PATH="$cuda_dir/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CC=gcc
    export CXX=g++
    export CUDAHOSTCXX="$toolchain_bin/g++"
    export ARCH=sm_86

    echo
    echo "Environment check"
    echo "-----------------"
    echo "CUDA_INSTALL_PATH=$CUDA_INSTALL_PATH"
    echo "ARCH=$ARCH"
    echo "gcc=$(command -v gcc)"
    echo "g++=$(command -v g++)"
    echo "nvcc=$(command -v nvcc)"
    gcc --version | sed -n '1p' || return 1
    g++ --version | sed -n '1p' || return 1
    nvcc --version | grep -E 'release|Build' || return 1

    if ! nvcc --list-gpu-code | grep -qx 'sm_86'; then
        _a6000_error "nvcc does not report SM 8.6 support."
        return 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo
        echo "GPU information"
        echo "---------------"
        gpu_info="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total \
            --format=csv,noheader 2>&1)"
        if [ $? -eq 0 ]; then
            echo "$gpu_info"
            if ! echo "$gpu_info" | grep -Eq '(^|, )[[:space:]]*8\.6([,[:space:]]|$)'; then
                echo "WARNING: the detected GPU is not compute capability 8.6; ARCH=sm_86 is intended for RTX A6000." >&2
            fi
        else
            echo "WARNING: nvidia-smi could not query a GPU: $gpu_info" >&2
        fi
    else
        echo "WARNING: nvidia-smi is unavailable; the CUDA toolkit is configured, but GPU/driver access was not verified." >&2
    fi

    echo
    echo "CUDA 12.8 + GCC/G++ 11 environment is ready for the remodeled A6000 tracer."
}

if _a6000_is_sourced; then
    _a6000_setup "$@"
    _a6000_status=$?
    if [ "$_a6000_status" -eq 0 ]; then
        echo "Environment variables are active in the current shell."
    fi
    unset -f _a6000_setup _a6000_run_root _a6000_error _a6000_is_sourced
    return "$_a6000_status"
else
    _a6000_setup "$@"
    _a6000_status=$?
    if [ "$_a6000_status" -eq 0 ]; then
        echo
        echo "NOTE: this script was executed, so its exported variables cannot persist."
        echo "Activate them in your current shell with:"
        echo "  source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    fi
    exit "$_a6000_status"
fi
