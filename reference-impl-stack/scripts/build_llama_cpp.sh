#!/usr/bin/env bash
# scripts/build_llama_cpp.sh — idempotent build for the vendored llama.cpp.
#
# What it does:
#   1. Initialise + update vendor/llama.cpp submodule if it isn't checked out.
#   2. Build vendor/llama.cpp/build/bin/llama-server (and llama-quantize)
#      with CUDA enabled for your GPU's compute capability if the binary
#      is missing OR older than the submodule's HEAD commit.
#   3. Print the binary path on success.
#
# What it does NOT do:
#   - Pull / update the submodule SHA. The pinned commit is part of the
#     codescribe-train repository state — bumping it is a deliberate, reviewed
#     change, not a side effect of running this script.
#   - Install the CUDA toolkit. The script expects /usr/local/cuda-12.8/
#     to already be present (see scripts/setup_wsl.sh and
#     docs/HARDWARE-AND-PERFORMANCE.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_DIR="${REPO_ROOT}/vendor/llama.cpp"
BUILD_DIR="${SUBMODULE_DIR}/build"
BINARY="${BUILD_DIR}/bin/llama-server"

# CUDA toolkit location. Override via env if you have it installed
# elsewhere (e.g. /usr/local/cuda-13.0/).
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

# CMAKE_CUDA_ARCHITECTURES target — set this to match your GPU's compute
# capability. The default below (120 = Blackwell sm_120) is just one
# option; pick the value for your card. A few representative archs:
#   CMAKE_CUDA_ARCHITECTURES=80  ./scripts/build_llama_cpp.sh   # Ampere
#   CMAKE_CUDA_ARCHITECTURES=89  ./scripts/build_llama_cpp.sh   # Ada
#   CMAKE_CUDA_ARCHITECTURES=120 ./scripts/build_llama_cpp.sh   # Blackwell
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-120}"

# Parallel jobs. CUDA compilations are RAM-heavy, so low-RAM machines may
# OOM with -j$(nproc); pass a smaller -j. Default to 2 to fit; override if
# you have more RAM (or less time).
BUILD_JOBS="${BUILD_JOBS:-2}"

echo "[build_llama_cpp] repo root:  ${REPO_ROOT}"
echo "[build_llama_cpp] CUDA home:  ${CUDA_HOME}"
echo "[build_llama_cpp] CUDA arch:  sm_${CMAKE_CUDA_ARCHITECTURES}"
echo "[build_llama_cpp] jobs:       ${BUILD_JOBS}"

# 1. Ensure submodule is checked out.
if [[ ! -f "${SUBMODULE_DIR}/CMakeLists.txt" ]]; then
    if [[ ! -f "${REPO_ROOT}/.gitmodules" ]]; then
        echo "[build_llama_cpp] ERROR: vendor/llama.cpp is missing and .gitmodules is absent." >&2
        exit 1
    fi
    echo "[build_llama_cpp] initialising vendor/llama.cpp submodule..."
    git -C "${REPO_ROOT}" submodule update --init --recursive vendor/llama.cpp
fi

if [[ ! -d "${SUBMODULE_DIR}" ]]; then
    echo "[build_llama_cpp] ERROR: ${SUBMODULE_DIR} not found after submodule init." >&2
    exit 1
fi

# 2. Decide whether to (re)build.
NEEDS_BUILD=0
if [[ ! -x "${BINARY}" ]]; then
    NEEDS_BUILD=1
    echo "[build_llama_cpp] binary missing -> will build"
else
    SUBMODULE_HEAD_EPOCH="$(git -C "${SUBMODULE_DIR}" log -1 --format=%ct 2>/dev/null || echo 0)"
    BINARY_EPOCH="$(stat -c %Y "${BINARY}" 2>/dev/null || stat -f %m "${BINARY}" 2>/dev/null || echo 0)"
    if [[ "${BINARY_EPOCH}" -lt "${SUBMODULE_HEAD_EPOCH}" ]]; then
        NEEDS_BUILD=1
        echo "[build_llama_cpp] binary older than submodule HEAD -> will rebuild"
    fi
fi

if [[ "${NEEDS_BUILD}" -eq 0 ]]; then
    echo "[build_llama_cpp] up to date: ${BINARY}"
    exit 0
fi

# 3. Need cmake + a CUDA toolkit on PATH (or fallback to macOS CPU/Metal build).
if ! command -v cmake >/dev/null 2>&1; then
    echo "[build_llama_cpp] ERROR: cmake not on PATH. Install it first:" >&2
    echo "  brew install cmake" >&2
    exit 2
fi

# On macOS (Darwin) build without CUDA and enable Metal/Accelerate backends
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[build_llama_cpp] Detected macOS — building CPU/Metal backend (no CUDA)"
    CMAKE_EXTRA_FLAGS=("-DGGML_CUDA=OFF" "-DGGML_METAL=ON" "-DGGML_ACCELERATE=ON")
    # Do not touch CUDA env vars on macOS
else
    if [[ ! -d "${CUDA_HOME}" ]]; then
        echo "[build_llama_cpp] ERROR: CUDA toolkit not found at ${CUDA_HOME}." >&2
        echo "  Install cuda-toolkit-12.8 (or set CUDA_HOME to your install)." >&2
        echo "  See docs/HARDWARE-AND-PERFORMANCE.md for the WSL2 install instructions." >&2
        exit 2
    fi

    # Compose CUDA env. Required so cmake's CUDA detection picks the right
    # nvcc + libraries; otherwise it falls back to the WSL driver shim and
    # fails to compile the GPU kernels.
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    CMAKE_EXTRA_FLAGS=("-DGGML_CUDA=ON" "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

# Configure + build.
echo "[build_llama_cpp] cmake configure..."
cmake -S "${SUBMODULE_DIR}" -B "${BUILD_DIR}" \
    "${CMAKE_EXTRA_FLAGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

echo "[build_llama_cpp] cmake build (j=${BUILD_JOBS}) ..."
cmake --build "${BUILD_DIR}" --target llama-server llama-quantize -j"${BUILD_JOBS}"

if [[ ! -x "${BINARY}" ]]; then
    echo "[build_llama_cpp] ERROR: build finished but ${BINARY} is missing." >&2
    exit 3
fi

echo "[build_llama_cpp] OK: ${BINARY}"
