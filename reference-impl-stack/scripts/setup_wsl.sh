#!/usr/bin/env bash
# scripts/setup_wsl.sh — one-time WSL2 setup for codescribe-train.
#
# Idempotent. Run with:    bash scripts/setup_wsl.sh
# Then either `source ~/.bashrc` or open a fresh shell.
#
# What it does:
#   1. Add CUDA 12.8 to PATH and LD_LIBRARY_PATH in ~/.bashrc (idempotent —
#      a marker block is used, won't duplicate on re-run).
#   2. Print a verification.
#
# What it does NOT do (deliberately):
#   - Install CUDA toolkit. Use NVIDIA's WSL Ubuntu instructions:
#       wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
#       sudo dpkg -i cuda-keyring_1.1-1_all.deb
#       sudo apt-get update
#       sudo apt-get install -y cuda-toolkit-12-8
#   - Bump WSL2 RAM. That lives on the Windows host, not in WSL — edit
#     `%USERPROFILE%\.wslconfig` and `wsl --shutdown`. Recommended for phase 2:
#       [wsl2]
#       memory=24GB
#       swap=8GB
#     (You've explicitly chosen to skip this for now; keep it in mind if
#     training OOMs on CPU offload.)
#   - Install Rust toolchain. That's a phase-3 prereq; defer until then:
#       curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

set -euo pipefail

CUDA_HOME="/usr/local/cuda-12.8"
MARKER_BEGIN="# >>> codescribe-train cuda setup >>>"
MARKER_END="# <<< codescribe-train cuda setup <<<"
BASHRC="${HOME}/.bashrc"

echo "[setup_wsl] CUDA_HOME=${CUDA_HOME}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[setup_wsl] ERROR: ${CUDA_HOME}/bin/nvcc not found." >&2
    echo "  Install cuda-toolkit-12-8 first (see header comment)." >&2
    exit 1
fi

if grep -qF "${MARKER_BEGIN}" "${BASHRC}" 2>/dev/null; then
    echo "[setup_wsl] CUDA exports already present in ${BASHRC} — skipping."
else
    echo "[setup_wsl] Appending CUDA exports to ${BASHRC}..."
    cat >> "${BASHRC}" <<EOF

${MARKER_BEGIN}
# Added by codescribe-train/scripts/setup_wsl.sh — safe to remove this block.
export CUDA_HOME=${CUDA_HOME}
export PATH=\${CUDA_HOME}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}
${MARKER_END}
EOF
fi

echo
echo "[setup_wsl] Done. Verify with:"
echo "    source ~/.bashrc && nvcc --version"
echo
echo "[setup_wsl] Quick check (using explicit path, current shell unaffected):"
"${CUDA_HOME}/bin/nvcc" --version | head -4
