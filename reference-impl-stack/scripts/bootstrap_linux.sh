#!/usr/bin/env bash
# scripts/bootstrap_linux.sh — port codescribe-train to a Linux host.
#
# Companion: docs/PORTING-LINUX.md
#
# What this does (in order):
#   1. rsync the artefacts (GGUF, rag.db, bge embedder) from a source
#      host you've already set up.
#   2. uv sync --extra rag --extra dev
#   3. generate .claw/settings.json with absolute Linux paths
#   4. detect NVIDIA GPU compute capability via nvidia-smi; emit a
#      CUDA-arch-aware llama.cpp build hint
#   5. verify the vendored binaries are ELF matching the host arch
#   6. smoke-test MCP imports + rag store status.
#
# Targets x86_64 + aarch64 Linux. CUDA-capable GPU strongly recommended
# (Qwen 7B Q4_K_M will run on CPU only but at single-digit tok/s).
#
# Usage:
#   bash scripts/bootstrap_linux.sh --from <ssh-target> [options]
#
# Example (LAN, WSL-hosted source behind a Windows host portproxy):
#   bash scripts/bootstrap_linux.sh --from remote-host@192.168.1.10 --ssh-port 2222
#
# Options:
#   --from <user@host>          Source SSH target (e.g. remote-host@192.168.1.10).
#                               Required unless --skip-rsync.
#   --ssh-key <path>            SSH private key (default: ~/.ssh/id_ed25519_remote-host)
#   --ssh-port <port>           SSH port on source (default: 22; use for LAN port-forward bypasses)
#   --source-repo <path>        codescribe-train path on source (default: ~/repos/codescribe-train)
#   --source-hf-models <path>   HF model cache on source (default: ~/.hf-models)
#   --skip-rsync                Skip artefact transfer; assume already present locally
#   --skip-smoke                Skip the smoke tests at the end
#   -h, --help                  Print this help and exit
#
# Strictly local: only network egress is SSH/rsync to <source-host>,
# a box you own on the same LAN. See docs/STRICTLY-LOCAL-POSTURE.md
# and docs/PORTING.md §4 for transport details + portproxy setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORTER_TAG="bootstrap_linux"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
PORTER_USAGE_FN=usage

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_porter_lib.sh"

porter_init_args "$@"
porter_find_uv

cd "$REPO_ROOT"
porter_log "host: $(uname -s)/$(uname -m), user=${USER:-$(id -un)}, repo=${REPO_ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[bootstrap_linux] WARN: this script targets Linux. You're on $(uname -s)." >&2
    echo "  macOS:   scripts/bootstrap_mac.sh" >&2
    echo "  Windows: scripts/bootstrap_windows.ps1" >&2
fi

porter_rsync_artefacts

porter_log "uv sync --extra rag --extra dev"
"${UV}" sync --extra rag --extra dev   # match bootstrap_linux.sh — keep dev tools alongside rag

porter_render_claw_settings

# Linux-specific arch + NVIDIA detection
case "$(uname -m)" in
    x86_64|amd64)   arch_pattern='x86-64|x86_64';;
    aarch64|arm64)  arch_pattern='aarch64|arm64';;
    *)              arch_pattern='.*';;
esac

NVIDIA_GPU=""
CUDA_ARCH=""
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    # compute_cap returns like "8.0" or "12.0" — strip the dot for CMake.
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
    if [[ -n "$NVIDIA_GPU" ]]; then
        porter_log "detected NVIDIA GPU: ${NVIDIA_GPU} (compute_cap=${CUDA_ARCH:-?})"
    fi
else
    porter_log "no nvidia-smi on PATH — llama-server will be CPU-only on this host (slow for 7B)"
fi

if [[ -n "$CUDA_ARCH" ]]; then
    CUDA_HINT="BUILD_JOBS=2 CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} bash scripts/build_llama_cpp.sh"
else
    CUDA_HINT="(install NVIDIA CUDA toolkit first, then) BUILD_JOBS=2 CMAKE_CUDA_ARCHITECTURES=<your_sm> bash scripts/build_llama_cpp.sh"
fi
CARGO_HINT='bash scripts/build_claw_code.sh'

porter_check_binary "claw"         "${REPO_ROOT}/vendor/claw-code/rust/target/release/claw" "$arch_pattern" "$CARGO_HINT" || true
porter_check_binary "llama-server" "${REPO_ROOT}/vendor/llama.cpp/build/bin/llama-server"   "$arch_pattern" "$CUDA_HINT" || true

porter_smoke_tests

porter_log "DONE."
porter_log ""
porter_log "Next steps to chat with the fine-tuned model on this Linux host:"
porter_log "  1. Start llama-server in a separate terminal (long-lived, CUDA-accelerated):"
porter_log "       ${REPO_ROOT}/vendor/llama.cpp/build/bin/llama-server \\"
porter_log "         -m ${REPO_ROOT}/checkpoints/sample-qwen7b-q4_k_m.gguf \\"
porter_log "         --host 127.0.0.1 --port 8080 -ngl -1 --ctx-size 32768"
porter_log "     (For long-lived service, see docs/PORTING-LINUX.md §systemd.)"
porter_log "  2. Launch claw against it from this repo root:"
porter_log "       bash ${REPO_ROOT}/scripts/run-claw.sh"
