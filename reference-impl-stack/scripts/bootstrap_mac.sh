#!/usr/bin/env bash
# scripts/bootstrap_mac.sh — port codescribe-train to a macOS host.
#
# Companion: docs/PORTING-MAC.md
#
# What this does (in order):
#   1. rsync the artefacts (GGUF, rag.db, bge embedder) from a source
#      host you've already set up.
#   2. uv sync --extra rag --extra dev
#   3. generate .claw/settings.json with absolute Mac paths
#   4. verify the vendored binaries are Mach-O matching the host arch;
#      print a Metal-aware build hint if not.
#   5. smoke-test MCP imports + rag store status.
#
# Built for Apple Silicon (arm64). Intel Macs work too — the script
# detects the host arch and warns if the binaries are mismatched.
#
# Usage:
#   bash scripts/bootstrap_mac.sh --from <ssh-target> [options]
#
# Example (LAN, WSL-hosted source behind a Windows host portproxy):
#   bash scripts/bootstrap_mac.sh --from remote-host@192.168.1.10 --ssh-port 2222
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
PORTER_TAG="bootstrap_mac"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
PORTER_USAGE_FN=usage

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_porter_lib.sh"

porter_init_args "$@"
porter_find_uv

cd "$REPO_ROOT"
porter_log "host: $(uname -s)/$(uname -m), user=${USER:-$(id -un)}, repo=${REPO_ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[bootstrap_mac] WARN: this script targets macOS. You're on $(uname -s)." >&2
    echo "  Linux:   scripts/bootstrap_linux.sh" >&2
    echo "  Windows: scripts/bootstrap_windows.ps1" >&2
fi

porter_rsync_artefacts

porter_log "uv sync --extra rag --extra dev"
"${UV}" sync --extra rag --extra dev   # match bootstrap_linux.sh — keep dev tools alongside rag

porter_render_claw_settings

# Mac-specific arch + build hints
case "$(uname -m)" in
    arm64)  arch_pattern='arm64';;
    x86_64) arch_pattern='x86_64';;
    *)      arch_pattern='.*';;
esac

METAL_HINT='cd vendor/llama.cpp && cmake -B build -DGGML_METAL=on -DLLAMA_BUILD_SERVER=ON && cmake --build build -j --target llama-server'
CARGO_HINT='bash scripts/build_claw_code.sh'

porter_check_binary "claw"         "${REPO_ROOT}/vendor/claw-code/rust/target/release/claw" "$arch_pattern" "$CARGO_HINT" || true
porter_check_binary "llama-server" "${REPO_ROOT}/vendor/llama.cpp/build/bin/llama-server"   "$arch_pattern" "$METAL_HINT" || true

porter_smoke_tests

porter_log "DONE."
porter_log ""
porter_log "Next steps to chat with the fine-tuned model on this Mac:"
porter_log "  1. Start llama-server in a separate terminal (long-lived, Metal-accelerated):"
porter_log "       ${REPO_ROOT}/vendor/llama.cpp/build/bin/llama-server \\"
porter_log "         -m ${REPO_ROOT}/checkpoints/sample-qwen7b-q4_k_m.gguf \\"
porter_log "         --host 127.0.0.1 --port 8080 -ngl -1 --ctx-size 32768"
porter_log "  2. Launch claw against it from this repo root:"
porter_log "       bash ${REPO_ROOT}/scripts/run-claw.sh"
