#!/usr/bin/env bash
# scripts/_porter_lib.sh — shared bash helpers for bootstrap_mac.sh and
# bootstrap_linux.sh. Not a standalone script — `source` it from a
# platform porter. The leading underscore marks it as internal.
#
# Transport: rsync over SSH on the local LAN. For WSL-hosted sources,
# the Windows host port-forwards a LAN port to the WSL distro's sshd;
# pass --ssh-port to use the host's listen port. See docs/PORTING.md §4.
#
# Functions provided:
#   porter_init_args            parse common args (--from, --ssh-key, ...)
#   porter_find_uv              locate uv binary or fail with install hint
#   porter_rsync_artefacts      sync GGUF + rag.db + embedder from source
#   porter_render_claw_settings write .claw/settings.json with host paths
#   porter_check_binary         verify a binary matches host arch
#   porter_smoke_tests          import MCP modules + rag status
#
# Callers must set the following before calling the functions:
#   REPO_ROOT       absolute path to the codescribe-train repo on this host
#   PORTER_TAG      short label used in log lines (e.g. "bootstrap_mac")
#   PORTER_USAGE_FN name of a function that prints the porter's --help
#
# After porter_init_args runs, these are set:
#   FROM SSH_KEY SOURCE_REPO SOURCE_HF_MODELS SKIP_RSYNC SKIP_SMOKE
# After porter_find_uv runs, this is set:
#   UV

set -euo pipefail

porter_log() {
    printf '[%s] %s\n' "${PORTER_TAG:-bootstrap}" "$*"
}

porter_init_args() {
    FROM=""
    SSH_KEY="${HOME}/.ssh/id_ed25519_remote-host"
    SSH_PORT=22
    SOURCE_REPO="~/repos/codescribe-train"
    SOURCE_HF_MODELS="~/.hf-models"
    SKIP_RSYNC=0
    SKIP_SMOKE=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from)              FROM="$2"; shift 2;;
            --ssh-key)           SSH_KEY="$2"; shift 2;;
            --ssh-port)          SSH_PORT="$2"; shift 2;;
            --source-repo)       SOURCE_REPO="$2"; shift 2;;
            --source-hf-models)  SOURCE_HF_MODELS="$2"; shift 2;;
            --skip-rsync)        SKIP_RSYNC=1; shift;;
            --skip-smoke)        SKIP_SMOKE=1; shift;;
            -h|--help)           "${PORTER_USAGE_FN:-true}"; exit 0;;
            *) echo "[${PORTER_TAG:-bootstrap}] ERROR: unknown arg: $1" >&2; exit 1;;
        esac
    done
}

porter_find_uv() {
    UV="$(command -v uv || true)"
    if [[ -z "$UV" ]]; then
        for candidate in "${HOME}/.local/bin/uv" "/opt/homebrew/bin/uv" "/usr/local/bin/uv"; do
            if [[ -x "$candidate" ]]; then UV="$candidate"; break; fi
        done
    fi
    if [[ -z "$UV" ]]; then
        echo "[${PORTER_TAG:-bootstrap}] ERROR: uv not on PATH. Install: https://docs.astral.sh/uv/" >&2
        exit 2
    fi
}

porter_rsync_artefacts() {
    if [[ $SKIP_RSYNC -eq 1 ]]; then
        porter_log "skipping artefact rsync (--skip-rsync)"
        return 0
    fi
    if [[ -z "$FROM" ]]; then
        echo "[${PORTER_TAG:-bootstrap}] ERROR: --from is required (or pass --skip-rsync to reuse local artefacts)" >&2
        exit 1
    fi
    if ! ssh -p "$SSH_PORT" -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$FROM" 'true' 2>/dev/null; then
        echo "[${PORTER_TAG:-bootstrap}] ERROR: cannot SSH to ${FROM}:${SSH_PORT} with key ${SSH_KEY}" >&2
        echo "  Verify with: ssh -p ${SSH_PORT} -i ${SSH_KEY} ${FROM} 'echo ok'" >&2
        exit 3
    fi

    mkdir -p "${REPO_ROOT}/checkpoints" "${REPO_ROOT}/indices" "${HOME}/.hf-models"

    porter_log "rsync GGUF (~4.6 GB)  ${FROM}:${SOURCE_REPO}/checkpoints/sample-qwen7b-q4_k_m.gguf"
    rsync -avh --progress -e "ssh -p ${SSH_PORT} -i ${SSH_KEY}" \
        "${FROM}:${SOURCE_REPO}/checkpoints/sample-qwen7b-q4_k_m.gguf" \
        "${REPO_ROOT}/checkpoints/"

    porter_log "rsync rag.db          ${FROM}:${SOURCE_REPO}/indices/rag.db"
    rsync -avh --progress -e "ssh -p ${SSH_PORT} -i ${SSH_KEY}" \
        "${FROM}:${SOURCE_REPO}/indices/rag.db" \
        "${REPO_ROOT}/indices/"

    porter_log "rsync embedder (~3.8 GB)  ${FROM}:${SOURCE_HF_MODELS}/bge-large-en-v1.5/"
    rsync -avh --progress -e "ssh -p ${SSH_PORT} -i ${SSH_KEY}" \
        "${FROM}:${SOURCE_HF_MODELS}/bge-large-en-v1.5/" \
        "${HOME}/.hf-models/bge-large-en-v1.5/"
}

porter_render_claw_settings() {
    local py="${REPO_ROOT}/.venv/bin/python"
    local sample_repo
    sample_repo="$(cd "${REPO_ROOT}/.." && pwd)/sample"

    mkdir -p "${REPO_ROOT}/.claw" "${REPO_ROOT}/logs"
    local claw_json="${REPO_ROOT}/.claw/settings.json"
    porter_log "writing ${claw_json} (gitignored, per-host; absolute python avoids PATH surprises)"

    # claw speaks LSP-framed stdio MCP; mcp 1.27.x's stdio_server speaks
    # newline-delimited. We wrap each server through _mcp_framing_bridge,
    # which translates between the two. See module docstring for details.
    #
    # permissions.defaultMode = "dontAsk" — claw's own term (parses to
    # DangerFullAccess in vendor/claw-code .../config.rs::parse_permission_mode_label)
    # meaning "no per-tool prompts, full tool access including Bash". This
    # is the file claw actually reads for the permission gate — claw's
    # config discovery merges .claw.json / .claw/settings.json /
    # settings.local.json, NOT .claude.json. To require interactive
    # approval instead, change "dontAsk" to "default"; to allow edits but
    # gate the rest, use "acceptEdits".
    cat > "${claw_json}" <<EOF
{
  "permissions": {
    "defaultMode": "dontAsk"
  },
  "mcpServers": {
    "repo-rag": {
      "command": "${py}",
      "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.rag_server"],
      "env": {
        "RAG_DB_PATH": "${REPO_ROOT}/indices/rag.db",
        "RAG_LOG_PATH": "${REPO_ROOT}/logs/rag-server.log"
      }
    },
    "repo-grep": {
      "command": "${py}",
      "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.repo_grep"],
      "env": {
        "TARGET_REPO": "${sample_repo}"
      }
    },
    "repo-docs": {
      "command": "${py}",
      "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.repo_docs"],
      "env": {
        "TARGET_REPO": "${sample_repo}"
      }
    }
  }
}
EOF

    if [[ ! -d "$sample_repo" ]]; then
        porter_log "WARN: ${sample_repo} does not exist; repo-grep MCP server will fail until you clone or rsync the target repo there."
    fi
}

porter_check_binary() {
    local label="$1" bin="$2" arch_pattern="$3" build_hint="$4"
    if [[ ! -x "$bin" ]]; then
        echo "[${PORTER_TAG:-bootstrap}] WARN: ${label} not built at ${bin}" >&2
        echo "  build hint: ${build_hint}" >&2
        return 1
    fi
    if ! file "$bin" 2>/dev/null | grep -qE "$arch_pattern"; then
        echo "[${PORTER_TAG:-bootstrap}] WARN: ${label} at ${bin} doesn't match expected arch pattern '${arch_pattern}':" >&2
        file "$bin" >&2 || true
        return 1
    fi
    porter_log "${label} ok: ${bin}"
}

porter_smoke_tests() {
    if [[ $SKIP_SMOKE -eq 1 ]]; then
        porter_log "skipping smoke tests (--skip-smoke)"
        return 0
    fi
    local py="${REPO_ROOT}/.venv/bin/python"
    porter_log "smoke: import MCP server modules"
    if ! "${py}" -c "import codescribe_train.servers.rag_server, codescribe_train.servers.repo_grep" 2>&1; then
        echo "[${PORTER_TAG:-bootstrap}] ERROR: MCP server modules don't import — uv sync did not provide the rag extra" >&2
        exit 4
    fi
    porter_log "smoke: rag store status"
    if ! "${UV}" run --extra rag python -m codescribe_train.rag status 2>&1 | head -20; then
        echo "[${PORTER_TAG:-bootstrap}] ERROR: rag status failed — rag.db missing or unreadable" >&2
        exit 5
    fi
}
