#!/usr/bin/env bash
# scripts/run-claw.sh — launch the vendored claw harness against an
# already-running llama-server.
#
# Use this when `llama-server` is up serving the fine-tuned GGUF and
# you do NOT want the orchestrator (`python -m codescribe_train.cli run`) to
# spawn a second one and collide on the port. The orchestrator remains
# the right path from cold: see docs/CLAW-MCP.md.
#
# What this script wires that `.claw/settings.json` does not:
#   - OPENAI_BASE_URL      → the loopback OpenAI-compat endpoint
#   - OPENAI_API_KEY       → `local-no-auth` placeholder (server ignores it)
#   - ANTHROPIC_BASE_URL   → pinned to the same loopback, so a model-name
#                            typo cannot trigger an outbound Anthropic call
#                            (see codescribe_train/harness/claw.py:209-214)
#   - --model openai/<id>  → forces claw's OpenAI-compat client; setting
#                            OPENAI_MODEL alone falls back to Anthropic
#                            (see codescribe_train/harness/claw.py:196-204)
#
# Strictly local: no network egress. All endpoints are 127.0.0.1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${CODESCRIBE_PORT:-8080}"
MODEL="${CODESCRIBE_MODEL:-openai/sample-qwen7b-local}"
BACKEND_URL="http://127.0.0.1:${PORT}"
BINARY="vendor/claw-code/rust/target/release/claw"

if [[ ! -x "$BINARY" ]]; then
    echo "[run-claw] ERROR: $BINARY not built. Run: bash scripts/build_claw_code.sh" >&2
    exit 1
fi

if [[ ! -f ".claw/settings.json" ]]; then
    echo "[run-claw] ERROR: .claw/settings.json missing — MCP servers will not load." >&2
    echo "  Run: bash scripts/bootstrap_linux.sh" >&2
    exit 1
fi

if ! curl -sf -m 3 "${BACKEND_URL}/v1/models" >/dev/null 2>&1; then
    echo "[run-claw] ERROR: no OpenAI-compat server answering at ${BACKEND_URL}/v1/models." >&2
    echo "  Either start llama-server manually, or use the orchestrator from cold:" >&2
    echo "    python -m codescribe_train.cli run --repo ../your-repo \\" >&2
    echo "      --adapter checkpoints/sample-qwen7b-q4_k_m.gguf \\" >&2
    echo "      --backend llama-server --harness claw --port ${PORT}" >&2
    exit 2
fi

export OPENAI_BASE_URL="$BACKEND_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-no-auth}"
export ANTHROPIC_BASE_URL="$BACKEND_URL"

# Pre-flight banner: print loopback trail before claw's own banner loads,
# so a misread of claw's "via openai" provider label can't masquerade as
# external network. claw's banner is its OpenAI-compat HTTP client name,
# NOT a destination — destination is OPENAI_BASE_URL above.
echo "[run-claw] LOCAL ONLY — strictly-local session, no external network egress" >&2
echo "[run-claw]   model: ${MODEL}  →  ${BACKEND_URL}  (loopback to llama-server)" >&2
echo "[run-claw]   claw will print 'via openai' — that's its OpenAI-compat client, NOT api.openai.com" >&2

exec "$BINARY" --model "$MODEL" "$@"
