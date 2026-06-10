#!/usr/bin/env bash
# scripts/relaunch-claw.sh — one-shot recovery: kill stale MCP server /
# bridge processes from a previous claw session, then relaunch claw via
# scripts/run-claw.sh.
#
# When to use this:
#
#   1. You saw `Invalid JSON: ...Content-Length: 156\n` errors in an MCP
#      server log and just regenerated `.claw/settings.json` via the
#      porter (per docs/known-issues.md "MCP stdio framing mismatch").
#      The old claw session still has the broken MCP children alive;
#      they need to die before the new claw session can spawn fresh
#      ones wired through `_mcp_framing_bridge`.
#
#   2. You edited `.claw/settings.json` (or regenerated the MCP config)
#      and want a clean reload without trusting that closing claw cleanly
#      reaped every child.
#
#   3. You changed a Python MCP server's code and want claw to pick up
#      the new module rather than the cached one.
#
# This script DOES NOT touch llama-server (a long-lived process by
# project convention — see docs/STRICTLY-LOCAL-POSTURE.md). It only
# kills the per-claw-session MCP children.
#
# Strictly local: no network IO. All work is local process management
# plus exec'ing the existing run-claw.sh wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Patterns that match the MCP server / bridge processes spawned by
# claw. We pkill on the COMMAND LINE so we hit both the bridge process
# itself and any direct-server processes that might be hanging around
# from pre-bridge configurations.
PATTERNS=(
    'codescribe_train.servers._mcp_framing_bridge'
    'codescribe_train.servers.rag_server'
    'codescribe_train.servers.repo_grep'
)

log() { printf '[relaunch-claw] %s\n' "$*"; }

# Find any matching pids first so we can report what we'll kill.
pids=()
for pat in "${PATTERNS[@]}"; do
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f "$pat" 2>/dev/null || true)
done

# pgrep -f matches command lines, which will hit this very script
# (the PATTERNS array contains the module names verbatim) and our
# bash parent. Filter ourselves + parent + their entire ancestor
# chain to avoid suicide.
self_pid=$$
parent_pid=$PPID
filtered=()
if (( ${#pids[@]} > 0 )); then
    for pid in "${pids[@]}"; do
        [[ -z "$pid" ]] && continue
        [[ "$pid" == "$self_pid" ]] && continue
        [[ "$pid" == "$parent_pid" ]] && continue
        filtered+=("$pid")
    done
fi

if (( ${#filtered[@]} == 0 )); then
    log "no stale MCP server / bridge processes found"
else
    log "killing ${#filtered[@]} stale MCP process(es): ${filtered[*]}"
    # SIGTERM first; give them ~1 s to clean up; then SIGKILL anything
    # still alive. This matches the polite-then-firm pattern used by
    # systemd's KillSignal/SendSIGKILLAfter pair.
    kill "${filtered[@]}" 2>/dev/null || true
    sleep 1
    survivors=()
    for pid in "${filtered[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            survivors+=("$pid")
        fi
    done
    if (( ${#survivors[@]} > 0 )); then
        log "SIGKILL'ing ${#survivors[@]} survivor(s): ${survivors[*]}"
        kill -9 "${survivors[@]}" 2>/dev/null || true
    fi
fi

# Relaunch claw via the existing wrapper, which probes llama-server +
# wires the OpenAI-compat env vars + execs claw with --model. Pass any
# extra args we received straight through (`bash scripts/relaunch-claw.sh
# mcp list` will end up running `claw --model ... mcp list`, useful for
# verifying the new settings without entering the TUI).
exec bash "${SCRIPT_DIR}/run-claw.sh" "$@"
