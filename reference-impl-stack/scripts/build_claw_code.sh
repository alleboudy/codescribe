#!/usr/bin/env bash
# scripts/build_claw_code.sh — idempotent build for the vendored Rust harness.
#
# What it does:
#   1. Initialise + update vendor/claw-code submodule if it isn't checked out.
#   2. Build target/release/claw if the binary is missing OR older than the
#      submodule's HEAD commit.
#   3. Print the binary path on success.
#
# What it does NOT do:
#   - Pull / update the submodule SHA. The pinned commit is part of the
#     codescribe-train repository state — bumping it is a deliberate, reviewed
#     change, not a side effect of running this script.
#   - Install the Rust toolchain. Run rustup yourself first:
#       curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#       source "$HOME/.cargo/env"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_DIR="${REPO_ROOT}/vendor/claw-code"
RUST_DIR="${SUBMODULE_DIR}/rust"
BINARY="${RUST_DIR}/target/release/claw"

echo "[build_claw_code] repo root: ${REPO_ROOT}"

# 1. Ensure submodule is checked out.
if [[ ! -f "${SUBMODULE_DIR}/Cargo.toml" && ! -f "${RUST_DIR}/Cargo.toml" ]]; then
    if [[ ! -f "${REPO_ROOT}/.gitmodules" ]]; then
        echo "[build_claw_code] ERROR: vendor/claw-code is missing and .gitmodules is absent." >&2
        exit 1
    fi
    echo "[build_claw_code] initialising vendor/claw-code submodule..."
    git -C "${REPO_ROOT}" submodule update --init --recursive vendor/claw-code
fi

if [[ ! -d "${RUST_DIR}" ]]; then
    echo "[build_claw_code] ERROR: ${RUST_DIR} not found after submodule init." >&2
    exit 1
fi

# 2. Decide whether to (re)build.
NEEDS_BUILD=0
if [[ ! -x "${BINARY}" ]]; then
    NEEDS_BUILD=1
    echo "[build_claw_code] binary missing -> will build"
else
    SUBMODULE_HEAD_EPOCH="$(git -C "${SUBMODULE_DIR}" log -1 --format=%ct 2>/dev/null || echo 0)"
    BINARY_EPOCH="$(stat -c %Y "${BINARY}" 2>/dev/null || stat -f %m "${BINARY}" 2>/dev/null || echo 0)"
    if [[ "${BINARY_EPOCH}" -lt "${SUBMODULE_HEAD_EPOCH}" ]]; then
        NEEDS_BUILD=1
        echo "[build_claw_code] binary older than submodule HEAD -> will rebuild"
    fi
fi

if [[ "${NEEDS_BUILD}" -eq 0 ]]; then
    echo "[build_claw_code] up to date: ${BINARY}"
    exit 0
fi

# 3. Need cargo on PATH.
if ! command -v cargo >/dev/null 2>&1; then
    echo "[build_claw_code] ERROR: cargo not on PATH. Install rustup first:" >&2
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" >&2
    echo "  source \"\$HOME/.cargo/env\"" >&2
    exit 2
fi

echo "[build_claw_code] cargo build --release --workspace (cwd: ${RUST_DIR})"
( cd "${RUST_DIR}" && cargo build --release --workspace )

if [[ ! -x "${BINARY}" ]]; then
    echo "[build_claw_code] ERROR: build finished but ${BINARY} is missing." >&2
    exit 3
fi

echo "[build_claw_code] OK: ${BINARY}"
