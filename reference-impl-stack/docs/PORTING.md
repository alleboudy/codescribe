# PORTING — running codescribe-train on another box you own

This doc is the hub for the **per-platform porters** that take a host
from "freshly cloned git repo" to "claw chats with the fine-tuned model
and the RAG/grep MCP servers reply with real data," without rebuilding
the rag index from scratch on the new host.

The porters are designed for the scenario where you've already set up
one host (typically the GPU machine, the primary inference + training box) and
want to use the same fine-tuned model and rag store from a second host
you own — your Mac, your Linux desktop, or your Windows box on the same
local network.

For the **first-time** bootstrap of the rag store from scratch, run the indexer
directly: `python -m codescribe_train.rag index --bootstrap` (the per-host porters
skip this; they rsync the already-built `indices/rag.db`).

---

## 1. Strictly-local posture

The porters preserve the project's strictly-local posture
(see [`STRICTLY-LOCAL-POSTURE.md`](STRICTLY-LOCAL-POSTURE.md)):

- The only network egress is `ssh`/`rsync` between **two boxes you own
  on the same LAN**. No third party, no cloud storage, no overlay
  network, no public upload.
- The model weights, the rag index, and the embedder cache travel
  over plain SSH on the local network — bytes never cross the public
  internet at all.
- The target host's `llama-server` binds to `127.0.0.1` only — same as
  the primary host.
- No `huggingface-cli upload`, no W&B, no Sentry, no `git push` of
  weights to GitHub. Per the strictly-local posture: *"weights never
  leave the box"* — the porter respects that "the box" can be plural as
  long as every box is one you control and they're physically on the
  same LAN.

---

## 2. Choose your porter

| Target OS | Script | Walkthrough | Built-in build automation |
|---|---|---|---|
| macOS (Apple Silicon or Intel) | [`scripts/bootstrap_mac.sh`](../scripts/bootstrap_mac.sh) | [`PORTING-MAC.md`](PORTING-MAC.md) | No — Metal build hint printed if `llama-server` missing/mismatched |
| Linux (x86_64 or aarch64) | [`scripts/bootstrap_linux.sh`](../scripts/bootstrap_linux.sh) | [`PORTING-LINUX.md`](PORTING-LINUX.md) | Detects `nvidia-smi` compute_cap, emits a CUDA-arch-aware build hint |
| Windows 10+ (PowerShell 7) | [`scripts/bootstrap_windows.ps1`](../scripts/bootstrap_windows.ps1) | [`PORTING-WINDOWS.md`](PORTING-WINDOWS.md) | No — CUDA + Rust build hints printed if binaries missing |

The two bash porters share [`scripts/_porter_lib.sh`](../scripts/_porter_lib.sh)
so the rsync flow, the `.claw/settings.json` generator, the binary-arch
checker, and the smoke tests stay identical across Mac and Linux.

Windows is its own world (PowerShell, no native `rsync`, different
venv layout), so `bootstrap_windows.ps1` is self-contained.

---

## 3. What every porter transfers

Three artefacts move from `<source>` to the target host:

| Artefact | From | To | Size | Why it doesn't live in git |
|---|---|---|---|---|
| `sample-qwen7b-q4_k_m.gguf` | `<source-repo>/checkpoints/` | `<local-repo>/checkpoints/` | ~4.6 GB | Q4_K_M weights — exceeds GitHub's 100 MB file limit; LFS would cost bandwidth and break the no-upload rule. |
| `rag.db` | `<source-repo>/indices/` | `<local-repo>/indices/` | ~70 MB | sqlite-vec + FTS5 store, regeneratable from source; live data, not source. |
| `bge-large-en-v1.5/` | `<source>:~/.hf-models/` | `~/.hf-models/` | ~3.8 GB | Embedder model — pulled from HF Hub once, then mirrored across boxes to avoid re-downloading per host. |

Total: ~8.5 GB per port. Over gigabit Ethernet that's typically 1–3
minutes for the whole sync; over Wi-Fi expect 20–40 minutes at
~3–5 MB/s. The porters use `rsync`, so re-running after a partial
transfer resumes where it left off.

---

## 4. Transfer transport — LAN-only, over SSH

The porters call `rsync` over plain SSH on the local network. **No
overlay network** (no Tailscale, no VPN). Both hosts must be on the
same physical LAN and able to route to each other.

### 4.1 If the source is a normal Linux/macOS host

Find the source host's LAN IP, ensure its `sshd` is reachable from
the target host, and pass `--from <user>@<lan-ip>`:

```bash
# Example: rsync from a Linux desktop at 192.168.1.20
bash scripts/bootstrap_mac.sh --from remote-host@192.168.1.20
```

The porter uses port 22 by default. If the source's `sshd` is on a
different port, add `--ssh-port`:

```bash
bash scripts/bootstrap_mac.sh --from remote-host@192.168.1.20 --ssh-port 2222
```

### 4.2 If the source is a WSL2 distro on a Windows host

The WSL VM lives behind a NAT inside the Windows host — its
`172.x.x.x` address is **not reachable from your LAN** without help.
Forward a port on the Windows host to the WSL's `sshd`:

**On the Windows host (admin PowerShell, one-time):**

```powershell
# Find the WSL distro's internal IP first (from inside WSL):
#   ip -4 addr show eth0 | grep 'inet '
#   -> inet 172.24.80.1/20 (or whatever your WSL2 reports)
#
# Then forward a LAN-side port to the WSL's sshd:
netsh interface portproxy add v4tov4 `
    listenport=2222 listenaddress=0.0.0.0 `
    connectport=22 connectaddress=172.24.80.1

# Open the inbound firewall rule:
New-NetFirewallRule -DisplayName "WSL SSH 2222" `
    -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow

# Verify:
netsh interface portproxy show v4tov4
# Listen on ipv4:             Connect to ipv4:
# Address       Port        Address       Port
# ------------- ----------  ------------- ----------
# 0.0.0.0       2222        172.24.80.1  22
```

The portproxy survives Windows reboots (stored in the registry under
`HKLM\SYSTEM\CurrentControlSet\Services\PortProxy\v4tov4\tcp`). The
firewall rule does too. WSL2's internal IP can change across distro
restarts, though — if SSH stops resolving, re-check the WSL's
`ip addr show eth0` and `netsh ... set v4tov4` to update the
`connectaddress`.

**On the target host (Mac / Linux / Windows client):**

```bash
# Use the Windows host's LAN IP (NOT the WSL's internal 172.x.x.x).
# Find it from a normal PowerShell on the Windows host: `ipconfig`
bash scripts/bootstrap_mac.sh \
    --from remote-host@192.168.1.10 --ssh-port 2222
```

The `--ssh-port` (bash) / `-SshPort` (PowerShell) flag tells both the
SSH connectivity probe and the underlying `rsync -e ssh` to use that
port.

### 4.3 Verifying connectivity before running the porter

A small test before the multi-GB transfer:

```bash
# Short SSH probe
ssh -p <port> -i ~/.ssh/id_ed25519_remote-host <user>@<lan-ip> 'echo ok; hostname; date'

# Small file rsync (fast confidence check)
rsync -avh --progress -e "ssh -p <port> -i ~/.ssh/id_ed25519_remote-host" \
    <user>@<lan-ip>:/etc/hostname /tmp/probe
```

Both should complete in under a second on a healthy LAN. If they
don't, fix the network (firewall, IP, sshd) before invoking the
porter — the porter does the same probe internally and refuses to
launch if SSH won't authenticate.

### 4.4 Removing the Windows portproxy later

When the WSL-hosted source host is decommissioned, or you simply
don't need the LAN listener any more:

```powershell
netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
Remove-NetFirewallRule -DisplayName "WSL SSH 2222"
```

---

## 5. What every porter does NOT do

Deliberately scoped out of every porter:

- **Build the native binaries.** `vendor/claw-code/rust/target/release/claw`
  and `vendor/llama.cpp/build/bin/llama-server` are platform-specific
  (Metal on Apple Silicon, CUDA on Linux/Windows + NVIDIA). The porter
  verifies the existing binary matches the host arch and prints a build
  hint if not — building is left to the operator because each platform
  needs different toolchains.
- **Install long-lived services.** No cron, no systemd, no launchd, no
  Task Scheduler. Re-indexing is run manually (or via a cron entry you add
  yourself) on whichever host owns the rag store; other hosts read the
  rsync'd snapshot and aren't expected to keep re-indexing.
- **Clone the target repo** (`../your-repo`). The porter warns if
  `../your-repo` doesn't exist next to codescribe-train but doesn't clone it —
  cloning may need credentials the porter doesn't have, and you might
  rather rsync the working tree from the source host.

---

## 6. After the porter finishes

Every porter ends with a "Next steps" block that prints two commands:

1. **Start `llama-server`** on the target host (long-lived, in a
   separate terminal or as a host-native service — see the per-platform
   doc). Binds to `127.0.0.1:8080`.
2. **Launch claw** via [`scripts/run-claw.sh`](../scripts/run-claw.sh),
   which probes the local `llama-server`, wires the OpenAI-compat env
   vars, prints a loopback banner (so claw's "via openai" provider
   label can't be misread as external traffic — see
   [`commit 07465af`](https://github.com/example-org/codescribe-train/commit/07465af))
   and execs claw against the local model.

The Windows wrapper for step 2 is documented in
[`PORTING-WINDOWS.md`](PORTING-WINDOWS.md#claw-on-native-windows).

---

## 7. Re-running a porter

All three porters are idempotent. Common re-run modes:

| Goal | Flags |
|---|---|
| Re-verify config + smoke after a manual change | `--skip-rsync` / `-SkipRsync` |
| Refresh just the rag index from the GPU machine | `--skip-rsync` won't refresh; instead delete `indices/rag.db` and re-run without `--skip-rsync` |
| Refresh after a new fine-tune merged on the GPU machine | Re-run without flags — rsync sees the newer GGUF mtime and copies it |
| Smoke-test only (no transfer, no smoke skip) | `--skip-rsync` |

Re-running with `--skip-rsync` is the typical pattern after a manual
binary build or a config edit — the porter will only re-`uv sync`,
re-render `.claw/settings.json`, and re-run smoke tests.

---

## 8. Shared internals

`scripts/_porter_lib.sh` exposes:

| Function | Purpose |
|---|---|
| `porter_init_args "$@"` | Parse common args (`--from`, `--ssh-key`, etc.) |
| `porter_find_uv` | Locate `uv` binary or fail with install hint |
| `porter_rsync_artefacts` | The 3-way rsync (GGUF + rag.db + embedder) |
| `porter_render_claw_settings` | Write `.claw/settings.json` — the file claw actually reads (its config discovery merges `.claw.json` / `.claw/settings.json`, **not** `.claude.json`). Carries both the MCP server registry **and** `permissions.defaultMode: "dontAsk"`. `dontAsk` is claw's term for full tool access (incl. Bash) with no per-call prompts; without it claw defaults to a read-only-ish mode and the model reports "I don't have terminal access". For per-call approval use `"default"`; to allow edits but gate Bash use `"acceptEdits"`. Valid values: `default`/`plan`/`read-only`, `acceptEdits`/`auto`/`workspace-write`, `dontAsk`/`danger-full-access`. |
| `porter_check_binary <label> <path> <arch> <hint>` | Verify a binary matches host arch |
| `porter_smoke_tests` | Import MCP modules + `rag status` |

The PowerShell porter duplicates this surface in pwsh idioms; keeping
the two implementations identical in *behaviour* is the maintenance
contract — see the comment at the top of each platform porter.

### `_mcp_framing_bridge` — why every `.claw/settings.json` wraps the servers

claw-code's stdio MCP transport uses **LSP-style Content-Length
framing** (`Content-Length: N\r\n\r\n<payload>`), while the Python
MCP SDK (`mcp` 1.27.x) speaks **newline-delimited JSON-RPC** on
stdio. Wiring the server directly produces a stream of
`Invalid JSON: ...Content-Length: 156\n` errors from the server and
a stream of tool-call timeouts on claw's side.

`codescribe_train/servers/_mcp_framing_bridge.py` is a stdlib-only shim
spawned by claw that translates between the two framings. Every
`.claw/settings.json` generated by the porters wraps each MCP server
through this bridge:

```jsonc
{
  "command": ".../.venv/bin/python",
  "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.rag_server"]
}
```

If you write a `.claw/settings.json` by hand for a new MCP server,
**use the bridge** — without it claw → Python MCP server is broken.
Covered by `tests/servers/test_mcp_framing_bridge.py` (unit tests on
the framing helpers + an integration test that spawns the bridge
against a synthetic child).

---

## 9. Adding a new platform porter

If you need a fourth platform (e.g., FreeBSD, NixOS-specific, a Docker
image):

1. Mirror `bootstrap_mac.sh`'s shape and `source _porter_lib.sh` for
   the bash flow, OR mirror `bootstrap_windows.ps1` for a non-bash flow.
2. Write `docs/PORTING-<NAME>.md` from the existing per-platform docs.
3. Add the script to `BASH_SCRIPTS` in
   [`tests/test_scripts_shape.py`](../tests/test_scripts_shape.py) only
   if it has a `#!/usr/bin/env bash` shebang and `+x` bit — sourced
   libraries and non-bash scripts stay out of that tuple by design.
4. Add a row to the table in §2 above.

The porter does not need to learn to build native binaries — printing a
build hint is the project's deliberate compromise between "one
universal one-execution flow" and "we have to ship four different
compiler toolchains."
