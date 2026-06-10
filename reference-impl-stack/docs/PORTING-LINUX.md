# PORTING-LINUX — running codescribe-train on Linux

Walkthrough for [`scripts/bootstrap_linux.sh`](../scripts/bootstrap_linux.sh).
See [`PORTING.md`](PORTING.md) for the shape that's shared across all
platform porters; this doc covers Linux-specific gotchas.

`bootstrap_linux.sh` rsyncs an existing rag index from a source host you've
already set up; it does not build the index from scratch and does not install
any cron jobs. To build a fresh index on a brand-new host, run the indexer
directly (`uv run --extra rag python -m codescribe_train.rag index`) and
schedule any nightly refresh yourself.

Tested on Ubuntu 24.04 (x86_64). Other distros and aarch64 should work;
the porter detects the host arch and warns on mismatches.

---

## 1. Prerequisites

Install once per host:

| Tool | How (Ubuntu/Debian) |
|---|---|
| Python 3.12 | `sudo apt install python3.12 python3.12-venv` (or via `pyenv` / `uv python install`) |
| `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Rust toolchain (for `claw` build) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh -s -- -y && source "$HOME/.cargo/env"` |
| Build basics | `sudo apt install build-essential cmake ninja-build git rsync ripgrep` |
| NVIDIA CUDA Toolkit 12.x | Follow [NVIDIA's instructions](https://developer.nvidia.com/cuda-downloads) for your distro — the porter assumes `/usr/local/cuda-12.X` |
| LAN reachability to source host | Both boxes on the same Ethernet/Wi-Fi LAN; if source is WSL2, port-forward on the Windows host per [`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host) |
| SSH key on the source host | Append `~/.ssh/id_ed25519_remote-host.pub` to `<source>:~/.ssh/authorized_keys` |

Verify:

```bash
python3.12 --version          # → Python 3.12.x
uv --version
cargo --version
cmake --version
nvidia-smi                    # GPU should appear with driver + compute_cap
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Short SSH probe to the source host's LAN IP (substitute your own).
# For a WSL2 source behind the Windows host, use the Windows host's
# LAN IP + the portproxy port (e.g. 2222).
ssh -p 2222 -i ~/.ssh/id_ed25519_remote-host remote-host@192.168.1.10 'echo ok; hostname'
```

Clone the repos:

```bash
mkdir -p ~/repos
git clone https://github.com/example-org/codescribe-train ~/repos/codescribe-train
git clone https://github.com/example-org/sample    ~/repos/sample   # for repo-grep MCP
cd ~/repos/codescribe-train
git submodule update --init --recursive vendor/claw-code vendor/llama.cpp
```

---

## 2. Build the native binaries (once)

### `claw` (Rust)

```bash
cd ~/repos/codescribe-train
bash scripts/build_claw_code.sh
```

Auto-detects `cargo` on PATH, builds release-mode workspace. Lands at
`vendor/claw-code/rust/target/release/claw`.

### `llama-server` (C++ with CUDA)

`scripts/build_llama_cpp.sh` is idempotent and honours environment
overrides — pass your GPU's compute capability:

```bash
# Get your compute capability (e.g. 80 for an A100, 89 for an RTX 4090,
# 120 for a Blackwell-class card, 61 for a Pascal-class card).
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "compute_cap: $SM"

BUILD_JOBS=2 CMAKE_CUDA_ARCHITECTURES=$SM bash scripts/build_llama_cpp.sh
```

The default in the script is `CMAKE_CUDA_ARCHITECTURES=120` (a Blackwell-class
card, sm_120). Override for your hardware — e.g. `80` for an A100 (sm_80) or
`89` for an RTX 4090 (sm_89). Override `BUILD_JOBS` if you have more than ~8 GB
RAM — CUDA compilations are memory-heavy.

Verify:

```bash
file vendor/claw-code/rust/target/release/claw
# → ELF 64-bit LSB executable, x86-64 (...)
file vendor/llama.cpp/build/bin/llama-server
# → ELF 64-bit LSB executable, x86-64 (...)
ldd vendor/llama.cpp/build/bin/llama-server | grep -i cuda
# → libcuda.so.1 (...), libcudart.so.12 (...)
```

---

## 3. Run the porter

Substitute your source host's LAN IP and SSH port — for a WSL2 source
behind a Windows host portproxy ([`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host)),
the IP is the Windows host's LAN IP and the port is the portproxy
listen port (commonly 2222):

```bash
cd ~/repos/codescribe-train
bash scripts/bootstrap_linux.sh --from remote-host@192.168.1.10 --ssh-port 2222
```

What you'll see (abridged):

```
[bootstrap_linux] host: Linux/x86_64, user=youruser, repo=/home/youruser/repos/codescribe-train
[bootstrap_linux] uv:   /home/youruser/.local/bin/uv
[bootstrap_linux] rsync GGUF (~4.6 GB) ...
[bootstrap_linux] rsync rag.db ...
[bootstrap_linux] rsync embedder (~3.8 GB) ...
[bootstrap_linux] uv sync --extra rag
[bootstrap_linux] writing .../.claw/settings.json
[bootstrap_linux] detected NVIDIA GPU: NVIDIA GeForce RTX 4090 (compute_cap=89)
[bootstrap_linux] claw ok: .../target/release/claw
[bootstrap_linux] llama-server ok: .../bin/llama-server
[bootstrap_linux] smoke: import MCP server modules
[bootstrap_linux] smoke: rag store status
[bootstrap_linux] DONE.
```

If `nvidia-smi` isn't on PATH, the porter logs a warning and skips
CUDA-arch detection:

```
[bootstrap_linux] no nvidia-smi on PATH — llama-server will be CPU-only on this host (slow for 7B)
```

You can still run `llama-server` without `-ngl -1`; expect single-digit
tok/s for the 7B Q4_K_M.

### Re-running

```bash
# Re-render config + smoke after a manual change
bash scripts/bootstrap_linux.sh --skip-rsync

# Refresh artefacts from source host
bash scripts/bootstrap_linux.sh --from remote-host@192.168.1.10 --ssh-port 2222
```

---

## 4. Start `llama-server` (long-lived)

### Option A — terminal multiplexer

In a `tmux` or `screen` session:

```bash
cd ~/repos/codescribe-train
vendor/llama.cpp/build/bin/llama-server \
    -m checkpoints/sample-qwen7b-q4_k_m.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl -1 --ctx-size 32768
```

Drop `--no-mmap` unless VRAM is tight (an 8 GB-VRAM GPU needs it;
a 24 GB GPU does not).

### Option B — systemd user unit (recommended for long-lived)

Write a user-scope unit (per-host, not committed):

```ini
# ~/.config/systemd/user/llama-server.service
[Unit]
Description=codescribe-train local llama-server (fine-tuned Qwen2.5-Coder-7B)
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/%u/repos/codescribe-train
ExecStart=/home/%u/repos/codescribe-train/vendor/llama.cpp/build/bin/llama-server \
    -m /home/%u/repos/codescribe-train/checkpoints/sample-qwen7b-q4_k_m.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl -1 --ctx-size 32768
Restart=on-failure
RestartSec=5
StandardOutput=append:/home/%u/repos/codescribe-train/logs/llama-server.log
StandardError=append:/home/%u/repos/codescribe-train/logs/llama-server.err
# Hard lock to loopback at the network namespace level (defense-in-depth
# against accidental binds to 0.0.0.0).
PrivateNetwork=no

[Install]
WantedBy=default.target
```

Then:

```bash
mkdir -p ~/.config/systemd/user ~/repos/codescribe-train/logs
# Edit the unit above with your username replacing %u (or use systemd-run --user with --uid)
systemctl --user daemon-reload
systemctl --user enable --now llama-server.service
systemctl --user status llama-server.service
# Logs:
journalctl --user -fu llama-server.service
```

Survives logout if linger is enabled (`loginctl enable-linger $USER`).
On reboot, comes up after the user session.

### Avoid binding to LAN

`--host 0.0.0.0` would expose the model to your LAN — do not do that.
The strictly-local posture is loopback only; if you want a second host
to use the model, run the porter there too rather than exposing the
endpoint.

---

## 5. Launch claw and verify the model answers

```bash
cd ~/repos/codescribe-train
bash scripts/run-claw.sh
```

Banner:

```
[run-claw] LOCAL ONLY — strictly-local session, no external network egress
[run-claw]   model: openai/sample-qwen7b-local  →  http://127.0.0.1:8080  (loopback to llama-server)
[run-claw]   claw will print 'via openai' — that's its OpenAI-compat client, NOT api.openai.com
```

Test prompt that exercises the rag MCP:

> Show me a similar past issue to: NPE on order import

Watch `logs/rag-server.log` for the tool calls landing.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[bootstrap_linux] WARN: this script targets Linux. You're on Darwin.` | Ran on macOS | Use `scripts/bootstrap_mac.sh` |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | CUDA toolkit too old for Blackwell | Use CUDA Toolkit ≥ 12.8 (or pass `CMAKE_CUDA_ARCHITECTURES=89` for Ada/RTX 4090 etc.) |
| `bitsandbytes` import errors during `uv sync` | Trying to install the `train` extra, which is CUDA-pinned | `--extra rag` only (the porter already passes this); don't `--extra train` on a host without the training stack |
| `ssh: connect to host ... port 22: No route to host` | Not on the same LAN, or the source's `sshd` isn't reachable from your subnet | Both hosts must be on the same physical LAN. For a WSL2 source, the portproxy ([`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host)) must be installed on the Windows host and the firewall rule opened. |
| `ssh` to a WSL-portproxy port works but `rsync` exits with `Connection reset by peer` | WSL2 distro restarted and its internal `172.x.x.x` IP changed; the portproxy is now pointing at a stale address | On the Windows host, re-run `netsh interface portproxy set v4tov4 listenport=2222 listenaddress=0.0.0.0 connectaddress=<new-wsl-ip>`. |
| `claw mcp list` shows 0 servers | `.claw/settings.json` missing | Re-run `bootstrap_linux.sh --skip-rsync` |
| `find_similar_issues` returns nothing | `rag.db` empty or stale | `uv run --extra rag python -m codescribe_train.rag status` |
| `[bootstrap_linux] WARN: claw doesn't match host arch x86-64` | Binary built on a different machine / arch | Re-run `bash scripts/build_claw_code.sh` |
| `journalctl --user` returns nothing | Linger not enabled | `loginctl enable-linger $USER` |
| First MCP call takes 10+ s | bge-large cold-load to VRAM | Expected on first call; sub-second after |
| GPU not at 100% during inference | Built without CUDA, or `-ngl 0` | Rebuild with `CMAKE_CUDA_ARCHITECTURES=<sm>`; verify with `nvtop` or `nvidia-smi dmon` |
