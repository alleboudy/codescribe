# PORTING-MAC — running codescribe-train on macOS

Walkthrough for [`scripts/bootstrap_mac.sh`](../scripts/bootstrap_mac.sh).
See [`PORTING.md`](PORTING.md) for the shape that's shared across all
platform porters; this doc covers Mac-specific gotchas.

Tested on Apple Silicon (M4). Intel Macs should work but you'll need to
build `llama-server` and `claw` for `x86_64` — the porter detects the
mismatch and warns.

---

## 1. Prerequisites

Install once per Mac:

| Tool | How |
|---|---|
| Homebrew | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| Python 3.12 | `brew install python@3.12` |
| `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Rust toolchain (for `claw` build) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh -s -- -y && source "$HOME/.cargo/env"` |
| CMake + Ninja (for `llama.cpp` build) | `brew install cmake ninja` |
| LAN reachability to source host | Both boxes on the same Wi-Fi or wired LAN; if source is WSL2, port-forward on the Windows host per [`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host) |
| SSH key on the source host | Copy `~/.ssh/id_ed25519_remote-host.pub` into `<source>:~/.ssh/authorized_keys` |

Verify:

```bash
brew --version
python3.12 --version          # → Python 3.12.x
uv --version
cargo --version
cmake --version
# Short SSH probe to the source host's LAN IP (substitute your own).
# For a WSL2 source behind the Windows host, use the Windows host's
# LAN IP + the portproxy port (e.g. 2222).
ssh -p 2222 -i ~/.ssh/id_ed25519_remote-host remote-host@192.168.1.10 'echo ok; hostname'
```

The repo itself should be cloned:

```bash
mkdir -p ~/repos
git clone https://github.com/example-org/codescribe-train ~/repos/codescribe-train
git clone https://github.com/example-org/sample    ~/repos/sample   # for repo-grep MCP
cd ~/repos/codescribe-train
git submodule update --init --recursive vendor/claw-code vendor/llama.cpp
```

---

## 2. Build the native binaries (once)

The porter does **not** build these — different platforms need different
flags. On Apple Silicon:

### `claw` (Rust, arm64 Mach-O)

```bash
cd ~/repos/codescribe-train
bash scripts/build_claw_code.sh
```

The script auto-detects `cargo` on PATH, builds release-mode workspace,
and lands the binary at `vendor/claw-code/rust/target/release/claw`.
First build is ~3–5 min on M4.

### `llama-server` (C++, arm64 Mach-O, Metal-accelerated)

`scripts/build_llama_cpp.sh` defaults to a CUDA build (`CMAKE_CUDA_ARCHITECTURES`
defaults to a recent arch; override it for your card on CUDA hosts) — that path
doesn't apply on Mac. Build by hand:

```bash
cd ~/repos/codescribe-train/vendor/llama.cpp
cmake -B build -DGGML_METAL=on -DLLAMA_BUILD_SERVER=ON
cmake --build build -j --target llama-server
```

This produces `build/bin/llama-server` linked against Apple's Metal
framework. First build is ~5–10 min on M4.

Verify both:

```bash
file vendor/claw-code/rust/target/release/claw
# → Mach-O 64-bit executable arm64
file vendor/llama.cpp/build/bin/llama-server
# → Mach-O 64-bit executable arm64
otool -L vendor/llama.cpp/build/bin/llama-server | grep -i metal
# → /System/Library/Frameworks/Metal.framework/Metal (...)
```

---

## 3. Run the porter

One-shot. Substitute your source host's LAN IP and SSH port — for a
WSL2 source behind a Windows host portproxy ([`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host)),
the IP is the Windows host's LAN IP and the port is the portproxy
listen port (commonly 2222):

```bash
cd ~/repos/codescribe-train
bash scripts/bootstrap_mac.sh --from remote-host@192.168.1.10 --ssh-port 2222
```

What you'll see (abridged):

```
[bootstrap_mac] host: Darwin/arm64, user=user, repo=$HOME/repos/codescribe-train
[bootstrap_mac] uv:   $HOME/.local/bin/uv
[bootstrap_mac] rsync GGUF (~4.6 GB)  remote-host@192.168.1.10:~/repos/codescribe-train/checkpoints/sample-qwen7b-q4_k_m.gguf
... 4.6 GB transferred ...
[bootstrap_mac] rsync rag.db          remote-host@192.168.1.10:~/repos/codescribe-train/indices/rag.db
... 70 MB transferred ...
[bootstrap_mac] rsync embedder (~3.8 GB)  remote-host@192.168.1.10:~/.hf-models/bge-large-en-v1.5/
... 3.8 GB transferred ...
[bootstrap_mac] uv sync --extra rag --extra dev
[bootstrap_mac] writing $HOME/repos/codescribe-train/.claw/settings.json (gitignored, per-host; ...)
[bootstrap_mac] claw ok: $HOME/repos/codescribe-train/vendor/claw-code/rust/target/release/claw
[bootstrap_mac] llama-server ok: $HOME/repos/codescribe-train/vendor/llama.cpp/build/bin/llama-server
[bootstrap_mac] smoke: import MCP server modules
[bootstrap_mac] smoke: rag store status
...
[bootstrap_mac] DONE.
```

Total time: ~2–3 min on gigabit Ethernet, ~20–40 min on Wi-Fi at
3–5 MB/s.

### Re-running

After a manual change (rebuilt a binary, edited `pyproject.toml`,
landed a new fine-tuned GGUF on the source):

```bash
# Fastest: skip rsync, just re-render config + smoke
bash scripts/bootstrap_mac.sh --skip-rsync

# Refresh artefacts from source host
bash scripts/bootstrap_mac.sh --from remote-host@192.168.1.10 --ssh-port 2222
```

---

## 4. Start `llama-server` (long-lived)

The porter does not start `llama-server` — you do, in a separate
terminal, so the process lifetime is decoupled from any shell session.

```bash
cd ~/repos/codescribe-train
vendor/llama.cpp/build/bin/llama-server \
    -m checkpoints/sample-qwen7b-q4_k_m.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl -1 --ctx-size 32768
```

Flag rationale on Apple Silicon:

| Flag | Why |
|---|---|
| `--host 127.0.0.1` | Loopback only. Do not change to `0.0.0.0` — would expose the model to the LAN. |
| `--port 8080` | Matches `scripts/run-claw.sh`'s default probe target. |
| `-ngl -1` | Offload all transformer layers to the GPU. On Metal, `-1` means "as many as fit"; for a Q4_K_M 7B with `--ctx-size 32768` on a 24 GB unified-memory M4, that's all of them. |
| `--ctx-size 32768` | Matches the source host's deploy. Drop to 8192 on Macs with <16 GB RAM. |

Drop `--no-mmap` (which an 8 GB-VRAM CUDA host needs for its VRAM constraint): on
Apple Silicon's unified memory, mmap-based sharing between CPU and GPU
is the fast path.

### Optional: launch as a background service via `launchd`

If you want `llama-server` to start at login and restart on crash,
write a launchd plist (not committed — per-host state). Example:

```xml
<!-- ~/Library/LaunchAgents/com.user.llama-server.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.user.llama-server</string>
  <key>ProgramArguments</key>
  <array>
    <string>$HOME/repos/codescribe-train/vendor/llama.cpp/build/bin/llama-server</string>
    <string>-m</string><string>$HOME/repos/codescribe-train/checkpoints/sample-qwen7b-q4_k_m.gguf</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8080</string>
    <string>-ngl</string><string>-1</string>
    <string>--ctx-size</string><string>32768</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/repos/codescribe-train/logs/llama-server.log</string>
  <key>StandardErrorPath</key><string>$HOME/repos/codescribe-train/logs/llama-server.err</string>
</dict>
</plist>
```

Load:

```bash
launchctl load ~/Library/LaunchAgents/com.user.llama-server.plist
launchctl list | grep llama-server
```

---

## 5. Launch claw and verify the model answers

In another terminal (or in the same one if `llama-server` is daemonised):

```bash
cd ~/repos/codescribe-train
bash scripts/run-claw.sh
```

You'll see the loopback banner before claw boots:

```
[run-claw] LOCAL ONLY — strictly-local session, no external network egress
[run-claw]   model: openai/sample-qwen7b-local  →  http://127.0.0.1:8080  (loopback to llama-server)
[run-claw]   claw will print 'via openai' — that's its OpenAI-compat client, NOT api.openai.com
```

Then claw's TUI loads. Try a query that exercises the rag MCP:

> Show me a similar past issue to: NPE on order import

The model should call `find_similar_issues`, which spawns the local
MCP server, which reads `indices/rag.db`, runs the bge-large embedder
(first call ~5–10 s as the model warms on Metal), and returns real
issue numbers from `example-org/sample`. Watch `logs/rag-server.log` to
see the tool calls land.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `rsync: command not found` | rsync not installed (very rare on macOS — system rsync 2.6 ships with macOS) | `brew install rsync` for a modern rsync 3.x |
| `ssh: connect to host ... port 22: Connection timed out` / `No route to host` | Not on the same LAN, or the source's `sshd` isn't reachable from your subnet | Both hosts must be on the same physical LAN. For a WSL2 source, the portproxy ([`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host)) must be installed on the Windows host and the firewall rule opened. |
| `ssh: connect to host ... port 2222` works but `rsync` exits with `Connection reset by peer` | WSL2 distro restarted and its internal `172.x.x.x` IP changed; the portproxy is now pointing at a stale address | On the Windows host, re-run `netsh interface portproxy set v4tov4 listenport=2222 listenaddress=0.0.0.0 connectaddress=<new-wsl-ip>`. |
| `[bootstrap_mac] WARN: llama-server doesn't match host arch arm64` | Binary built for x86_64 (e.g. cloned from a different machine) | Re-run the Metal build in §2 |
| `claw mcp list` shows 0 servers | `.claw/settings.json` missing or in the wrong cwd | Re-run the porter; verify `cat .claw/settings.json` parses |
| `find_similar_issues` returns nothing | `rag.db` empty or stale | `uv run --extra rag python -m codescribe_train.rag status` shows the watermark counts |
| First MCP call takes 10+ s | bge-large embedder cold-load on Metal | Expected on first call; subsequent calls are sub-second |
| `ImportError: sqlite_vec` from the MCP server | `.venv` missing the `rag` extra | `uv sync --extra rag` (or re-run the porter; it does this) |
| `OPENAI_API_KEY` is set by your shell rc and is a real key | The wrapper would propagate it, but `OPENAI_BASE_URL` is pinned to loopback so it goes nowhere | No-op risk on Mac; the wrapper already pins both base URLs |

If the model answers but generations are slow (< 5 tok/s on M4):

- Check `Activity Monitor → GPU` — `llama-server` should be saturating
  the GPU. If it isn't, you may have built without Metal; rebuild per §2.
- `--ctx-size 32768` with a long conversation can exhaust unified
  memory; drop to 8192 if you see swap.
