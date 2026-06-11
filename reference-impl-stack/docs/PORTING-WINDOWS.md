# PORTING-WINDOWS — running codescribe-train on Windows

Walkthrough for [`scripts/bootstrap_windows.ps1`](../scripts/bootstrap_windows.ps1).
See [`PORTING.md`](PORTING.md) for the shape that's shared across all
platform porters; this doc covers Windows-specific gotchas.

Two viable paths exist for Windows:

1. **Native Windows** (this doc, `bootstrap_windows.ps1`) — PowerShell 7+,
   native OpenSSH, Git-for-Windows `rsync`, native CUDA + MSVC builds.
2. **WSL2** — install Ubuntu or another WSL distro, then use
   [`bootstrap_linux.sh`](../scripts/bootstrap_linux.sh) inside the
   distro. A source host can run this way too. Simpler if you already use WSL
   for everything else; mainly limited by WSL2's CUDA / file-share
   performance.

The "Windows porter" is the native PowerShell one. WSL is just the
Linux porter inside a Linux distro.

Tested on Windows 11 23H2 with PowerShell 7.4 (`pwsh`) and Git for
Windows 2.45.

---

## 1. Prerequisites

| Tool | How |
|---|---|
| PowerShell 7+ (`pwsh`) | `winget install Microsoft.PowerShell` (not Windows PowerShell 5.1 — too old) |
| OpenSSH client | Built-in on Windows 10/11; `Get-Command ssh` should find `C:\Windows\System32\OpenSSH\ssh.exe` |
| `rsync` | Install [Git for Windows](https://git-scm.com/download/win) — bundles `rsync.exe` in Git Bash's `usr/bin`. Add that dir to PATH, or use Git Bash to invoke the script. MSYS2 (`pacman -S rsync`) also works. |
| Python 3.12 | `winget install Python.Python.3.12` |
| `uv` | `powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 \| iex"` (installs to `%USERPROFILE%\.local\bin\uv.exe`) |
| Rust toolchain (for `claw` build) | [rustup-init.exe](https://win.rustup.rs/x86_64) — pick the MSVC toolchain |
| Visual Studio 2022 Build Tools (for `llama.cpp` CUDA build) | `winget install Microsoft.VisualStudio.2022.BuildTools` with "Desktop development with C++" + Windows 10/11 SDK + Spectre-mitigated MSVC |
| NVIDIA CUDA Toolkit 12.x | [Download from NVIDIA](https://developer.nvidia.com/cuda-downloads); installer integrates with Visual Studio |
| CMake + Ninja | `winget install Kitware.CMake Ninja-build.Ninja` |
| LAN reachability to source host | Both boxes on the same Ethernet/Wi-Fi LAN; if source is WSL2, port-forward on the Windows host per [`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host) |
| SSH key on source host | Generate with `ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_remote-host`, copy `.pub` into `<source>:~/.ssh/authorized_keys` |

Verify (in `pwsh`):

```powershell
pwsh --version                    # → PowerShell 7.4.x or later
Get-Command ssh, rsync, uv, cargo, cmake | Format-Table Source
python --version                  # → Python 3.12.x
nvidia-smi                        # GPU + compute_cap
# Short SSH probe to the source host's LAN IP (substitute your own).
# For a WSL2 source behind a Windows host portproxy, use the Windows
# host's LAN IP + the portproxy port (e.g. 2222).
ssh -p 2222 -i $env:USERPROFILE\.ssh\id_ed25519_remote-host remote-host@192.168.1.10 'echo ok; hostname'
```

Clone the repos:

```powershell
mkdir $env:USERPROFILE\repos -Force
git clone https://github.com/example-org/codescribe-train $env:USERPROFILE\repos\codescribe-train
git clone https://github.com/example-org/sample    $env:USERPROFILE\repos\sample
cd $env:USERPROFILE\repos\codescribe-train
git submodule update --init --recursive vendor/claw-code vendor/llama.cpp
```

---

## 2. Build the native binaries (once)

### `claw` (Rust, x86_64 PE)

In an MSVC-aware shell (PowerShell after running `rustup default
stable-x86_64-pc-windows-msvc`):

```powershell
cd $env:USERPROFILE\repos\codescribe-train\vendor\claw-code\rust
cargo build --release --workspace
```

Output: `vendor\claw-code\rust\target\release\claw.exe`.

### `llama-server` (C++ with CUDA, x86_64 PE)

The Linux build script (`scripts/build_llama_cpp.sh`) is bash-only.
Build by hand from a "Developer PowerShell for VS 2022" so the MSVC
environment variables are set:

```powershell
# Get your GPU's compute capability (e.g. 89 for RTX 4090, 86 for RTX 3080).
$sm = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim('.')

cd $env:USERPROFILE\repos\codescribe-train\vendor\llama.cpp
cmake -B build `
    -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CUDA=on `
    -DCMAKE_CUDA_ARCHITECTURES=$sm `
    -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release -j --target llama-server
```

Output: `vendor\llama.cpp\build\bin\Release\llama-server.exe`.

(The `Release` subdirectory is MSVC multi-config-generator convention;
the porter looks for the binary there.)

Verify:

```powershell
Get-ChildItem $env:USERPROFILE\repos\codescribe-train\vendor\claw-code\rust\target\release\claw.exe
Get-ChildItem $env:USERPROFILE\repos\codescribe-train\vendor\llama.cpp\build\bin\Release\llama-server.exe
# Sanity-check that CUDA is linked:
dumpbin /dependents `
    $env:USERPROFILE\repos\codescribe-train\vendor\llama.cpp\build\bin\Release\llama-server.exe `
    | Select-String -Pattern 'cudart|cuda'
```

---

## 3. Run the porter

In `pwsh` (not Windows PowerShell 5.1). Substitute your source host's
LAN IP and SSH port — for a WSL2 source behind a Windows host
portproxy ([`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host)),
the IP is the Windows host's LAN IP and the port is the portproxy
listen port (commonly 2222):

```powershell
cd $env:USERPROFILE\repos\codescribe-train
.\scripts\bootstrap_windows.ps1 -From remote-host@192.168.1.10 -SshPort 2222
```

If you get a script-execution-policy error, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

What you'll see (abridged):

```
[bootstrap_windows] host: Windows/AMD64, user=YourName, repo=C:\Users\YourName\repos\codescribe-train
[bootstrap_windows] uv: C:\Users\YourName\.local\bin\uv.exe
[bootstrap_windows] rsync GGUF (~4.6 GB)  remote-host@192.168.1.10:~/repos/codescribe-train/checkpoints/sample-qwen7b-q4_k_m.gguf
... 4.6 GB transferred ...
[bootstrap_windows] rsync rag.db          remote-host@192.168.1.10:~/repos/codescribe-train/indices/rag.db
[bootstrap_windows] rsync embedder (~3.8 GB) ...
[bootstrap_windows] uv sync --extra rag --extra dev
[bootstrap_windows] writing C:\Users\YourName\repos\codescribe-train\.claw\settings.json
[bootstrap_windows] claw ok: ...\target\release\claw.exe
[bootstrap_windows] llama-server ok: ...\bin\Release\llama-server.exe
[bootstrap_windows] smoke: import MCP server modules
[bootstrap_windows] smoke: rag store status
[bootstrap_windows] DONE.
```

### Re-running

```powershell
.\scripts\bootstrap_windows.ps1 -SkipRsync
.\scripts\bootstrap_windows.ps1 -From remote-host@192.168.1.10 -SshPort 2222   # refresh
```

---

## 4. Start `llama-server` (long-lived)

### Option A — separate `pwsh` terminal

```powershell
cd $env:USERPROFILE\repos\codescribe-train
.\vendor\llama.cpp\build\bin\Release\llama-server.exe `
    -m .\checkpoints\sample-qwen7b-q4_k_m.gguf `
    --host 127.0.0.1 --port 8080 `
    -ngl -1 --ctx-size 32768
```

Leave the terminal open. Closing it terminates the server.

### Option B — Task Scheduler (recommended for long-lived)

Create a task that starts `llama-server` at login and restarts on
failure. From an elevated `pwsh`:

```powershell
$Action = New-ScheduledTaskAction `
    -Execute "$env:USERPROFILE\repos\codescribe-train\vendor\llama.cpp\build\bin\Release\llama-server.exe" `
    -Argument "-m $env:USERPROFILE\repos\codescribe-train\checkpoints\sample-qwen7b-q4_k_m.gguf --host 127.0.0.1 --port 8080 -ngl -1 --ctx-size 32768" `
    -WorkingDirectory "$env:USERPROFILE\repos\codescribe-train"

$Trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask -TaskName "llama-server" `
    -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal
```

To remove later: `Unregister-ScheduledTask -TaskName "llama-server" -Confirm:$false`.

### Option C — NSSM (run as a Windows service)

[NSSM](https://nssm.cc/) wraps any binary as a Windows service. Useful
if you want `llama-server` to start at boot, before login. Skip unless
you specifically need pre-login startup.

### Avoid binding to LAN

Same rule as Mac/Linux: `--host 127.0.0.1` only. `--host 0.0.0.0`
exposes the model to your LAN; the strictly-local posture forbids it.

---

## 5. claw on native Windows

The bash wrapper [`scripts/run-claw.sh`](../scripts/run-claw.sh) is
bash-only. Three options on Windows:

### Option A — run via Git Bash

```bash
# In Git Bash:
cd ~/repos/codescribe-train
bash scripts/run-claw.sh
```

The wrapper's `curl` and `cd` work fine under Git Bash's MSYS layer.
This is the simplest path.

### Option B — invoke `claw.exe` directly

The wrapper does three things: probe `127.0.0.1:8080`, export env vars,
exec claw. Do them by hand once if you don't want Git Bash:

```powershell
# 1. Probe
$probe = try { (Invoke-WebRequest -Uri http://127.0.0.1:8080/v1/models -UseBasicParsing).StatusCode } catch { 0 }
if ($probe -ne 200) { Write-Error "llama-server not answering at 127.0.0.1:8080"; exit 2 }

# 2. Pre-flight banner so claw's "via openai" label can't be misread
Write-Host "[run-claw] LOCAL ONLY - claw -> http://127.0.0.1:8080 (loopback, NOT api.openai.com)"

# 3. Env vars + exec
$env:OPENAI_BASE_URL    = "http://127.0.0.1:8080"
$env:OPENAI_API_KEY     = "local-no-auth"
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
& .\vendor\claw-code\rust\target\release\claw.exe --model openai/sample-qwen7b-local
```

A PowerShell equivalent of `run-claw.sh` (`scripts/run-claw.ps1`) is a
reasonable follow-up commit but does not exist yet — the wrapper isn't
on the same critical path as the porter.

### Option C — WSL alternative

In your WSL distro, follow [`PORTING-LINUX.md`](PORTING-LINUX.md) — both
the porter and the wrapper "just work" there. `llama-server` running
inside WSL on a CUDA-equipped Windows host still talks to the host GPU
via WSL2's CUDA passthrough.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `rsync : The term 'rsync' is not recognized` | rsync not on PATH | Install Git for Windows; add `C:\Program Files\Git\usr\bin` to PATH; restart pwsh |
| `Set-ExecutionPolicy` error when running the porter | Default `Restricted` policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (one-time) |
| `bootstrap_windows.ps1` complains uv not found | uv installed but `%USERPROFILE%\.local\bin` not on PATH | Add it: `[Environment]::SetEnvironmentVariable("PATH", $env:Path + ";$env:USERPROFILE\.local\bin", "User")`; restart pwsh |
| `Test-Path llama-server.exe` fails after CUDA build | Binary lands under `build\bin\Release\` (MSVC multi-config), not `build\bin\` | The porter already looks under `Release\`; if you used Ninja single-config, move/symlink to `Release\` |
| `Path too long` errors during git clone or cargo build | Windows MAX_PATH 260 limit | `git config --system core.longpaths true`; enable long-path support in Windows (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`) |
| CUDA build OOMs the linker | MSVC linker is RAM-heavy | Reduce parallelism: `cmake --build build --config Release -j 2`; or close other apps |
| `ssh: connect to host ... port 22: Connection timed out` to a LAN IP | Not actually on the same LAN, or Windows Firewall on either end is blocking SSH | Verify both hosts are on the same subnet (`ipconfig` / `ip addr show`); open inbound TCP 22 (or your portproxy port) on the source-side firewall |
| Antivirus quarantines `llama-server.exe` | Some AV products flag CUDA binaries as suspicious | Add the `vendor\llama.cpp\build\bin\Release\` directory to AV exclusions |
| `find_similar_issues` returns "no results" | rag.db rsync'd to wrong location | Check `RAG_DB_PATH` in `.claw\settings.json` — should be an absolute Windows path; the porter writes this correctly |
| `OPENAI_API_KEY` is set globally in your env to a real key | Wrapper / direct invocation would pass it, but `OPENAI_BASE_URL` is pinned to loopback so it goes nowhere on the wire | No real risk; if you want belt-and-suspenders, `$env:OPENAI_API_KEY = $null` before launching claw |

---

## 7. Hosting from Windows (port forwarding so others can rsync from your WSL)

This is the *source-side* counterpart to [`PORTING.md` §4.2](PORTING.md#42-if-the-source-is-a-wsl2-distro-on-a-windows-host).
If a WSL-hosted llama-server box is the **source** that other hosts on
your LAN port from, you need to expose the WSL distro's
`sshd` through the Windows host's LAN interface — the WSL VM lives
behind a NAT inside the Windows host and isn't directly reachable
from your LAN without this forward.

### Find the WSL's internal IP

From inside the WSL distro:

```bash
ip -4 addr show eth0 | grep "inet "
# inet 172.24.80.1/20 brd 172.24.80.255 scope global eth0
```

That `172.24.x.x/20` is the WSL VM's address inside the Windows
host's NAT — **only reachable from the Windows host itself**. Note
that this IP can change across WSL restarts; either pin it (search
"WSL2 static IP" — requires `wsl.conf` + a startup script) or
re-run the portproxy command after each WSL boot.

### Set up the port forward (Windows host, admin PowerShell)

```powershell
# Pick any free port on the Windows host. 2222 avoids colliding with
# the Windows-native OpenSSH server (if you have one on 22).
netsh interface portproxy add v4tov4 `
    listenport=2222 listenaddress=0.0.0.0 `
    connectport=22 connectaddress=172.24.80.1

# Open the inbound firewall for that port:
New-NetFirewallRule -DisplayName "WSL SSH 2222" `
    -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow

# Verify:
netsh interface portproxy show v4tov4
# Listen on ipv4:             Connect to ipv4:
# Address       Port        Address       Port
# ------------- ----------  ------------- ----------
# 0.0.0.0       2222        172.24.80.1  22
```

The portproxy survives reboots (it's stored in the registry under
`HKLM\SYSTEM\CurrentControlSet\Services\PortProxy\v4tov4\tcp`). The
firewall rule does too.

### Find the Windows host's LAN IP

From a normal PowerShell:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.InterfaceAlias -notmatch 'Loopback' -and
    $_.IPAddress -match '^(192\.168|10\.|172\.(1[6-9]|2[0-9]|3[01]))'
} | Select-Object IPAddress, InterfaceAlias
# IPAddress     InterfaceAlias
# ---------     --------------
# 192.168.1.10 Ethernet
```

That `192.168.x.x` (or `10.x.x.x`) is what client hosts should rsync
*from*. SSH to `your-windows-ip:2222` lands inside the WSL distro
authenticated against `~remote-host/.ssh/authorized_keys` exactly as if you
were SSH-ing to the WSL natively.

### Verify from the client (Mac / Linux)

```bash
ssh -p 2222 -i ~/.ssh/id_ed25519_remote-host remote-host@192.168.1.10 'hostname; date'
# example-host
# Thu Jan  1 00:00:00 UTC 2026
```

Then point the porter at it (don't forget `--ssh-port`):

```bash
bash scripts/bootstrap_mac.sh \
    --from remote-host@192.168.1.10 \
    --ssh-port 2222
```

(SSH lands inside the WSL distro authenticated against
`~remote-host/.ssh/authorized_keys` exactly as a direct ssh would.)

### Removing the port forward later

```powershell
netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
Remove-NetFirewallRule -DisplayName "WSL SSH 2222"
```

---

## 8. Why WSL might be the better choice on Windows

If you don't already have a Windows-native build of `llama.cpp` and
`claw`, WSL2 is the lower-friction path:

- One package manager (`apt` / `uv`) instead of three (`winget`, MSVC,
  CUDA installer).
- Same scripts as your other Linux boxes — one codepath to learn.
- CUDA passthrough works well; performance is within a few percent of
  native Windows for inference workloads.

The native Windows path matters when:

- You need the model accessible to a Windows-native application
  (PowerShell scripting, Windows GUI tools) that can't easily reach a
  WSL-internal `127.0.0.1` (WSL2 networking quirks notwithstanding).
- You're on a Windows host without WSL enabled (corporate locked
  hardware, Server SKUs without WSL).
- You want one fewer hypervisor in the loop.

Pick the path that matches your existing workflow.
