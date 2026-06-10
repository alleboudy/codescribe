#!/usr/bin/env pwsh
# scripts/bootstrap_windows.ps1 — port codescribe-train to a Windows host.
#
# Companion: docs/PORTING-WINDOWS.md
#
# Native Windows (PowerShell 7+). For Windows users running WSL,
# `scripts/bootstrap_linux.sh` works inside the WSL distro.
#
# What this does (in order):
#   1. rsync the artefacts (GGUF, rag.db, bge embedder) from a source
#      host you've already set up.
#   2. uv sync --extra rag --extra dev
#   3. generate .claw/settings.json with absolute Windows paths
#   4. verify the vendored binaries exist (.exe under vendor/...);
#      print a CUDA-aware build hint if not.
#   5. smoke-test MCP imports + rag store status.
#
# Targets x86_64 Windows. CUDA-capable NVIDIA GPU strongly recommended.
#
# Usage:
#   pwsh scripts/bootstrap_windows.ps1 -From <ssh-target> [options]
#
# Options:
#   -From <user@host>         Source SSH target. Required unless -SkipRsync.
#   -SshKey <path>            SSH private key (default: %USERPROFILE%\.ssh\id_ed25519_remote-host)
#   -SshPort <port>           SSH port on source (default: 22; use for LAN port-forward bypasses)
#   -SourceRepo <path>        codescribe-train path on source (default: ~/repos/codescribe-train)
#   -SourceHfModels <path>    HF model cache on source (default: ~/.hf-models)
#   -SkipRsync                Skip artefact transfer; assume already present locally
#   -SkipSmoke                Skip the smoke tests at the end
#   -Help                     Print this help and exit
#
# Prerequisites:
#   - PowerShell 7+ (pwsh)
#   - OpenSSH client (built-in on Windows 10+)
#   - rsync — install via Git for Windows (bundles `rsync.exe`) or MSYS2
#   - Python 3.12 + uv on PATH (uv installer puts it under %USERPROFILE%\.local\bin)
#   - vendored binaries built for Windows-x64 (CUDA + llama.cpp + claw)
#
# Strictly local: only network egress is SSH/rsync to <source-host>,
# a box you own on the same LAN. See docs/STRICTLY-LOCAL-POSTURE.md
# and docs/PORTING.md §4 for transport details + portproxy setup.

[CmdletBinding()]
param(
    [string]$From = "",
    [string]$SshKey = "$env:USERPROFILE\.ssh\id_ed25519_remote-host",
    [int]$SshPort = 22,
    [string]$SourceRepo = "~/repos/codescribe-train",
    [string]$SourceHfModels = "~/.hf-models",
    [switch]$SkipRsync,
    [switch]$SkipSmoke,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Show-Usage {
    Get-Content -Path $MyInvocation.MyCommand.Path |
        Select-Object -First 50 |
        ForEach-Object { $_ -replace '^# ?','' } |
        Where-Object { $_ -notmatch '^!' } |
        Write-Host
}

if ($Help) { Show-Usage; exit 0 }

function Log { param([string]$Message) Write-Host "[bootstrap_windows] $Message" }

# ---- 0. locate repo root, uv, ssh, rsync ---------------------------------

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

Log "host: Windows/$env:PROCESSOR_ARCHITECTURE, user=$env:USERNAME, repo=$RepoRoot"

$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) {
    foreach ($candidate in @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:USERPROFILE\.cargo\bin\uv.exe",
        "C:\Program Files\uv\uv.exe"
    )) {
        if (Test-Path $candidate) { $Uv = $candidate; break }
    }
}
if (-not $Uv) {
    Write-Error "uv not on PATH. Install: https://docs.astral.sh/uv/"
    exit 2
}
Log "uv: $Uv"

foreach ($tool in @("rsync", "ssh")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "$tool not on PATH. Install Git for Windows (bundles both)."
        exit 2
    }
}

# ---- 1. rsync artefacts from source --------------------------------------

if (-not $SkipRsync) {
    if (-not $From) {
        Write-Error "-From is required (or pass -SkipRsync to reuse local artefacts)"
        exit 1
    }

    & ssh -p $SshPort -i $SshKey -o BatchMode=yes -o ConnectTimeout=5 $From "true" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "cannot SSH to ${From}:$SshPort with key $SshKey. Verify: ssh -p $SshPort -i $SshKey $From 'echo ok'"
        exit 3
    }

    foreach ($dir in @(
        "$RepoRoot\checkpoints",
        "$RepoRoot\indices",
        "$env:USERPROFILE\.hf-models"
    )) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    $sshArg = "ssh -p $SshPort -i $SshKey"

    Log "rsync GGUF (~4.6 GB)  ${From}:${SourceRepo}/checkpoints/sample-qwen7b-q4_k_m.gguf"
    & rsync -avh --progress -e $sshArg "${From}:${SourceRepo}/checkpoints/sample-qwen7b-q4_k_m.gguf" "$RepoRoot/checkpoints/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Log "rsync rag.db          ${From}:${SourceRepo}/indices/rag.db"
    & rsync -avh --progress -e $sshArg "${From}:${SourceRepo}/indices/rag.db" "$RepoRoot/indices/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Log "rsync embedder (~3.8 GB)  ${From}:${SourceHfModels}/bge-large-en-v1.5/"
    & rsync -avh --progress -e $sshArg "${From}:${SourceHfModels}/bge-large-en-v1.5/" "$env:USERPROFILE/.hf-models/bge-large-en-v1.5/"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Log "skipping artefact rsync (-SkipRsync)"
}

# ---- 2. sync Python deps -------------------------------------------------

Log "uv sync --extra rag --extra dev"
& $Uv sync --extra rag --extra dev   # match bootstrap_linux.sh — keep dev tools alongside rag
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 3. generate .claw/settings.json -------------------------------------

# Windows venv layout: Scripts\python.exe (not bin/python).
$PyExe = "$RepoRoot\.venv\Scripts\python.exe"

# Heuristic: sample repo lives next to codescribe-train.
$SampleRepoCandidate = Join-Path (Split-Path -Parent $RepoRoot) "sample"
$SampleRepo = if (Test-Path $SampleRepoCandidate) {
    (Resolve-Path $SampleRepoCandidate).Path
} else {
    $SampleRepoCandidate
}

New-Item -ItemType Directory -Force -Path "$RepoRoot\.claw","$RepoRoot\logs" | Out-Null
$ClawJson = "$RepoRoot\.claw\settings.json"
Log "writing $ClawJson (gitignored, per-host; absolute python avoids PATH surprises)"

# JSON uses forward slashes — Windows accepts them as path separators, and
# this sidesteps JSON-escaping the backslashes.
$PyJson         = $PyExe        -replace '\\','/'
$RagDbJson      = "$RepoRoot\indices\rag.db"       -replace '\\','/'
$RagLogJson     = "$RepoRoot\logs\rag-server.log"  -replace '\\','/'
$SampleRepoJson = $SampleRepo                      -replace '\\','/'

# claw speaks LSP-framed stdio MCP; mcp 1.27.x's stdio_server speaks
# newline-delimited. Each server runs through _mcp_framing_bridge,
# which translates between the two. See module docstring for details.
#
# permissions.defaultMode = "dontAsk" (claw's term -> DangerFullAccess):
# no per-tool prompts, full tool access including Bash. This is the file
# claw actually reads (it does NOT read .claude.json). Change to
# "default" for interactive approval, "acceptEdits" to allow edits only.
@"
{
  "permissions": {
    "defaultMode": "dontAsk"
  },
  "mcpServers": {
    "repo-rag": {
      "command": "$PyJson",
      "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.rag_server"],
      "env": {
        "RAG_DB_PATH": "$RagDbJson",
        "RAG_LOG_PATH": "$RagLogJson"
      }
    },
    "repo-grep": {
      "command": "$PyJson",
      "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.repo_grep"],
      "env": {
        "TARGET_REPO": "$SampleRepoJson"
      }
    },
    "repo-docs": {
      "command": "$PyJson",
      "args": ["-m", "codescribe_train.servers._mcp_framing_bridge", "codescribe_train.servers.repo_docs"],
      "env": {
        "TARGET_REPO": "$SampleRepoJson"
      }
    }
  }
}
"@ | Set-Content -Path $ClawJson -Encoding UTF8

if (-not (Test-Path $SampleRepo)) {
    Log "WARN: $SampleRepo does not exist; repo-grep MCP server will fail until you clone or rsync the target repo there."
}

# ---- 4. verify binaries --------------------------------------------------

$ClawBin  = "$RepoRoot\vendor\claw-code\rust\target\release\claw.exe"
$LlamaBin = "$RepoRoot\vendor\llama.cpp\build\bin\Release\llama-server.exe"

function Test-Binary {
    param([string]$Label, [string]$Path, [string]$BuildHint)
    if (-not (Test-Path $Path)) {
        Write-Warning "$Label not built at $Path"
        Write-Warning "  build hint: $BuildHint"
        return $false
    }
    Log "$Label ok: $Path"
    return $true
}

$CudaHint  = "Install VS 2022 Build Tools + NVIDIA CUDA Toolkit 12.x, then: cmake -B vendor\llama.cpp\build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=<your_sm> -DLLAMA_BUILD_SERVER=ON vendor\llama.cpp; cmake --build vendor\llama.cpp\build --config Release -j --target llama-server"
$CargoHint = "Install Rust via https://win.rustup.rs, then: cd vendor\claw-code\rust; cargo build --release --workspace"

Test-Binary "claw"         $ClawBin  $CargoHint  | Out-Null
Test-Binary "llama-server" $LlamaBin $CudaHint   | Out-Null

# ---- 5. smoke tests ------------------------------------------------------

if (-not $SkipSmoke) {
    Log "smoke: import MCP server modules"
    & $PyExe -c "import codescribe_train.servers.rag_server, codescribe_train.servers.repo_grep"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "MCP server modules don't import — uv sync did not provide the rag extra"
        exit 4
    }
    Log "smoke: rag store status"
    & $Uv run --extra rag python -m codescribe_train.rag status
    if ($LASTEXITCODE -ne 0) {
        Write-Error "rag status failed — rag.db missing or unreadable"
        exit 5
    }
} else {
    Log "skipping smoke tests (-SkipSmoke)"
}

# ---- 6. next steps -------------------------------------------------------

Log "DONE."
Log ""
Log "Next steps to chat with the fine-tuned model on this Windows host:"
Log "  1. Start llama-server in a separate terminal (long-lived, CUDA-accelerated):"
Log "       $LlamaBin ``"
Log "         -m $RepoRoot\checkpoints\sample-qwen7b-q4_k_m.gguf ``"
Log "         --host 127.0.0.1 --port 8080 -ngl -1 --ctx-size 32768"
Log "  2. Launch claw against it. The bash wrapper (scripts/run-claw.sh) needs"
Log "     Git Bash or WSL; native claw.exe invocation is documented in"
Log "     docs/PORTING-WINDOWS.md."
