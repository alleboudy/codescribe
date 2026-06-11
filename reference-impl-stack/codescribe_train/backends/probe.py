"""Hardware probe + backend recommendation.

Reads the local box's specs (GPU VRAM, system RAM, CPU count) and
recommends a default backend + a starter set of options for a given
model size. The output is consumed by the top-level ``codescribe-train run``
glue to fill in defaults, and exposed standalone via
``python -m codescribe_train.backends probe``.

Privacy posture: this module never reaches the network. It uses
``pynvml`` if available; otherwise it parses ``nvidia-smi --query-gpu=...``
output. ``psutil.virtual_memory`` and ``os.cpu_count`` cover the rest. All
imports of optional / heavy deps are inside the functions that need them.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Bytes-to-GiB factor used everywhere in this module so the surface stays
# in human-friendly units while the underlying APIs hand back bytes.
_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Frozen snapshot of the local hardware. Hashable, comparable, immutable."""

    gpu_name: str | None
    gpu_vram_gib: float | None
    system_ram_gib: float
    cpu_count: int

    def has_gpu(self) -> bool:
        return self.gpu_name is not None and (self.gpu_vram_gib or 0.0) > 0.0

    def summary(self) -> str:
        parts = []
        if self.has_gpu():
            parts.append(f"gpu={self.gpu_name} ({self.gpu_vram_gib:.1f} GiB VRAM)")
        else:
            parts.append("gpu=none")
        parts.append(f"ram={self.system_ram_gib:.1f} GiB")
        parts.append(f"cpu={self.cpu_count} threads")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class BackendChoice:
    """Recommended backend and key knobs for a given (profile, model) pair.

    ``backend_name`` is one of ``"llama-server"``, ``"vllm"``, ``"ollama"``.
    ``options`` is a dict of suggested defaults the CLI surfaces as flags.
    ``rationale`` is a one-line human-readable why-this-pick string for
    logs / ``probe`` output.
    """

    backend_name: str
    options: dict[str, object]
    rationale: str


# ---------------------------------------------------------------------- #
# Probing
# ---------------------------------------------------------------------- #


def probe_hardware() -> HardwareProfile:
    """Snapshot the local hardware. Strictly local — no network calls."""
    gpu_name, gpu_vram_gib = _probe_gpu()
    ram_gib = _probe_ram_gib()
    cpu_count = os.cpu_count() or 1
    profile = HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_gib=gpu_vram_gib,
        system_ram_gib=ram_gib,
        cpu_count=cpu_count,
    )
    logger.info("probed hardware: %s", profile.summary())
    return profile


def _probe_gpu() -> tuple[str | None, float | None]:
    """Probe the primary CUDA GPU. Prefers pynvml; falls back to nvidia-smi."""
    # 1) pynvml — fast, structured, optional dependency.
    pynvml_result = _probe_gpu_via_pynvml()
    if pynvml_result is not None:
        return pynvml_result

    # 2) nvidia-smi — present whenever the NVIDIA driver is. Parse plain CSV.
    return _probe_gpu_via_nvidia_smi()


def _probe_gpu_via_pynvml() -> tuple[str | None, float | None] | None:
    """Return (name, vram_gib), or ``None`` if pynvml isn't usable."""
    try:
        import pynvml  # noqa: PLC0415 — optional dep, lazy by design

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return None, None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else name_raw
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return name, mem.total / _GIB
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml not installed; falling back to nvidia-smi")
    except Exception as e:  # noqa: BLE001 — pynvml init can fail many ways
        logger.info("pynvml probe failed (%s); falling back to nvidia-smi", e)
    return None


def _probe_gpu_via_nvidia_smi() -> tuple[str | None, float | None]:  # noqa: PLR0911 — early returns track failure modes
    """Return (name, vram_gib) by shelling out to ``nvidia-smi``."""
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        return None, None
    try:
        result = subprocess.run(  # noqa: S603 — fixed args, no shell
            [
                nvsmi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.info("nvidia-smi probe failed: %s", e)
        return None, None
    if result.returncode != 0:
        logger.info("nvidia-smi returned %s; stderr=%s", result.returncode, result.stderr.strip())
        return None, None
    first = result.stdout.strip().splitlines()[:1]
    if not first:
        return None, None
    parts = [p.strip() for p in first[0].split(",")]
    if len(parts) < 2:
        return None, None
    name = parts[0]
    try:
        # nvidia-smi reports memory.total in MiB by default.
        vram_mib = float(parts[1])
    except ValueError:
        return name, None
    return name, vram_mib / 1024.0


def _probe_ram_gib() -> float:
    """System RAM in GiB. Prefers psutil; falls back to /proc/meminfo."""
    try:
        import psutil  # noqa: PLC0415 — optional dep, lazy by design

        return psutil.virtual_memory().total / _GIB
    except ImportError:
        logger.debug("psutil not installed; falling back to /proc/meminfo")

    meminfo = "/proc/meminfo"
    try:
        with open(meminfo, encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # MemTotal is reported in kB.
                    return float(parts[1]) / (1024.0 * 1024.0)
    except OSError:
        logger.info("could not read %s for RAM probe", meminfo)
    return 0.0


# ---------------------------------------------------------------------- #
# Recommendation
# ---------------------------------------------------------------------- #


def recommend_backend(profile: HardwareProfile, *, model_size_gib: float) -> BackendChoice:
    """Pick a sensible backend + knobs given the box and the target model size.

    Heuristics (locked at planning time, see ``docs/HARDWARE-AND-PERFORMANCE.md``):

    * **No GPU** → llama-server with all CPU layers and ``--no-mmap`` if
      RAM is tight, but a clear note that this is going to be slow.
    * **Model > VRAM** → llama-server with partial offload. The 7B Q4_K_M
      model on an 8 GB-VRAM GPU fits with ``-ngl 99 --no-mmap``;
      ``--ctx-size 8192`` is the comfort cap.
    * **Model fits in VRAM with > 2 GiB headroom** → vLLM is permitted but
      llama-server is still the default (simpler, fewer deps, same OpenAI-
      compat surface).

    vLLM is **never** the default on tight-VRAM consumer GPUs. Ollama is
    **never** the default — selecting it requires the user to ask for it
    explicitly.
    """
    gpu_vram = float(profile.gpu_vram_gib or 0.0)

    if not profile.has_gpu():
        return BackendChoice(
            backend_name="llama-server",
            options={
                "gpu_layers": 0,
                "ctx_size": 4096,
                "no_mmap": False,
            },
            rationale="no GPU detected; running on CPU is going to be slow",
        )

    headroom_gib = gpu_vram - model_size_gib
    if headroom_gib < 0:
        # Model bigger than VRAM → still llama-server, but more offload-friendly.
        return BackendChoice(
            backend_name="llama-server",
            options={
                # llama.cpp interprets large positive numbers as "all
                # transformer layers"; -1 also works. We keep 99 for
                # clarity in logs.
                "gpu_layers": 99,
                "ctx_size": 8192,
                "no_mmap": True,
            },
            rationale=(
                f"model ~{model_size_gib:.1f} GiB > VRAM ~{gpu_vram:.1f} GiB; "
                "partial GPU offload via llama-server"
            ),
        )

    if headroom_gib < 2.0:
        return BackendChoice(
            backend_name="llama-server",
            options={
                "gpu_layers": 99,
                "ctx_size": 8192,
                "no_mmap": True,
            },
            rationale=(
                f"model fits in VRAM but headroom ~{headroom_gib:.1f} GiB is tight; "
                "llama-server with --no-mmap"
            ),
        )

    # Comfortable fit. Still default llama-server (per docs/HARDWARE-AND-PERFORMANCE.md).
    return BackendChoice(
        backend_name="llama-server",
        options={
            "gpu_layers": 99,
            "ctx_size": 8192,
            "no_mmap": False,
        },
        rationale=(
            f"comfortable fit (model ~{model_size_gib:.1f} GiB, "
            f"VRAM ~{gpu_vram:.1f} GiB headroom ~{headroom_gib:.1f} GiB); "
            "default llama-server"
        ),
    )
