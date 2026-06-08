"""Path B — distributed data-parallel helpers (codescribe issue #3 §6).

Pure builders for the ``torchrun`` command line and the NCCL environment that
actually works on consumer LAN hardware. No process is launched here — the CLI's
``ddp-plan`` subcommand prints these for you to run via ``scripts/run_ddp.sh`` on
each node, and the unit tests pin their shape.

**Read §3 of the issue first.** On 1 Gbps Ethernet, DDP gives ~1.4x wall-clock at
~3x the compute. Path A almost always wins on an 8 GB-VRAM laptop fleet.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class DDPNode:
    """One node's role in a torchrun rendezvous."""

    node_rank: int
    nnodes: int
    rdzv_host: str
    rdzv_port: int = 29400
    rdzv_id: str = "ft-001"
    nproc_per_node: int = 1


def build_torchrun_command(
    node: DDPNode,
    *,
    train_module: str,
    config: str,
    dataset: str,
    out: str,
    runner: str = "uv run",
) -> str:
    """Return the ``torchrun`` command to run on a given node.

    Mirrors issue #3 §6.2. Run the SAME command on every node, varying only
    ``node.node_rank`` (0 on the rendezvous master). All interpolated values are
    ``shlex.quote``d; ``runner``/``train_module`` are trusted operator strings.
    """
    if not 0 <= node.node_rank < node.nnodes:
        raise ValueError(
            f"node_rank {node.node_rank} out of range for nnodes={node.nnodes}"
        )
    endpoint = f"{node.rdzv_host}:{node.rdzv_port}"
    prefix = f"{runner} " if runner else ""  # torchrun may be on PATH directly
    return (
        f"{prefix}torchrun "
        f"--nproc-per-node={node.nproc_per_node} "
        f"--nnodes={node.nnodes} "
        f"--node-rank={node.node_rank} "
        f"--rdzv-id={shlex.quote(node.rdzv_id)} "
        f"--rdzv-backend=c10d "
        f"--rdzv-endpoint={shlex.quote(endpoint)} "
        f"-m {shlex.quote(train_module)} run "
        f"--config {shlex.quote(config)} "
        f"--dataset {shlex.quote(dataset)} "
        f"--out {shlex.quote(out)}"
    )


def nccl_env(socket_ifname: str, *, debug: bool = True) -> dict[str, str]:
    """NCCL env vars for consumer hardware over plain TCP (issue #3 §6.3).

    ``socket_ifname`` is the *wired* interface name (``ip -br link``), e.g. ``eno1``.
    Getting this wrong is the #1 cause of ``NCCL WARN Could not find network address``.
    """
    env = {
        "NCCL_IB_DISABLE": "1",            # no InfiniBand on a laptop
        "NCCL_NET": "Socket",              # force TCP transport
        "NCCL_SOCKET_IFNAME": socket_ifname,
        "TORCH_DIST_INIT_BARRIER_TIMEOUT": "600",  # slow-LAN rendezvous tolerance
    }
    if debug:
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "NET,GRAPH"
    return env


def scale_lr_for_nodes(base_lr: float, nnodes: int) -> float:
    """DDP multiplies effective batch size by node count; scale LR by sqrt(N).

    Issue #3 §6.6: if you tuned LR for single-node bs=16, lower it by ~sqrt(N) when
    going to N-node DDP, or expect instability.
    """
    if nnodes < 1:
        raise ValueError("nnodes must be >= 1")
    return base_lr / math.sqrt(nnodes)


def effective_batch_size(batch_size: int, grad_accum_steps: int, nnodes: int) -> int:
    """Effective batch = per-device batch x grad-accum x nodes (each node = 1 GPU here)."""
    return batch_size * grad_accum_steps * nnodes
