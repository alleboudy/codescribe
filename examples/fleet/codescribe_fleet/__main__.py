"""CLI for the fleet orchestrator.

    python -m codescribe_fleet sweep     --fleet F --sweep S --out O [--mode local|ssh]
    python -m codescribe_fleet status    --sweep O
    python -m codescribe_fleet summarise --sweep O
    python -m codescribe_fleet ddp-plan  --fleet F --sweep S [--rdzv-host H]
    python -m codescribe_fleet monitor   --fleet F [--mode ...] [--interval N] [--rounds N]

This is an entry point, so it prints to stdout/stderr (library modules never do).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import FleetConfig, load_fleet_config, load_sweep_config
from .config import WorkerSpec
from .ddp import DDPNode, build_torchrun_command, nccl_env
from .monitor import poll_fleet
from .orchestrator import Orchestrator
from .summary import best, format_table, load_state, summarise
from .transport import LocalTransport, SSHTransport, Transport


def _make_transport_factory(mode: str):
    if mode == "local":
        return lambda spec: LocalTransport(spec.name)
    if mode == "ssh":
        def factory(spec: WorkerSpec) -> Transport:
            return SSHTransport(
                spec.ssh_host,
                username=spec.ssh_user,
                port=spec.ssh_port,
                client_keys=[spec.identity_file] if spec.identity_file else None,
            )
        return factory
    raise SystemExit(f"unknown --mode {mode!r} (expected 'local' or 'ssh')")


def _cmd_sweep(args: argparse.Namespace) -> int:
    fleet = load_fleet_config(args.fleet)
    sweep = load_sweep_config(args.sweep)
    orch = Orchestrator(
        fleet,
        sweep,
        out_dir=args.out,
        transport_factory=_make_transport_factory(args.mode),
        base_config_dir=args.base_dir,
    )
    result = asyncio.run(orch.run())
    print(format_table(summarise(args.out)))
    win = best(summarise(args.out))
    if win is not None:
        print(f"\nbest: {win.job_id} (task_mean={win.task_mean:.3f}) on {win.worker}")
    if result.aborted:
        print("\nsweep ABORTED — see state.json", file=sys.stderr)
        return 2
    return 0 if not result.failed else 1


def _cmd_status(args: argparse.Namespace) -> int:
    state = load_state(args.sweep)
    print(
        f"sweep {state['sweep']}: {state['settled']}/{state['total']} settled"
        f"{' (ABORTED)' if state.get('aborted') else ''}"
    )
    for name, w in sorted(state.get("workers", {}).items()):
        print(
            f"  worker {name:<12} {w['status']:<11} "
            f"completed={w['completed']} consec_fail={w['consecutive_failures']}"
        )
    counts: dict[str, int] = {}
    for j in state.get("jobs", {}).values():
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print("  jobs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def _cmd_summarise(args: argparse.Namespace) -> int:
    summaries = summarise(args.sweep)
    print(format_table(summaries))
    win = best(summaries)
    if win is not None:
        print(f"\nbest: {win.job_id} (task_mean={win.task_mean:.3f}) on {win.worker}")
    return 0


def _cmd_ddp_plan(args: argparse.Namespace) -> int:
    fleet = load_fleet_config(args.fleet)
    sweep = load_sweep_config(args.sweep)
    if fleet.train.module is None:
        print(
            "ddp-plan needs a `train.module` (torchrun uses `-m`); "
            "the script-based launcher can't be used with torchrun.",
            file=sys.stderr,
        )
        return 2
    nnodes = args.nnodes or len(fleet.workers)
    if nnodes > len(fleet.workers):
        print(
            f"ddp-plan: --nnodes {nnodes} exceeds the {len(fleet.workers)} workers "
            "in the fleet (the rendezvous would wait forever for nodes that never join)",
            file=sys.stderr,
        )
        return 2
    rdzv_host = args.rdzv_host or fleet.workers[0].ssh_host
    out = args.out or f"checkpoints/{sweep.name}-ddp/"
    print(f"# Path B plan: {nnodes} nodes, rendezvous master = {rdzv_host}")
    print("# Run scripts/run_ddp.sh on each node, OR run the command below per node.\n")
    for rank, w in enumerate(fleet.workers[:nnodes]):
        node = DDPNode(node_rank=rank, nnodes=nnodes, rdzv_host=rdzv_host)
        env = nccl_env(w.socket_ifname or "eno1")
        print(f"## {w.name} (NODE_RANK={rank}, ssh {w.ssh_host})")
        print("export " + " ".join(f"{k}={v}" for k, v in env.items()))
        print(
            build_torchrun_command(
                node,
                train_module=fleet.train.module,
                config=sweep.base_train_config,
                dataset=sweep.base_dataset,
                out=out,
                runner=_torchrun_runner(fleet.train.runner),
            )
        )
        print()
    return 0


def _torchrun_runner(runner: str) -> str:
    """Turn a train runner into a torchrun runner by dropping a trailing python.

    torchrun is its own console entry point, so `uv run python` -> `uv run`,
    `python` -> `` (torchrun on PATH). Avoids emitting a non-runnable
    `python torchrun ...`.
    """
    tokens = runner.split()
    if tokens and tokens[-1] in ("python", "python3"):
        tokens = tokens[:-1]
    return " ".join(tokens)


def _cmd_monitor(args: argparse.Namespace) -> int:
    fleet = load_fleet_config(args.fleet)
    out = asyncio.run(
        poll_fleet(
            fleet.workers,
            _make_transport_factory(args.mode),
            out_jsonl=args.out,
            interval=args.interval,
            rounds=args.rounds,
        )
    )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # --log-level lives on a shared parent so it works AFTER the subcommand
    # (the natural place, e.g. `... sweep --log-level WARNING`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO")

    p = argparse.ArgumentParser(prog="codescribe_fleet", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", parents=[common], help="run a Path A hyperparameter sweep")
    s.add_argument("--fleet", required=True)
    s.add_argument("--sweep", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", default="ssh", choices=["local", "ssh"])
    s.add_argument("--base-dir", default=".", help="resolve base_train_config/dataset here")
    s.set_defaults(func=_cmd_sweep)

    st = sub.add_parser("status", parents=[common], help="print a sweep's current state.json")
    st.add_argument("--sweep", required=True, help="the sweep output dir")
    st.set_defaults(func=_cmd_status)

    sm = sub.add_parser("summarise", parents=[common], help="print the sorted results table")
    sm.add_argument("--sweep", required=True, help="the sweep output dir")
    sm.set_defaults(func=_cmd_summarise)

    d = sub.add_parser("ddp-plan", parents=[common], help="print the Path B torchrun plan per node")
    d.add_argument("--fleet", required=True)
    d.add_argument("--sweep", required=True)
    d.add_argument("--rdzv-host", default=None)
    d.add_argument("--nnodes", type=int, default=None)
    d.add_argument("--out", default=None)
    d.set_defaults(func=_cmd_ddp_plan)

    m = sub.add_parser("monitor", parents=[common], help="poll nvidia-smi across the fleet to JSONL")
    m.add_argument("--fleet", required=True)
    m.add_argument("--mode", default="ssh", choices=["local", "ssh"])
    m.add_argument("--out", default="logs/fleet-monitor.jsonl")
    m.add_argument("--interval", type=float, default=5.0)
    m.add_argument("--rounds", type=int, default=None)
    m.set_defaults(func=_cmd_monitor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
