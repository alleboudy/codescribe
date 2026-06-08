"""Path B builder tests (torchrun command + NCCL env + LR scaling)."""

from __future__ import annotations

import shlex

import pytest

from codescribe_fleet.ddp import (
    DDPNode,
    build_torchrun_command,
    effective_batch_size,
    nccl_env,
    scale_lr_for_nodes,
)


def test_torchrun_command_shape() -> None:
    node = DDPNode(node_rank=2, nnodes=4, rdzv_host="dell-precision-01")
    cmd = build_torchrun_command(
        node,
        train_module="codescribe_train.train",
        config="configs/train/qwen7b_qlora.yaml",
        dataset="datasets/repo/",
        out="checkpoints/v1/",
    )
    assert "--nnodes=4" in cmd
    assert "--node-rank=2" in cmd
    assert "--rdzv-endpoint=dell-precision-01:29400" in cmd
    assert "--rdzv-backend=c10d" in cmd
    assert "-m codescribe_train.train run" in cmd
    # command is shell-parseable (quoting is balanced)
    assert "torchrun" in shlex.split(cmd)


def test_torchrun_rejects_bad_rank() -> None:
    with pytest.raises(ValueError):
        build_torchrun_command(
            DDPNode(node_rank=4, nnodes=4, rdzv_host="h"),
            train_module="m",
            config="c",
            dataset="d",
            out="o",
        )


def test_nccl_env_consumer_hardware() -> None:
    env = nccl_env("eno1")
    assert env["NCCL_IB_DISABLE"] == "1"
    assert env["NCCL_NET"] == "Socket"
    assert env["NCCL_SOCKET_IFNAME"] == "eno1"
    assert env["TORCH_DIST_INIT_BARRIER_TIMEOUT"] == "600"
    assert "NCCL_DEBUG" in env


def test_nccl_env_debug_off() -> None:
    assert "NCCL_DEBUG" not in nccl_env("eno1", debug=False)


def test_scale_lr_sqrt() -> None:
    assert scale_lr_for_nodes(2e-4, 4) == pytest.approx(1e-4)
    assert scale_lr_for_nodes(2e-4, 1) == pytest.approx(2e-4)
    with pytest.raises(ValueError):
        scale_lr_for_nodes(2e-4, 0)


def test_effective_batch_size() -> None:
    assert effective_batch_size(1, 16, 4) == 64
