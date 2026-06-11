#!/usr/bin/env bash
# Path B per-node launcher (codescribe issue #3 §6.2 / §6.5).
#
# Run this on EVERY node, including the rendezvous master. Each node gets a
# distinct NODE_RANK (master = 0). The orchestrator (or a tmux/cssh fan-out)
# SSH-invokes it across the fleet:
#
#   for rank in 0 1 2 3; do
#     ssh fleet-0$((rank+1)) "cd ~/repos/codescribe-fleet && \
#       NODE_RANK=$rank RDZV_HOST=dell-precision-01 NCCL_SOCKET_IFNAME=eno1 \
#       nohup bash scripts/run_ddp.sh > logs/ddp-rank-$rank.log 2>&1 &"
#   done
#
# Generate the exact per-node commands + NCCL env with:
#   python -m codescribe_fleet ddp-plan --fleet configs/fleet.yaml --sweep sweeps/lr-rank.yaml
#
# HONEST WARNING: on 1 Gbps LAN this is ~1.4x wall-clock at ~3x compute. Path A
# (a parallel sweep — `python -m codescribe_fleet sweep`) almost always wins on
# an 8 GB-VRAM laptop fleet. Read issue #3 §3 before using this.
set -euo pipefail

# ---- required / tunable env --------------------------------------------------
NODE_RANK="${NODE_RANK:?set NODE_RANK (0 on the rendezvous master)}"
RDZV_HOST="${RDZV_HOST:?set RDZV_HOST (the rendezvous master hostname/IP)}"
NNODES="${NNODES:-4}"
RDZV_PORT="${RDZV_PORT:-29400}"
RDZV_ID="${RDZV_ID:-ft-001}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

TRAIN_MODULE="${TRAIN_MODULE:-codescribe_train.train}"
CONFIG="${CONFIG:-configs/train/qwen7b_qlora.yaml}"
DATASET="${DATASET:-datasets/example-repo/}"
OUT="${OUT:-checkpoints/ddp-v1/}"
RUNNER="${RUNNER:-uv run}"

# ---- NCCL config that works on consumer hardware (issue #3 §6.3) -------------
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"          # no InfiniBand on a laptop
export NCCL_NET="${NCCL_NET:-Socket}"                   # force TCP transport
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:?set NCCL_SOCKET_IFNAME (e.g. eno1)}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"                 # leave on for first runs
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-NET,GRAPH}"
export TORCH_DIST_INIT_BARRIER_TIMEOUT="${TORCH_DIST_INIT_BARRIER_TIMEOUT:-600}"

echo "run_ddp: rank=$NODE_RANK/$NNODES rdzv=$RDZV_HOST:$RDZV_PORT iface=$NCCL_SOCKET_IFNAME"

exec ${RUNNER} torchrun \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --rdzv-id="${RDZV_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${RDZV_HOST}:${RDZV_PORT}" \
    -m "${TRAIN_MODULE}" run \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --out "${OUT}"
