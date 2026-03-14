#!/bin/bash
# Off-Policy Distillation (GKD) - SLURM launch script
# Uses flash-fish stool.py for multi-node submission

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ms-swift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

CONFIG_YAML="${SCRIPT_DIR}/off_policy_distill.yaml"

# Job parameters
config_name="off_policy_distill"
ngpu=${1:-8}
nodes=${2:-2}
wall_time=${3:-96:00:00}
partition=${4:-"gpu"}

python tools/slurm/stool.py \
    --nodes $nodes \
    --ngpu $ngpu \
    --project ${config_name} \
    --time $wall_time \
    --partition $partition \
    --no-auto-inject-cmd \
    --cmd "-m swift.cli.rlhf --config ${CONFIG_YAML}" \
    --submit \
    --conda-script ~/miniconda3/etc/profile.d/conda.sh \
    --conda-env ~/miniconda3/envs/ms-swift \
    --envs "MAX_PIXELS=1003520" \
    --envs "VIDEO_MAX_PIXELS=50176" \
    --envs "FPS_MAX_FRAMES=12" \
    --envs "USE_HF=1" \
    --envs "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800" \
    --envs "TORCH_CUDA_ALLOC_CONF=expandable_segments:True"
