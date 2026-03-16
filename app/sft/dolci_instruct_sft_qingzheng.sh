#!/bin/bash
# Dolci Instruct SFT - SLURM 提交脚本 (6 nodes × 8 H200)
# 使用 flash-fish 的 stool.py 提交到 SLURM

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ms-swift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SWIFT_ROOT"

config_name="dolci_instruct_sft_qingzheng"
CONFIG_YAML="${SCRIPT_DIR}/${config_name}.yaml"

# Auto-resume: find the latest checkpoint in output_dir
OUTPUT_DIR="${SWIFT_ROOT}/output/qwen3-omni-dolci-lora-qingzheng"
RESUME_ARG=""
LAST_CKPT=$(ls -dt "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LAST_CKPT" ]; then
    echo "Auto-resume: found last checkpoint at ${LAST_CKPT}"
    RESUME_ARG="--resume_from_checkpoint ${LAST_CKPT}"
else
    echo "Auto-resume: no checkpoint found, training from scratch."
fi

# Job parameters
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
    --cmd "-m swift.cli.sft --config ${CONFIG_YAML} ${RESUME_ARG}" \
    --submit \
    --conda-script ~/miniconda3/etc/profile.d/conda.sh \
    --conda-env ~/miniconda3/envs/ms-swift \
    --envs "MAX_PIXELS=1003520" \
    --envs "VIDEO_MAX_PIXELS=50176" \
    --envs "FPS_MAX_FRAMES=12" \
    --envs "USE_HF=1" \
    --envs "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800" \
    --envs "TORCH_CUDA_ALLOC_CONF=expandable_segments:True"
