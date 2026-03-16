#!/bin/bash
# Launch vLLM teacher server via SLURM
# Usage: sbatch vllm_teacher.sh [port] [tp_size]

#SBATCH --job-name=vllm-teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/qingzhengw/ms-swift/results/off_policy_distill/logs/vllm_teacher_%j.out
#SBATCH --error=/home/qingzhengw/ms-swift/results/off_policy_distill/logs/vllm_teacher_%j.err

set -e

# Activate uv environment (do NOT unset LD_LIBRARY_PATH - gpu nodes need it for libcudart)
source /home/qingzhengw/ms-swift/.venv/bin/activate

# Fix nvcc permission on gpu nodes: copy to writable location with +x
# NVCC_ORIG=$(which nvcc 2>/dev/null || find /usr/local/cuda/bin -name nvcc 2>/dev/null | head -1)
# if [ -n "$NVCC_ORIG" ] && [ ! -x "$NVCC_ORIG" ]; then
#     mkdir -p /tmp/nvcc_fix
#     cp "$NVCC_ORIG" /tmp/nvcc_fix/nvcc
#     chmod +x /tmp/nvcc_fix/nvcc
#     export PATH="/tmp/nvcc_fix:$PATH"
#     echo "Fixed nvcc permission: copied $NVCC_ORIG to /tmp/nvcc_fix/nvcc"
# fi

SWIFT_ROOT="/home/qingzhengw/ms-swift"
config_name="off_policy_distill"
CONFIG_YAML="${SWIFT_ROOT}/app/${config_name}/${config_name}.yaml"

PORT=${1:-8848}
TP_SIZE=${2:-8}
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"

# Read max-logprobs from the shared YAML config (gkd_logits_topk)
MAX_LOGPROBS=$(python -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG_YAML}'))
print(cfg.get('gkd_logits_topk', 20))
")

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG_YAML"
echo "Port: $PORT"
echo "TP size: $TP_SIZE"
echo "Model: $MODEL"
echo "Max logprobs: $MAX_LOGPROBS"
echo "Started at: $(date)"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-logprobs "$MAX_LOGPROBS" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code

# vllm serve "Qwen/Qwen3-235B-A22B-Instruct-2507" \
#     --port "8848" \
#     --max-logprobs "20" \
#     --tensor-parallel-size "8" \
#     --trust-remote-code
