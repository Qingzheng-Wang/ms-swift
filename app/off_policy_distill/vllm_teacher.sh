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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/ms-swift

# Avoid cuBLAS version mismatch
unset LD_LIBRARY_PATH

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
