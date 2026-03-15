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
#SBATCH --output=results/off_policy_distill/logs/vllm_teacher_%j.out
#SBATCH --error=results/off_policy_distill/logs/vllm_teacher_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/ms-swift

# Avoid cuBLAS version mismatch
unset LD_LIBRARY_PATH

PORT=${1:-8000}
TP_SIZE=${2:-8}
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Port: $PORT"
echo "TP size: $TP_SIZE"
echo "Model: $MODEL"
echo "Started at: $(date)"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-logprobs 20 \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code
