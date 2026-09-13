#!/usr/bin/env bash
# Serve a model with vLLM on one GPU, as a Slurm job:
#
#   sbatch scripts/serve_model.sh Qwen/Qwen3.8-27B teacher 8100
#
# The server listens on 127.0.0.1:<port> and answers to the name <served-name>.
#SBATCH --job-name=serve-model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=data/logs/serve_%j.log
set -euo pipefail

MODEL=${1:?model}
NAME=${2:?served name}
PORT=${3:?port}
VLLM=${VLLM:-data/envs/vllm/bin/vllm}

exec "$VLLM" serve "$MODEL" \
    --served-model-name "$NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization 0.92
