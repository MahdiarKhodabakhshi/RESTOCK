#!/usr/bin/env bash
# Efficiency-RL (GRPO) on Saleor 3.20 from the memorised SFT weights.
#
#   sbatch scripts/grpo_efficiency.sh
#
# Requires Saleor 3.20 up:  uv run restock saleor up 3.20
#SBATCH --job-name=grpo-efficiency
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=data/logs/grpo_efficiency_%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p data/logs data/models

PYTHON=${PYTHON:-data/envs/vllm/bin/python}
export TOKENIZERS_PARALLELISM=false
export TRL_EXPERIMENTAL_SILENCE=1

exec "$PYTHON" scripts/grpo_efficiency.py \
    --tasks data/tasks/train.jsonl \
    --model data/models/memorised-3.20/merged \
    --out data/models/efficiency-3.20 \
    --release 3.20 \
    --max-steps 200 \
    --num-generations 4 \
    "$@"
