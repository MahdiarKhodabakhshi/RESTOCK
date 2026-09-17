#!/usr/bin/env bash
# Fine-tune the memorised baseline (LoRA on Qwen3-8B).
#
#   sbatch scripts/sft_memorised.sh
#
#SBATCH --job-name=sft-memorised
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=data/logs/sft_memorised_%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p data/logs data/models

PYTHON=${PYTHON:-data/envs/vllm/bin/python}
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" scripts/sft_memorised.py \
    --data data/sft/memorised_3.20.jsonl \
    --model Qwen/Qwen3-8B \
    --out data/models/memorised-3.20 \
    "$@"
