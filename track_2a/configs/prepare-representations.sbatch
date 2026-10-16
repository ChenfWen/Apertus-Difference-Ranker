#!/bin/bash
#SBATCH --account=g238
#SBATCH --job-name=ap15-final-prepare
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=ap15-final-prepare-%j.out
#SBATCH --error=ap15-final-prepare-%j.err
set -euo pipefail
: "${MODEL_TAG:?MODEL_TAG is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the track_2a project root}}"
cd "$PROJECT_DIR"
source .venv/bin/activate
export HF_HOME="${SCRATCH}/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
apertus-final-rsd prepare-representations \
  --model-name "$MODEL_NAME" --split dev \
  --scan-dir "artifacts/scan-${MODEL_TAG}" \
  --output-dir "artifacts/representations-${MODEL_TAG}"
