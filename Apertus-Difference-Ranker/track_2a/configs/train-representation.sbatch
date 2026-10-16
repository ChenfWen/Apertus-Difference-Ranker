#!/bin/bash
#SBATCH --account=g238
#SBATCH --job-name=ap15-final-mlp
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=ap15-final-mlp-%j.out
#SBATCH --error=ap15-final-mlp-%j.err
set -euo pipefail
: "${MODEL_TAG:?MODEL_TAG is required}"
: "${LINGUISTIC:?LINGUISTIC must be 0 or 1}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the track_2a project root}}"
cd "$PROJECT_DIR"
source .venv/bin/activate
suffix=""
flag=()
if [[ "$LINGUISTIC" == "1" ]]; then suffix="-ling"; flag=(--linguistic); fi
apertus-final-rsd train-representation \
  --scan-dir "artifacts/scan-${MODEL_TAG}" \
  --representation-dir "artifacts/representations-${MODEL_TAG}" \
  --output-dir "models/representation${suffix}-${MODEL_TAG}" \
  "${flag[@]}"
