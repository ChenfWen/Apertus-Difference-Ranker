#!/bin/bash
#SBATCH --account=g238
#SBATCH --job-name=ap15-final-smoke
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=ap15-final-smoke-%j.out
#SBATCH --error=ap15-final-smoke-%j.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the track_2a project root}}"
cd "$PROJECT_DIR"
source .venv/bin/activate
export HF_HOME="${SCRATCH}/huggingface"
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
tag=smoke-8b
model=swiss-ai/Apertus-v1.5-8B

apertus-final-rsd scan --model-name "$model" --split dev --max-documents 9 \
  --validation-per-language 1 --top-layers 3 --output-dir "artifacts/scan-${tag}"
apertus-final-rsd prepare-representations --model-name "$model" --split dev \
  --scan-dir "artifacts/scan-${tag}" --output-dir "artifacts/representations-${tag}"
apertus-final-rsd train-gradient-boosting --scan-dir "artifacts/scan-${tag}" \
  --max-iterations 20 --output-dir "models/gradient-boosting-${tag}"
apertus-final-rsd train-gradient-boosting --linguistic \
  --scan-dir "artifacts/scan-${tag}" --max-iterations 20 \
  --output-dir "models/gradient-boosting-ling-${tag}"
apertus-final-rsd train-representation --scan-dir "artifacts/scan-${tag}" \
  --representation-dir "artifacts/representations-${tag}" --epochs 3 --patience 2 \
  --output-dir "models/representation-${tag}"
apertus-final-rsd train-representation --linguistic \
  --scan-dir "artifacts/scan-${tag}" \
  --representation-dir "artifacts/representations-${tag}" --epochs 3 --patience 2 \
  --output-dir "models/representation-ling-${tag}"
apertus-final-rsd evaluate-final --model-name "$model" --split dev --max-documents 3 \
  --scan-dir "artifacts/scan-${tag}" \
  --representation-model-dir "models/representation-${tag}" \
  --gradient-boosting-model-dir "models/gradient-boosting-${tag}" \
  --representation-ling-model-dir "models/representation-ling-${tag}" \
  --gradient-boosting-ling-model-dir "models/gradient-boosting-ling-${tag}" \
  --output-dir "results/smoke-final-${tag}"
