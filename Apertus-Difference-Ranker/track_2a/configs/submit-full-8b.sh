#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PROJECT_DIR MODEL_TAG=8b MODEL_NAME=swiss-ai/Apertus-v1.5-8B
scan=$(sbatch --parsable --export=ALL configs/scan.sbatch)
prepare=$(sbatch --parsable --dependency="afterok:${scan}" --export=ALL configs/prepare-representations.sbatch)
gb=$(sbatch --parsable --dependency="afterok:${scan}" --export=ALL,LINGUISTIC=0 configs/train-gradient-boosting.sbatch)
gb_ling=$(sbatch --parsable --dependency="afterok:${scan}" --export=ALL,LINGUISTIC=1 configs/train-gradient-boosting.sbatch)
mlp=$(sbatch --parsable --dependency="afterok:${prepare}" --export=ALL,LINGUISTIC=0 configs/train-representation.sbatch)
mlp_ling=$(sbatch --parsable --dependency="afterok:${prepare}" --export=ALL,LINGUISTIC=1 configs/train-representation.sbatch)
test=$(sbatch --parsable --dependency="afterok:${gb}:${gb_ling}:${mlp}:${mlp_ling}" --export=ALL configs/evaluate-final.sbatch)
printf '8B scan: %s\n8B prepare: %s\n8B GB: %s\n8B GB+tags: %s\n8B MLP: %s\n8B MLP+tags: %s\n8B test: %s\n' \
  "$scan" "$prepare" "$gb" "$gb_ling" "$mlp" "$mlp_ling" "$test"
