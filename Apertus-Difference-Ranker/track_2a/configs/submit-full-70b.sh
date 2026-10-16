#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PROJECT_DIR MODEL_TAG=70b MODEL_NAME=swiss-ai/Apertus-v1.5-70B
scan=$(sbatch --parsable --export=ALL configs/scan.sbatch)
prepare=$(sbatch --parsable --dependency="afterok:${scan}" --export=ALL configs/prepare-representations.sbatch)
gb=$(sbatch --parsable --dependency="afterok:${scan}" --export=ALL,LINGUISTIC=0 configs/train-gradient-boosting.sbatch)
gb_ling=$(sbatch --parsable --dependency="afterok:${scan}" --export=ALL,LINGUISTIC=1 configs/train-gradient-boosting.sbatch)
mlp=$(sbatch --parsable --dependency="afterok:${prepare}" --export=ALL,LINGUISTIC=0 configs/train-representation.sbatch)
mlp_ling=$(sbatch --parsable --dependency="afterok:${prepare}" --export=ALL,LINGUISTIC=1 configs/train-representation.sbatch)
test=$(sbatch --parsable --dependency="afterok:${gb}:${gb_ling}:${mlp}:${mlp_ling}" --export=ALL configs/evaluate-final.sbatch)
printf '70B scan: %s\n70B prepare: %s\n70B GB: %s\n70B GB+tags: %s\n70B MLP: %s\n70B MLP+tags: %s\n70B test: %s\n' \
  "$scan" "$prepare" "$gb" "$gb_ling" "$mlp" "$mlp_ling" "$test"
