#!/bin/bash
#SBATCH --account=g238
#SBATCH --job-name=ap15-final-dev8b
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=ap15-final-dev8b-%j.out
#SBATCH --error=ap15-final-dev8b-%j.err
set -euo pipefail
export MODEL_TAG="${MODEL_TAG:-8b}"
export MODEL_NAME="${MODEL_NAME:-swiss-ai/Apertus-v1.5-8B}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the track_2a project root}}"
cd "$PROJECT_DIR"
bash configs/evaluate-dev.sbatch
