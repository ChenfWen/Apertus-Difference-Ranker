#!/bin/bash
set -euo pipefail

python -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install \
  "numpy>=2.0,<3" \
  "scipy>=1.12" \
  "datasets==5.0.1" \
  "huggingface-hub==1.5.0" \
  "joblib>=1.4" \
  "matplotlib>=3.8" \
  "scikit-learn>=1.5" \
  "tqdm>=4.66" \
  "accelerate>=1.0" \
  "safetensors==0.8.0" \
  "sentencepiece==0.2.2" \
  "pytest>=8"

python -m pip install --no-deps --force-reinstall \
  "git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest

echo "Setup complete."
