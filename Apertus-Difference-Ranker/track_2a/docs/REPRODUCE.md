# Reproduction guide

## 1. Lightweight result replay

From `track_2a/`:

```bash
make run
```

This builds a dependency-free Docker image, validates the frozen 8B and 70B
test summaries, recomputes every macro score, and prints both five-system
rankings. It does not download gated weights.

Without Docker, the same check is:

```bash
make replay
```

## 2. Full Clariden experiment

### Requirements

- approved Hugging Face access to `swiss-ai/Apertus-v1.5-8B` and, optionally,
  `swiss-ai/Apertus-v1.5-70B`;
- CSCS Clariden `pytorch/v2.9.1:v2` uenv;
- Hugging Face token at `${HOME}/.cache/huggingface/token`;
- model cache at `${SCRATCH}/huggingface`;
- sufficient scratch space for scan checkpoints and representation caches.

### Environment

From a fresh login shell:

```bash
cd /users/cwen/projects/<submission-repository>/track_2a
uenv start pytorch/v2.9.1:v2 --view=default
bash setup-clariden.sh
source .venv/bin/activate
python -c "import apertus_final_rsd; print(apertus_final_rsd.__version__)"
python -m pytest
deactivate
exit
```

`setup-clariden.sh` already runs the tests once; the explicit invocation is an
optional verification. Submit Slurm jobs from the ordinary login shell, not
from inside `uenv start`, to avoid environment-library conflicts with Slurm.

### Smoke pipeline

```bash
cd /users/cwen/projects/<submission-repository>/track_2a
sbatch configs/smoke-pipeline-8b.sbatch
```

Smoke scores use only a few development documents and are functional checks,
not reportable held-out results.

### Full 8B dependency chain

```bash
bash configs/submit-full-8b.sh
```

The launcher prints seven job IDs: layer scan, representation cache, two
gradient boosters, two representation MLPs, and final test evaluation. The test
job has `afterok` dependencies on all four fitted models.

Final outputs:

```text
results/final-test-8b/run_summary.json
results/final-test-8b/test_predictions.csv.gz
```

Expected invariants:

- `input_documents = successful_documents = 168`;
- `failed_documents = 0`;
- `test_data_used_for_training_or_selection = false`;
- all five methods have `n_tokens_pooled = 132080`;
- best completed 8B macro Spearman is `0.32747887920100066`.

### Layer-analysis artifacts

```text
artifacts/scan-8b/layer_spearman_scores.csv
artifacts/scan-8b/layer_metrics_train.csv
artifacts/scan-8b/layer_metrics_validation.csv
artifacts/scan-8b/layer_metrics_all_dev.csv
artifacts/scan-8b/layer_spearman.png
artifacts/scan-8b/promising_layers.json
```

The training CSV alone controls layer selection. Other curves are diagnostic.

### Full 70B dependency chain

```bash
bash configs/submit-full-70b.sh
```

The completed run must satisfy the same invariants as 8B. The frozen completed
result has best macro Spearman `0.31684805271423544` from the linguistic
representation MLP, while raw DiffAlign reaches `0.27371072881356534`.

Final outputs:

```text
results/final-test-70b/run_summary.json
results/final-test-70b/test_predictions.csv.gz
```

The repository freezes both aggregate summaries for replay. The 8B result is
the primary submission; 70B is the controlled scaling ablation.

## 3. Prediction deliverable

Copy the frozen 8B predictions from Clariden:

```bash
scp cwen@ela.cscs.ch:/users/cwen/projects/apertus-v15-final-rsd/results/final-test-8b/test_predictions.csv.gz \
  data/predictions/
```

The prediction file is not reconstructed from the aggregate summary.

