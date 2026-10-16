# Frozen prediction deliverables

This directory contains the exact token-level prediction files produced by the
completed held-out evaluations:

- `final_test_8b_predictions.csv.gz` — primary Apertus-v1.5-8B submission;
- `final_test_70b_predictions.csv.gz` — controlled Apertus-v1.5-70B scaling
  ablation.

The corresponding aggregate metrics are stored in
`../results/final_test_8b_summary.json` and
`../results/final_test_70b_summary.json`. The prediction files are outputs of
the frozen evaluation pipeline and are not reconstructed from those summaries.

