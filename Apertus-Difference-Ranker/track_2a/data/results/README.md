# Frozen experiment results

This directory preserves the completed experiment evidence used by the report:

- held-out test summaries for Apertus-v1.5-8B and Apertus-v1.5-70B;
- full-development summaries for both model sizes;
- per-layer train, validation, and all-development Spearman tables;
- selected-layer manifests and scan configurations;
- lightweight-model training summaries.

Large representation caches, model checkpoints, and gated Apertus weights are
not redistributed. Exact reproduction instructions are in
`../../docs/REPRODUCE.md`.
