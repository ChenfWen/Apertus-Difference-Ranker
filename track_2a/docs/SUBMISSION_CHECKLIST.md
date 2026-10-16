# Final submission checklist

- [x] Keep the required `track_2a/` directory name and layout.
- [x] Include complete source code and Clariden configurations.
- [x] Include the frozen aggregate 8B and 70B held-out test results.
- [x] Document all five controlled systems and POS/NER auxiliary models.
- [x] State test isolation explicitly.
- [x] Store per-layer Spearman tables and plotting instructions in the pipeline.
- [x] Make `make run` validate and display the frozen result in Docker.
- [x] Remove team/demo placeholders from the technical report.
- [x] Include the frozen 8B and 70B token-level prediction files in
  `data/predictions/`.
- [ ] Run `make run` from a clean checkout with Docker available.
- [x] Confirm `data/` remains below 100 MB after adding predictions (16.4 MB).
- [ ] Add a frozen Git commit identifier to the submitted release/tag.
- [x] Add 70B results after its complete held-out pipeline succeeds.

