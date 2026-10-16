# Apertus Difference Ranker

**Track 2A — UZH: Detecting Cross-Lingual Semantic Differences in Swiss Government Websites**

Apertus Difference Ranker assigns a continuous semantic-difference score to
every non-punctuation token in aligned English–German, English–French, and
English–Italian government-document pairs. Apertus v1.5 stays frozen; only
small downstream models are trained.

## Headline result

On the untouched SwissGov-RSD test split (168 document pairs and 132,080 scored
tokens), the best system reaches **0.3275 macro Spearman**, compared with
**0.2632** for raw Apertus DiffAlign—a gain of **0.0643 absolute**.

| Method | DE | FR | IT | Macro | Pooled |
|---|---:|---:|---:|---:|---:|
| Raw DiffAlign | 0.2934 | 0.1612 | 0.3350 | 0.2632 | 0.2672 |
| Representation MLP | 0.3413 | 0.2177 | 0.3685 | 0.3092 | 0.3199 |
| Engineered gradient boosting | 0.3678 | 0.2118 | 0.3944 | 0.3247 | 0.3296 |
| Representation MLP + POS/NER | 0.3621 | **0.2188** | 0.3762 | 0.3190 | 0.3250 |
| Engineered gradient boosting + POS/NER | **0.3705** | 0.2158 | **0.3961** | **0.3275** | **0.3325** |

No test document was used for layer selection, fitting, feature selection,
model selection, or early stopping.

### Controlled 70B scaling ablation

The identical protocol was completed with `swiss-ai/Apertus-v1.5-70B`. Raw
DiffAlign improves from 0.2632 to 0.2737 macro, but the best 70B lightweight
system reaches 0.3168 and does not surpass the 8B primary result.

| 70B method | DE | FR | IT | Macro | Pooled |
|---|---:|---:|---:|---:|---:|
| Raw DiffAlign | 0.3048 | 0.1687 | 0.3476 | 0.2737 | 0.2786 |
| Representation MLP | 0.3588 | 0.2178 | 0.3646 | 0.3137 | 0.3201 |
| Engineered gradient boosting | 0.3674 | 0.1996 | 0.3804 | 0.3158 | 0.3204 |
| Representation MLP + POS/NER | 0.3555 | **0.2253** | 0.3698 | **0.3168** | **0.3244** |
| Engineered gradient boosting + POS/NER | **0.3677** | 0.2007 | **0.3812** | 0.3165 | 0.3213 |

The 8B system was prespecified as the primary submission and also finishes
0.0106 macro points ahead while requiring substantially less computation. The
70B result is reported as a controlled scaling analysis, not used to tune,
select, or ensemble the 8B predictions.

## Method

![Apertus Difference Ranker paradigm](docs/paradigm_tikz.png)

The figure summarizes the complete controlled paradigm: development-only layer
selection and lightweight training, followed by one-time evaluation on the
untouched official test split.

The publication-ready TikZ source for the three-stage method diagram is available in
[`docs/paradigm_tikz.tex`](docs/paradigm_tikz.tex).

1. Encode both complete documents independently with frozen
   `swiss-ai/Apertus-v1.5-8B`.
2. Scan every Transformer layer on development-training documents and retain
   layer 14 for paired states and layers 14–18 for engineered features.
3. Compute bidirectional token-to-document DiffAlign statistics.
4. Train four lightweight alternatives:
   - an MLP over source and nearest-target hidden states;
   - histogram gradient boosting over 54 engineered alignment features;
   - the same MLP with 23 fixed-width POS/NER one-hot features;
   - the same booster with the 23 linguistic features.
5. Freeze every choice and evaluate once on the test split.

The linguistic taggers are auxiliary XLM-R token classifiers. They supply only
coarse POS/NER input features; all cross-lingual semantic representations and
alignment features come from Apertus.

## Run the submission artifact

The official template requires Docker execution from `track_2a/`:

```bash
make run
```

This builds a lightweight replay image, validates the frozen result artifact,
recomputes each macro average, and prints the five-system comparison. The
gated Apertus weights and SwissGov-RSD records are not redistributed.

## Full reproduction

The complete experiment requires approved access to the Apertus v1.5 models,
the Swiss AI Transformers fork, SwissGov-RSD, and a suitable NVIDIA GPU node.
See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the exact Clariden workflow and
[`technical_report.md`](technical_report.md) for methodology and limitations.

## Repository structure

```text
track_2a/
├── Makefile
├── Dockerfile
├── README.md
├── technical_report.md
├── configs/                    # Clariden dependency-chain jobs
├── data/results/               # test/dev summaries, scans, and model summaries
├── data/predictions/           # frozen 8B and 70B token predictions
├── docs/                       # architecture and reproduction notes
├── src/apertus_final_rsd/      # complete experiment implementation
├── src/submission_replay.py    # Docker replay/integrity checks
└── tests/
```

## Data and licensing

The project uses
[`ZurichNLP/SwissGov-RSD`](https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD),
licensed CC BY 4.0. Dataset contents and gated model weights are not vendored.
Code is Apache-2.0 via the repository-level `LICENSE`; this documentation is
CC BY 4.0.

