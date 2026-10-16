# Apertus Difference Ranker - Short Method Description

**Track:** Hack Apertus Track 2A - UZH SwissGov-RSD  
**Task:** Rank token-level semantic differences in related multilingual government documents  
**Primary backbone:** `swiss-ai/Apertus-v1.5-8B`

## Overview

Apertus Difference Ranker combines frozen Apertus representations with small
supervised probes. The method first searches for semantic-sensitive Transformer
layers, then extracts token-level cross-document evidence, and finally fits
lightweight rankers on the SwissGov-RSD development split. Apertus is never
fine-tuned, and the official test split is used only for the final evaluation.

![Three-stage method](docs/paradigm_tikz.png)

## 1. Semantic-sensitive layer search

Each side of a bilingual document pair is encoded independently with frozen
Apertus. SwissGov whitespace tokens are mapped to Apertus subwords and pooled
back to the benchmark tokenization. At every Transformer layer, raw DiffAlign
assigns source token `i` the score

```text
d_l(i) = 1 - max_j cosine(h_l(a_i), h_l(b_j)).
```

The operation is repeated in both language directions. Layer sensitivity is
measured with macro Spearman on the 453-document fitting partition only. The
8B system selects layer 14 for paired representations and layers 14-18 for
engineered features. The controlled 70B run selects representation layer 17
and engineered layers 17, 19, 18, 16, and 20.

## 2. Representation and feature extraction

The project evaluates two trainable branches:

1. **Paired-representation MLP.** Each source state is paired with its
   cosine-nearest target state. The probe receives the source state, target
   state, absolute difference, and elementwise product. A two-layer MLP maps
   this vector to a continuous difference score in `[0, 1]`.
2. **Engineered gradient boosting.** For each of five selected layers, the
   system computes eight alignment statistics: nearest cosine distance, mean
   top-5 distance, top-1/top-2 margin, normalized alignment entropy,
   cosine-row mean, cosine-row standard deviation, mutual-nearest-neighbour
   status, and source-state norm. Local-context distances and four static token
   features produce a 54-dimensional vector for histogram gradient boosting.

Both branches also have tagged variants that append 23 fixed-width indicators:
six coarse NER categories and seventeen Universal POS categories. These tags
come from frozen auxiliary taggers; all semantic representations and alignment
features come from Apertus.

## 3. Lightweight fitting and controlled evaluation

The 504 SwissGov-RSD development documents are split by document and language
into 453 fitting documents and 51 validation documents. Positive-difference
tokens receive higher sample weight to reduce label imbalance. The MLP uses a
weighted regression and ranking objective with validation-based early stopping.
The gradient booster uses 300 iterations with fixed hyperparameters. Layer
selection, fitting, early stopping, and model choice use development data only.

The frozen models are then applied once to all 168 official test documents,
covering 132,080 non-punctuation tokens. Performance is the unweighted mean of
token-level Spearman correlations for English-German, English-French, and
English-Italian.

## Held-out results

| Backbone and method | Test macro Spearman | Test pooled Spearman |
| --- | ---: | ---: |
| 8B raw DiffAlign | 0.2632 | 0.2672 |
| 8B paired-representation MLP | 0.3092 | 0.3199 |
| 8B engineered gradient boosting | 0.3247 | 0.3296 |
| 8B paired-representation MLP + POS/NER | 0.3190 | 0.3250 |
| **8B engineered gradient boosting + POS/NER** | **0.3275** | **0.3325** |
| 70B raw DiffAlign | 0.2737 | 0.2786 |
| 70B best lightweight probe | 0.3168 | 0.3244 |

The submitted 8B system improves macro Spearman by **0.0643 absolute** over raw
8B DiffAlign. The 70B backbone slightly improves raw alignment, but its best
probe does not surpass the smaller model. The final system therefore favors the
8B engineered-feature ranker with POS/NER as the strongest and most efficient
configuration.

## Reproduction

Run the frozen result replay from the `track_2a` directory:

```bash
make run
```

Full GPU reproduction instructions, exact model revisions, and Clariden job
commands are provided in `docs/REPRODUCE.md`. A detailed discussion of
architecture, evaluation, and limitations is available in
`technical_report.md` and `technical_report.pdf`.
