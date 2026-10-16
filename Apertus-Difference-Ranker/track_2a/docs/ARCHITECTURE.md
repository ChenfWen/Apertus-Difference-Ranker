# Architecture

```mermaid
flowchart LR
    D[SwissGov-RSD bilingual document] --> A[Frozen Apertus v1.5]
    A --> H[All hidden layers]
    H --> S[Training-only layer scan]
    S --> R[Representation layer 14]
    S --> F[Feature layers 14–18]
    R --> P[Source and nearest-target states]
    P --> M1[Representation MLP]
    F --> X[54 alignment/context features]
    X --> G1[Histogram gradient boosting]
    D --> T[Frozen POS and NER taggers]
    T --> L[23 one-hot linguistic features]
    L --> M2[Representation MLP + tags]
    L --> G2[Gradient boosting + tags]
    P --> M2
    X --> G2
    H --> B[Raw DiffAlign]
    B --> E[Five frozen token-score streams]
    M1 --> E
    G1 --> E
    M2 --> E
    G2 --> E
    E --> Q[DE / FR / IT Spearman and macro]
```

## Leakage boundary

```text
SwissGov development split (504 document pairs)
  ├── 453 training documents
  │     ├── layer selection
  │     └── lightweight-model fitting
  └── 51 validation documents (17 per language)
        ├── MLP early stopping
        └── development diagnostics

SwissGov test split (168 document pairs)
  └── frozen evaluation after all choices are fixed
```

Layer selection uses only the training documents. The test split is never read
by the scanner or training commands. POS/NER schemas are fixed constants, and
punctuation is identified from observable input tokens and POS predictions.

## Scaling ablation

The diagram describes the primary 8B system (representation layer 14 and
feature layers 14–18). The frozen 70B ablation uses the same graph and data
boundary, with representation layer 17 and feature layers 17, 19, 18, 16, and
20 selected exclusively from development data. It is evaluated as a separate
system; no test-tuned 8B/70B ensemble is constructed.

