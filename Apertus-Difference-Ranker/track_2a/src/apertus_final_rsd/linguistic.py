from __future__ import annotations

import numpy as np

from .ner import COARSE_LABELS
from .pos import UPOS_LABELS


NER_LABELS = (*COARSE_LABELS, "PUNCT")
POS_LABELS = UPOS_LABELS
NER_INDEX = {label: index for index, label in enumerate(NER_LABELS)}
POS_INDEX = {label: index for index, label in enumerate(POS_LABELS)}
LINGUISTIC_DIMENSION = len(NER_LABELS) + len(POS_LABELS)


def normalize_linguistic_labels(
    ner_labels: list[str] | np.ndarray,
    pos_labels: list[str] | np.ndarray,
    punctuation_mask: list[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable coarse labels and force punctuation into its own category."""
    if len(ner_labels) != len(punctuation_mask) or len(pos_labels) != len(punctuation_mask):
        raise ValueError("Linguistic label count differs from the SwissGov word count")
    # Object dtype prevents assignments such as "PUNCT" from being truncated
    # to the width of a shorter input label such as "O" or "ORG".
    ner = np.asarray([str(value) for value in ner_labels], dtype=object)
    pos = np.asarray([str(value) for value in pos_labels], dtype=object)
    ner[~np.isin(ner, NER_LABELS)] = "MISC"
    pos[~np.isin(pos, POS_LABELS)] = "X"
    punctuation = np.asarray(punctuation_mask, dtype=bool)
    ner[punctuation] = "PUNCT"
    pos[punctuation] = "PUNCT"
    return ner.astype(str), pos.astype(str)


def observable_punctuation_mask(
    words: list[str] | np.ndarray,
    pos_labels: list[str] | np.ndarray,
) -> np.ndarray:
    """Identify punctuation without consulting SwissGov gold labels."""
    if len(words) != len(pos_labels):
        raise ValueError("Word and POS label counts differ")
    return np.asarray(
        [
            str(pos) == "PUNCT" or not any(character.isalnum() for character in str(word))
            for word, pos in zip(words, pos_labels, strict=True)
        ],
        dtype=bool,
    )


def encode_linguistic_features(
    ner_labels: list[str] | np.ndarray,
    pos_labels: list[str] | np.ndarray,
) -> np.ndarray:
    """One-hot encode NER and UPOS with a schema fixed before test evaluation."""
    if len(ner_labels) != len(pos_labels):
        raise ValueError("NER and POS label counts differ")
    result = np.zeros((len(ner_labels), LINGUISTIC_DIMENSION), dtype=np.float32)
    offset = len(NER_LABELS)
    for row, (ner, pos) in enumerate(zip(ner_labels, pos_labels, strict=True)):
        result[row, NER_INDEX.get(str(ner), NER_INDEX["MISC"])] = 1.0
        result[row, offset + POS_INDEX.get(str(pos), POS_INDEX["X"])] = 1.0
    return result


def linguistic_feature_names() -> list[str]:
    return [*(f"ner_{label}" for label in NER_LABELS), *(f"pos_{label}" for label in POS_LABELS)]
