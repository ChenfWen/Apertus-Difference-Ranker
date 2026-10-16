from __future__ import annotations

from collections import defaultdict

import numpy as np


def stratified_subset(examples, max_documents: int | None, seed: int):
    if max_documents is None or max_documents >= len(examples):
        return list(examples)
    groups = defaultdict(list)
    for example in examples:
        groups[example.language].append(example)
    rng = np.random.default_rng(seed)
    selected = []
    languages = sorted(groups)
    base, remainder = divmod(max_documents, len(languages))
    for index, language in enumerate(languages):
        values = list(groups[language])
        rng.shuffle(values)
        selected.extend(values[: min(len(values), base + (index < remainder))])
    order = {example.example_id: index for index, example in enumerate(examples)}
    return sorted(selected, key=lambda example: order[example.example_id])


def document_split(examples, validation_per_language: int, seed: int):
    groups = defaultdict(list)
    for example in examples:
        groups[example.language].append(example.example_id)
    rng = np.random.default_rng(seed)
    validation = set()
    for language, identifiers in sorted(groups.items()):
        identifiers = list(identifiers)
        rng.shuffle(identifiers)
        count = min(validation_per_language, max(1, round(0.1 * len(identifiers))))
        if len(identifiers) > validation_per_language:
            count = validation_per_language
        if len(identifiers) == 1:
            count = 0
        validation.update(identifiers[:count])
    train = {example.example_id for example in examples} - validation
    if not train:
        raise ValueError("Document split produced an empty training set")
    return train, validation
