from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class RSDExample:
    example_id: str
    language: str
    text_a: str
    text_b: str
    labels_a: np.ndarray
    labels_b: np.ndarray


def resolve_split_name(requested: str, available: Iterable[str]) -> str:
    available_set = set(available)
    if requested in available_set:
        return requested
    aliases = {"dev": ("validation", "valid"), "validation": ("dev", "valid")}
    for candidate in aliases.get(requested, ()):
        if candidate in available_set:
            return candidate
    raise ValueError(f"Split {requested!r} not found; available: {sorted(available_set)}")


def language_from_example(example: dict[str, Any]) -> str:
    for field in ("subset", "id"):
        pieces = set(filter(None, re.split(r"[^a-z]+", str(example.get(field, "")).lower())))
        matches = pieces & {"de", "fr", "it"}
        if len(matches) == 1:
            return matches.pop()
    raise ValueError(f"Cannot infer language from {example.get('id')!r}")


def _labels(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    result = np.asarray(value, np.float32)
    if result.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    return result


def parse_example(example: dict[str, Any], index: int) -> RSDExample:
    required = {"text_a", "text_b", "labels_a", "labels_b"}
    missing = required - set(example)
    if missing:
        raise ValueError(f"Dataset row lacks fields: {sorted(missing)}")
    result = RSDExample(
        example_id=str(example.get("id", index)),
        language=language_from_example(example),
        text_a=str(example["text_a"]),
        text_b=str(example["text_b"]),
        labels_a=_labels(example["labels_a"]),
        labels_b=_labels(example["labels_b"]),
    )
    if len(result.text_a.split()) != len(result.labels_a):
        raise ValueError(f"{result.example_id}: text_a/labels_a mismatch")
    if len(result.text_b.split()) != len(result.labels_b):
        raise ValueError(f"{result.example_id}: text_b/labels_b mismatch")
    return result


def load_examples(dataset_name: str, revision: str | None, split: str) -> tuple[list[RSDExample], str]:
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision else {}
    dataset_dict = load_dataset(dataset_name, **kwargs)
    resolved = resolve_split_name(split, dataset_dict.keys())
    return [parse_example(row, index) for index, row in enumerate(dataset_dict[resolved])], resolved
