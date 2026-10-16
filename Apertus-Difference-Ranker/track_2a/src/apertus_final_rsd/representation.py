from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import load_examples
from .features import raw_word_pairs
from .model import ApertusExtractor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RepresentationConfig:
    scan_dir: Path
    output_dir: Path
    model_name: str = "swiss-ai/Apertus-v1.5-8B"
    dataset_name: str = "ZurichNLP/SwissGov-RSD"
    dataset_revision: str | None = None
    split: str = "dev"
    max_length: int = 4096
    dtype: str = "auto"
    force: bool = False


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def _fingerprint(config: RepresentationConfig, layer: int):
    scan_config = config.scan_dir / "run_config.json"
    payload = {
        "model_name": config.model_name, "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision, "split": config.split,
        "max_length": config.max_length, "layer": layer,
        "scan_config_sha256": (
            hashlib.sha256(scan_config.read_bytes()).hexdigest()
            if scan_config.is_file() else "missing"
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        # Deliberately uncompressed: training repeatedly reads these arrays.
        np.savez(handle, **payload)
    os.replace(temporary, path)


def prepare_example(example, extractor, layer: int, fingerprint: str, linguistic):
    left, right = extractor.extract(example.text_a), extractor.extract(example.text_b)
    source_a, target_a = raw_word_pairs(left, right, layer, extractor.torch)
    source_b, target_b = raw_word_pairs(right, left, layer, extractor.torch)
    gold = np.concatenate((example.labels_a, example.labels_b)).astype(np.float32)
    if len(linguistic["ner_labels"]) != len(gold):
        raise RuntimeError("Scan linguistic labels differ from representation word count")
    return {
        "fingerprint": np.asarray(fingerprint),
        "document_id": np.asarray(example.example_id),
        "language": np.asarray(example.language),
        "source": np.concatenate((source_a, source_b)).astype(np.float16),
        "target": np.concatenate((target_a, target_b)).astype(np.float16),
        "gold": gold,
        "sides": np.asarray(["a"] * len(source_a) + ["b"] * len(source_b), dtype=str),
        "ner_labels": linguistic["ner_labels"],
        "pos_labels": linguistic["pos_labels"],
    }


def run_prepare(config: RepresentationConfig) -> dict:
    selection = json.loads((config.scan_dir / "promising_layers.json").read_text(encoding="utf-8"))
    manifest = json.loads((config.scan_dir / "split_manifest.json").read_text(encoding="utf-8"))
    selected_ids = set(manifest["train_ids"]) | set(manifest["validation_ids"])
    layer = int(selection["representation_layer"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "documents"
    checkpoint_dir.mkdir(exist_ok=True)
    if config.force:
        for path in checkpoint_dir.glob("*.npz"):
            path.unlink()
    examples, resolved = load_examples(config.dataset_name, config.dataset_revision, config.split)
    examples = [example for example in examples if example.example_id in selected_ids]
    linguistic_by_document = {}
    for scan_path in sorted((config.scan_dir / "checkpoints").glob("*.npz")):
        with np.load(scan_path, allow_pickle=False) as item:
            linguistic_by_document[str(item["document_id"].item())] = {
                "ner_labels": item["ner_labels"].copy(),
                "pos_labels": item["pos_labels"].copy(),
            }
    fingerprint = _fingerprint(config, layer)
    pending, resumed = [], 0
    for index, example in enumerate(examples):
        path = checkpoint_dir / f"{index:06d}-{_safe(example.example_id)}.npz"
        valid = False
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as item:
                    valid = str(item["fingerprint"].item()) == fingerprint
            except (OSError, KeyError, ValueError):
                valid = False
        if valid:
            resumed += 1
        else:
            pending.append((example, path))
    extractor = ApertusExtractor(
        config.model_name, (layer,), config.max_length, config.dtype, "float32"
    ) if pending else None
    errors = []
    for example, path in tqdm(pending, desc="Prepare raw Apertus pairs"):
        try:
            _save(
                path,
                prepare_example(
                    example, extractor, layer, fingerprint,
                    linguistic_by_document[example.example_id],
                ),
            )
        except Exception as exc:
            LOGGER.exception("Failed %s", example.example_id)
            errors.append({"document_id": example.example_id, "error": str(exc)})
    with (config.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    summary = {
        "resolved_split": resolved,
        "documents": len(examples),
        "processed_this_run": len(pending),
        "resumed_documents": resumed,
        "failed_documents": len(errors),
        "model_name": config.model_name,
        "representation_layer": layer,
        "hidden_size": None if extractor is None else extractor.hidden_size,
        "model_forward_passes_this_run": 0 if extractor is None else extractor.forward_passes,
        "storage_dtype": "float16",
    }
    if summary["hidden_size"] is None:
        first = next(checkpoint_dir.glob("*.npz"))
        with np.load(first, allow_pickle=False) as item:
            summary["hidden_size"] = int(item["source"].shape[1])
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (config.output_dir / "source_scan_dir.txt").write_text(
        str(config.scan_dir.resolve()) + "\n", encoding="utf-8"
    )
    return summary
