from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import load_examples
from .evaluation import score_predictions
from .features import direction_features, static_features, vector_matrix
from .model import ApertusExtractor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorTestConfig:
    scan_dir: Path
    model_dir: Path
    output_dir: Path
    model_name: str = "swiss-ai/Apertus-v1.5-8B"
    dataset_name: str = "ZurichNLP/SwissGov-RSD"
    dataset_revision: str | None = None
    split: str = "test"
    max_documents: int | None = None
    max_length: int = 8192
    dtype: str = "auto"
    force: bool = False


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def _fingerprint(config: VectorTestConfig, layers: list[int]) -> str:
    model_path = config.model_dir / "model.joblib"
    payload = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "split": config.split,
        "max_length": config.max_length,
        "layers": layers,
        "trained_model_size": model_path.stat().st_size,
        "trained_model_mtime_ns": model_path.stat().st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _load(path: Path, fingerprint: str):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["fingerprint"].item()) != fingerprint:
                return None
            return {key: archive[key].copy() for key in archive.files}
    except (OSError, KeyError, ValueError):
        return None


def _predict_example(example, extractor, estimator, layers: list[int], fingerprint: str):
    left, right = extractor.extract(example.text_a), extractor.extract(example.text_b)
    features_a = direction_features(left, right, layers, extractor.torch)
    features_b = direction_features(right, left, layers, extractor.torch)
    static_a, static_b = static_features(left, False), static_features(right, True)
    sides = np.asarray(["a"] * len(left.words) + ["b"] * len(right.words), dtype=str)
    features = np.concatenate((features_a, features_b)).astype(np.float32)
    static = np.concatenate((static_a, static_b)).astype(np.float32)
    matrix = vector_matrix(features, static, list(range(len(layers))), sides)
    prediction = np.clip(estimator.predict(matrix), 0.0, 1.0).astype(np.float32)
    gold = np.concatenate((example.labels_a, example.labels_b)).astype(np.float32)
    return {
        "fingerprint": np.asarray(fingerprint),
        "document_id": np.asarray(example.example_id),
        "language": np.asarray(example.language),
        "words": np.asarray(left.words + right.words, dtype=str),
        "sides": sides,
        "gold": gold,
        "prediction": prediction,
    }


def _write_predictions(path: Path, documents: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("document_id", "language", "side", "token_index", "word", "gold", "prediction")
        )
        for item in documents:
            counters = {"a": 0, "b": 0}
            for word, side, gold, prediction in zip(
                item["words"], item["sides"], item["gold"], item["prediction"], strict=True
            ):
                token_index = counters[str(side)]
                counters[str(side)] += 1
                writer.writerow(
                    (
                        str(item["document_id"].item()), str(item["language"].item()),
                        str(side), token_index, str(word), float(gold), float(prediction),
                    )
                )


def run_vector_test(config: VectorTestConfig) -> dict:
    import joblib

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if config.force:
        for path in checkpoint_dir.glob("*.npz"):
            path.unlink()

    selection = json.loads((config.scan_dir / "promising_layers.json").read_text(encoding="utf-8"))
    layers = [int(layer) for layer in selection["vector_layers"]]
    estimator = joblib.load(config.model_dir / "model.joblib")
    examples, resolved = load_examples(config.dataset_name, config.dataset_revision, config.split)
    if config.max_documents is not None:
        examples = examples[: config.max_documents]
    fingerprint = _fingerprint(config, layers)

    documents, pending = [], []
    for index, example in enumerate(examples):
        path = checkpoint_dir / f"{index:06d}-{_safe(example.example_id)}.npz"
        existing = _load(path, fingerprint)
        if existing is None:
            pending.append((example, path))
        else:
            documents.append(existing)

    extractor = ApertusExtractor(
        config.model_name, tuple(layers), config.max_length, config.dtype, "float32"
    ) if pending else None
    errors = []
    for example, path in tqdm(pending, desc="Evaluate vector model on test"):
        try:
            payload = _predict_example(example, extractor, estimator, layers, fingerprint)
            _save(path, payload)
            documents.append(payload)
        except Exception as exc:
            LOGGER.exception("Failed test document %s", example.example_id)
            errors.append({"document_id": example.example_id, "error": str(exc)})

    with (config.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    if not documents:
        raise RuntimeError("No test documents were evaluated successfully")
    documents.sort(key=lambda item: str(item["document_id"].item()))
    gold = np.concatenate([item["gold"] for item in documents])
    prediction = np.concatenate([item["prediction"] for item in documents])
    languages = np.concatenate([
        np.full(len(item["gold"]), str(item["language"].item())) for item in documents
    ])
    metrics = score_predictions(gold, prediction, languages)
    _write_predictions(config.output_dir / "test_predictions.csv.gz", documents)
    summary = {
        "requested_split": config.split,
        "resolved_split": resolved,
        "input_documents": len(examples),
        "processed_this_run": len(pending),
        "resumed_documents": len(documents) - len(pending) + len(errors),
        "successful_documents": len(documents),
        "failed_documents": len(errors),
        "model_name": config.model_name,
        "trained_model": str((config.model_dir / "model.joblib").resolve()),
        "selected_layers_from_development_only": layers,
        "test_data_used_for_training_or_selection": False,
        "model_forward_passes_this_run": 0 if extractor is None else extractor.forward_passes,
        "test_metrics": metrics,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return summary
