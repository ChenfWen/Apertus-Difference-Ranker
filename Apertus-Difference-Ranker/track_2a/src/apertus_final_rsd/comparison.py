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
from .features import direction_features, raw_word_pairs, static_features, vector_matrix
from .model import ApertusExtractor
from .training import build_feature_regressor, build_pair_regressor

LOGGER = logging.getLogger(__name__)
SCORES = ("diffalign_raw", "representation_mlp", "engineered_feature_mlp")


@dataclass(slots=True)
class ComparisonConfig:
    scan_dir: Path
    representation_model_dir: Path
    feature_model_dir: Path
    output_dir: Path
    model_name: str = "swiss-ai/Apertus-v1.5-8B"
    dataset_name: str = "ZurichNLP/SwissGov-RSD"
    dataset_revision: str | None = None
    split: str = "test"
    max_documents: int | None = None
    max_length: int = 8192
    dtype: str = "auto"
    batch_size: int = 1024
    force: bool = False


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def _fingerprint(config: ComparisonConfig, layers: list[int]) -> str:
    paths = (
        config.representation_model_dir / "model.pt",
        config.feature_model_dir / "model.pt",
    )
    payload = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "split": config.split,
        "max_length": config.max_length,
        "layers": layers,
        "checkpoints": [
            {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
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


def _head_predictions(model, arrays: tuple[np.ndarray, ...], batch_size: int, torch, device):
    values = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), batch_size):
            tensors = [
                torch.from_numpy(value[start:start + batch_size].astype(np.float32)).to(device)
                for value in arrays
            ]
            values.append(model(*tensors).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _predict_example(
    example, extractor, representation_model, feature_model, feature_checkpoint,
    layers: list[int], representation_layer: int, fingerprint: str, batch_size: int, device,
):
    torch = extractor.torch
    left, right = extractor.extract(example.text_a), extractor.extract(example.text_b)
    features_a = direction_features(left, right, layers, torch)
    features_b = direction_features(right, left, layers, torch)
    static_a, static_b = static_features(left, False), static_features(right, True)
    sides = np.asarray(["a"] * len(left.words) + ["b"] * len(right.words), dtype=str)
    features = np.concatenate((features_a, features_b)).astype(np.float32)
    static = np.concatenate((static_a, static_b)).astype(np.float32)
    matrix = vector_matrix(features, static, list(range(len(layers))), sides)
    normalized = (
        (matrix - np.asarray(feature_checkpoint["feature_mean"], np.float32))
        / np.asarray(feature_checkpoint["feature_scale"], np.float32)
    ).astype(np.float32)

    source_a, target_a = raw_word_pairs(left, right, representation_layer, torch)
    source_b, target_b = raw_word_pairs(right, left, representation_layer, torch)
    source = np.concatenate((source_a, source_b)).astype(np.float32)
    target = np.concatenate((target_a, target_b)).astype(np.float32)
    representation_prediction = _head_predictions(
        representation_model, (source, target), batch_size, torch, device
    )
    feature_prediction = _head_predictions(
        feature_model, (normalized,), batch_size, torch, device
    )
    layer_index = layers.index(representation_layer)
    return {
        "fingerprint": np.asarray(fingerprint),
        "document_id": np.asarray(example.example_id),
        "language": np.asarray(example.language),
        "words": np.asarray(left.words + right.words, dtype=str),
        "sides": sides,
        "gold": np.concatenate((example.labels_a, example.labels_b)).astype(np.float32),
        "diffalign_raw": features[:, layer_index, 0].astype(np.float32),
        "representation_mlp": representation_prediction,
        "engineered_feature_mlp": feature_prediction,
    }


def _write_predictions(path: Path, documents: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("document_id", "language", "side", "token_index", "word", "gold", *SCORES)
        )
        for item in documents:
            counters = {"a": 0, "b": 0}
            rows = zip(
                item["words"], item["sides"], item["gold"],
                *(item[score] for score in SCORES), strict=True,
            )
            for word, side, gold, diffalign, representation, feature in rows:
                token_index = counters[str(side)]
                counters[str(side)] += 1
                writer.writerow(
                    (
                        str(item["document_id"].item()), str(item["language"].item()),
                        str(side), token_index, str(word), float(gold), float(diffalign),
                        float(representation), float(feature),
                    )
                )


def run_comparison(config: ComparisonConfig) -> dict:
    import torch

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if config.force:
        for path in checkpoint_dir.glob("*.npz"):
            path.unlink()

    selection = json.loads((config.scan_dir / "promising_layers.json").read_text(encoding="utf-8"))
    layers = [int(layer) for layer in selection["vector_layers"]]
    representation_layer = int(selection["representation_layer"])
    representation_checkpoint = torch.load(
        config.representation_model_dir / "model.pt", map_location="cpu", weights_only=False
    )
    feature_checkpoint = torch.load(
        config.feature_model_dir / "model.pt", map_location="cpu", weights_only=False
    )
    if [int(layer) for layer in feature_checkpoint["selected_layers"]] != layers:
        raise ValueError("Engineered-feature checkpoint layers differ from scan selection")
    if int(representation_checkpoint["representation_layer"]) != representation_layer:
        raise ValueError("Representation checkpoint layer differs from scan selection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation_model = build_pair_regressor(
        int(representation_checkpoint["hidden_size"]),
        int(representation_checkpoint["hidden_width"]),
        float(representation_checkpoint["dropout"]), torch,
    ).to(device)
    representation_model.load_state_dict(representation_checkpoint["state_dict"])
    representation_model.eval()
    feature_model = build_feature_regressor(
        int(feature_checkpoint["feature_count"]), int(feature_checkpoint["hidden_width"]),
        float(feature_checkpoint["dropout"]), torch,
    ).to(device)
    feature_model.load_state_dict(feature_checkpoint["state_dict"])
    feature_model.eval()

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
    resumed = len(documents)
    extractor = ApertusExtractor(
        config.model_name, tuple(layers), config.max_length, config.dtype, "float32"
    ) if pending else None
    errors = []
    for example, path in tqdm(pending, desc="Controlled RSD comparison"):
        try:
            payload = _predict_example(
                example, extractor, representation_model, feature_model, feature_checkpoint,
                layers, representation_layer, fingerprint, config.batch_size, device,
            )
            _save(path, payload)
            documents.append(payload)
        except Exception as exc:
            LOGGER.exception("Failed comparison document %s", example.example_id)
            errors.append({"document_id": example.example_id, "error": str(exc)})
    with (config.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    if not documents:
        raise RuntimeError("No comparison documents succeeded")
    documents.sort(key=lambda item: str(item["document_id"].item()))
    gold = np.concatenate([item["gold"] for item in documents])
    languages = np.concatenate([
        np.full(len(item["gold"]), str(item["language"].item())) for item in documents
    ])
    metrics = {
        score: score_predictions(
            gold, np.concatenate([item[score] for item in documents]), languages
        )
        for score in SCORES
    }
    _write_predictions(config.output_dir / "comparison_predictions.csv.gz", documents)
    summary = {
        "requested_split": config.split,
        "resolved_split": resolved,
        "input_documents": len(examples),
        "processed_this_run": len(pending),
        "resumed_documents": resumed,
        "successful_documents": len(documents),
        "failed_documents": len(errors),
        "model_name": config.model_name,
        "controlled_scores": list(SCORES),
        "representation_layer": representation_layer,
        "engineered_feature_layers": layers,
        "test_data_used_for_training_or_selection": False,
        "model_forward_passes_this_run": 0 if extractor is None else extractor.forward_passes,
        "metrics": metrics,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return summary
