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
from .linguistic import (
    encode_linguistic_features,
    normalize_linguistic_labels,
    observable_punctuation_mask,
)
from .model import ApertusExtractor
from .ner import prepare_ner_cache, text_cache_key
from .pos import prepare_pos_cache
from .training import build_pair_regressor


LOGGER = logging.getLogger(__name__)
SCORES = (
    "diffalign_raw",
    "representation_mlp",
    "engineered_gradient_boosting",
    "representation_ling_mlp",
    "engineered_ling_gradient_boosting",
)


@dataclass(slots=True)
class FinalEvaluationConfig:
    scan_dir: Path
    representation_model_dir: Path
    gradient_boosting_model_dir: Path
    representation_ling_model_dir: Path
    gradient_boosting_ling_model_dir: Path
    output_dir: Path
    model_name: str = "swiss-ai/Apertus-v1.5-8B"
    dataset_name: str = "ZurichNLP/SwissGov-RSD"
    dataset_revision: str | None = None
    split: str = "test"
    max_documents: int | None = None
    max_length: int = 8192
    dtype: str = "auto"
    batch_size: int = 1024
    ner_model_name: str = "Davlan/xlm-roberta-base-ner-hrl"
    pos_model_name: str = "wietsedv/xlm-roberta-base-ft-udpos28-en"
    tagger_max_length: int = 512
    tagger_stride: int = 128
    tagger_batch_size: int = 8
    force: bool = False

    @property
    def ner_max_length(self):
        return self.tagger_max_length

    @property
    def ner_stride(self):
        return self.tagger_stride

    @property
    def ner_batch_size(self):
        return self.tagger_batch_size

    @property
    def pos_max_length(self):
        return self.tagger_max_length

    @property
    def pos_stride(self):
        return self.tagger_stride

    @property
    def pos_batch_size(self):
        return self.tagger_batch_size


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def _model_files(config: FinalEvaluationConfig) -> tuple[Path, ...]:
    return (
        config.representation_model_dir / "model.pt",
        config.gradient_boosting_model_dir / "model.joblib",
        config.representation_ling_model_dir / "model.pt",
        config.gradient_boosting_ling_model_dir / "model.joblib",
    )


def _fingerprint(config: FinalEvaluationConfig, layers: list[int]) -> str:
    payload = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "split": config.split,
        "max_length": config.max_length,
        "layers": layers,
        "ner_model": config.ner_model_name,
        "pos_model": config.pos_model_name,
        "tagger_max_length": config.tagger_max_length,
        "tagger_stride": config.tagger_stride,
        "models": [
            {"path": str(path), "size": path.stat().st_size, "mtime": path.stat().st_mtime_ns}
            for path in _model_files(config)
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


def _head_prediction(model, arrays, batch_size, torch, device):
    values = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), batch_size):
            tensors = [
                torch.from_numpy(value[start:start + batch_size].astype(np.float32)).to(device)
                if value is not None else None
                for value in arrays
            ]
            values.append(model(*tensors).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _predict_example(
    example,
    extractor,
    representation_model,
    representation_ling_model,
    gradient_boosting,
    gradient_boosting_ling,
    layers,
    representation_layer,
    ner_cache,
    pos_cache,
    config,
    fingerprint,
    device,
):
    torch = extractor.torch
    left, right = extractor.extract(example.text_a), extractor.extract(example.text_b)
    if left.words != example.text_a.split() or right.words != example.text_b.split():
        raise RuntimeError("Apertus word recovery differs from dataset whitespace tokenization")
    features_a = direction_features(left, right, layers, torch)
    features_b = direction_features(right, left, layers, torch)
    static_a, static_b = static_features(left, False), static_features(right, True)
    sides = np.asarray(["a"] * len(left.words) + ["b"] * len(right.words), dtype=str)
    features = np.concatenate((features_a, features_b)).astype(np.float32)
    static = np.concatenate((static_a, static_b)).astype(np.float32)
    engineered = vector_matrix(features, static, list(range(len(layers))), sides)
    gold = np.concatenate((example.labels_a, example.labels_b)).astype(np.float32)
    raw_ner = (
        ner_cache[text_cache_key(config.ner_model_name, example.text_a)]
        + ner_cache[text_cache_key(config.ner_model_name, example.text_b)]
    )
    raw_pos = (
        pos_cache[text_cache_key(config.pos_model_name, example.text_a)]
        + pos_cache[text_cache_key(config.pos_model_name, example.text_b)]
    )
    words = left.words + right.words
    ner, pos = normalize_linguistic_labels(
        raw_ner, raw_pos, observable_punctuation_mask(words, raw_pos)
    )
    linguistic = encode_linguistic_features(ner, pos)
    engineered_ling = np.concatenate((engineered, linguistic), axis=1).astype(np.float32)

    source_a, target_a = raw_word_pairs(left, right, representation_layer, torch)
    source_b, target_b = raw_word_pairs(right, left, representation_layer, torch)
    source = np.concatenate((source_a, source_b)).astype(np.float32)
    target = np.concatenate((target_a, target_b)).astype(np.float32)
    layer_index = layers.index(representation_layer)
    return {
        "fingerprint": np.asarray(fingerprint),
        "document_id": np.asarray(example.example_id),
        "language": np.asarray(example.language),
        "words": np.asarray(words, dtype=str),
        "sides": sides,
        "gold": gold,
        "ner_labels": ner,
        "pos_labels": pos,
        "diffalign_raw": features[:, layer_index, 0].astype(np.float32),
        "representation_mlp": _head_prediction(
            representation_model, (source, target, None), config.batch_size, torch, device
        ),
        "engineered_gradient_boosting": np.clip(
            gradient_boosting.predict(engineered), 0.0, 1.0
        ).astype(np.float32),
        "representation_ling_mlp": _head_prediction(
            representation_ling_model,
            (source, target, linguistic),
            config.batch_size,
            torch,
            device,
        ),
        "engineered_ling_gradient_boosting": np.clip(
            gradient_boosting_ling.predict(engineered_ling), 0.0, 1.0
        ).astype(np.float32),
    }


def _write_predictions(path: Path, documents: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "document_id", "language", "side", "token_index", "word", "gold",
                "ner", "pos", *SCORES,
            )
        )
        for item in documents:
            counters = {"a": 0, "b": 0}
            columns = (
                item["words"], item["sides"], item["gold"], item["ner_labels"],
                item["pos_labels"], *(item[score] for score in SCORES),
            )
            for values in zip(*columns, strict=True):
                word, side, gold, ner, pos, *scores = values
                token_index = counters[str(side)]
                counters[str(side)] += 1
                writer.writerow(
                    (
                        str(item["document_id"].item()), str(item["language"].item()),
                        str(side), token_index, str(word), float(gold), str(ner), str(pos),
                        *(float(value) for value in scores),
                    )
                )


def run_final_evaluation(config: FinalEvaluationConfig) -> dict:
    import joblib
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
    if representation_layer not in layers:
        raise ValueError("Representation layer must be among engineered feature layers")
    representation_checkpoint = torch.load(
        config.representation_model_dir / "model.pt", map_location="cpu", weights_only=False
    )
    representation_ling_checkpoint = torch.load(
        config.representation_ling_model_dir / "model.pt", map_location="cpu", weights_only=False
    )
    if int(representation_checkpoint["representation_layer"]) != representation_layer:
        raise ValueError("Representation checkpoint layer differs from scan selection")
    if int(representation_ling_checkpoint["representation_layer"]) != representation_layer:
        raise ValueError("Linguistic representation checkpoint layer differs from scan selection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation_model = build_pair_regressor(
        int(representation_checkpoint["hidden_size"]),
        int(representation_checkpoint["hidden_width"]),
        float(representation_checkpoint["dropout"]), torch,
        int(representation_checkpoint.get("linguistic_dimension", 0)),
    ).to(device)
    representation_model.load_state_dict(representation_checkpoint["state_dict"])
    representation_model.eval()
    representation_ling_model = build_pair_regressor(
        int(representation_ling_checkpoint["hidden_size"]),
        int(representation_ling_checkpoint["hidden_width"]),
        float(representation_ling_checkpoint["dropout"]), torch,
        int(representation_ling_checkpoint["linguistic_dimension"]),
    ).to(device)
    representation_ling_model.load_state_dict(representation_ling_checkpoint["state_dict"])
    representation_ling_model.eval()
    gradient_boosting = joblib.load(config.gradient_boosting_model_dir / "model.joblib")
    gradient_boosting_ling = joblib.load(
        config.gradient_boosting_ling_model_dir / "model.joblib"
    )

    examples, resolved = load_examples(config.dataset_name, config.dataset_revision, config.split)
    if config.max_documents is not None:
        examples = examples[: config.max_documents]
    linguistic_cache = config.output_dir / "linguistic_cache"
    ner_cache = prepare_ner_cache(examples, config, linguistic_cache / "ner")
    pos_cache = prepare_pos_cache(examples, config, linguistic_cache / "pos")
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
    for example, path in tqdm(pending, desc="Final five-system RSD evaluation"):
        try:
            payload = _predict_example(
                example, extractor, representation_model, representation_ling_model,
                gradient_boosting, gradient_boosting_ling, layers, representation_layer,
                ner_cache, pos_cache, config, fingerprint, device,
            )
            _save(path, payload)
            documents.append(payload)
        except Exception as exc:
            LOGGER.exception("Failed final evaluation document %s", example.example_id)
            errors.append({"document_id": example.example_id, "error": str(exc)})
    with (config.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    if not documents:
        raise RuntimeError("No final-evaluation documents succeeded")
    documents.sort(key=lambda item: str(item["document_id"].item()))
    gold = np.concatenate([item["gold"] for item in documents])
    languages = np.concatenate(
        [np.full(len(item["gold"]), str(item["language"].item())) for item in documents]
    )
    metrics = {
        score: score_predictions(
            gold, np.concatenate([item[score] for item in documents]), languages
        )
        for score in SCORES
    }
    _write_predictions(config.output_dir / "test_predictions.csv.gz", documents)
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
