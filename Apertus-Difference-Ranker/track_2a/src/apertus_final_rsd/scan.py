from __future__ import annotations

import csv
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
from .features import BASE_FEATURES, STATIC_FEATURES, direction_features, static_features
from .model import ApertusExtractor
from .linguistic import normalize_linguistic_labels, observable_punctuation_mask
from .ner import prepare_ner_cache, text_cache_key
from .pos import prepare_pos_cache
from .split import document_split, stratified_subset

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ScanConfig:
    output_dir: Path
    model_name: str = "swiss-ai/Apertus-v1.5-8B"
    dataset_name: str = "ZurichNLP/SwissGov-RSD"
    dataset_revision: str | None = None
    split: str = "dev"
    max_documents: int | None = None
    validation_per_language: int = 17
    seed: int = 42
    top_layers: int = 5
    max_length: int = 4096
    dtype: str = "auto"
    storage_dtype: str = "float32"
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


def _fingerprint(config: ScanConfig, resolved_split: str) -> str:
    payload = asdict(config)
    payload["output_dir"] = "excluded"
    payload["force"] = False
    payload["resolved_split"] = resolved_split
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _save(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _load(path: Path, fingerprint: str):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["fingerprint"].item()) != fingerprint:
            return None
        return {key: archive[key].copy() for key in archive.files}


def scan_example(example, extractor, fingerprint: str, ner_cache, pos_cache, config):
    left, right = extractor.extract(example.text_a), extractor.extract(example.text_b)
    features_a = direction_features(left, right, extractor.layers, extractor.torch)
    features_b = direction_features(right, left, extractor.layers, extractor.torch)
    static_a, static_b = static_features(left, False), static_features(right, True)
    gold = np.concatenate((example.labels_a, example.labels_b)).astype(np.float32)
    ner_a = ner_cache[text_cache_key(config.ner_model_name, example.text_a)]
    ner_b = ner_cache[text_cache_key(config.ner_model_name, example.text_b)]
    pos_a = pos_cache[text_cache_key(config.pos_model_name, example.text_a)]
    pos_b = pos_cache[text_cache_key(config.pos_model_name, example.text_b)]
    if left.words != example.text_a.split() or right.words != example.text_b.split():
        raise RuntimeError("Apertus word recovery differs from dataset whitespace tokenization")
    words = left.words + right.words
    raw_pos = pos_a + pos_b
    ner, pos = normalize_linguistic_labels(
        ner_a + ner_b, raw_pos, observable_punctuation_mask(words, raw_pos)
    )
    return {
        "fingerprint": np.asarray(fingerprint),
        "document_id": np.asarray(example.example_id),
        "language": np.asarray(example.language),
        "words": np.asarray(words, dtype=str),
        "sides": np.asarray(["a"] * len(left.words) + ["b"] * len(right.words), dtype=str),
        "gold": gold,
        "ner_labels": ner,
        "pos_labels": pos,
        "features": np.concatenate((features_a, features_b)).astype(np.float32),
        "static": np.concatenate((static_a, static_b)).astype(np.float32),
        "layers": np.asarray(extractor.layers, np.int16),
    }


def _select_layers(rows: list[dict], count: int) -> list[int]:
    ranked = sorted(
        (row for row in rows if np.isfinite(row["spearman_macro"])),
        key=lambda row: row["spearman_macro"], reverse=True,
    )
    return [int(row["layer"]) for row in ranked[:count]]


def _plot_layers(rows: list[dict], path: Path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 6))
    series = (
        ("EN-DE", "spearman_de", "-", 1.8),
        ("EN-FR", "spearman_fr", "--", 1.8),
        ("EN-IT", "spearman_it", ":", 2.1),
        ("ALL", "spearman_macro", "-.", 2.4),
    )
    for label, field, line_style, line_width in series:
        axis.plot(
            [row["layer"] for row in rows],
            [row[field] for row in rows],
            color="#1F4E79",
            linestyle=line_style,
            linewidth=line_width,
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Apertus Transformer layer")
    axis.set_ylabel("Spearman correlation")
    axis.set_title("Whole-document DiffAlign layer sensitivity on fitting data")
    axis.grid(alpha=0.25)
    axis.legend(title="Language pair")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _score_layers(checkpoints: list[dict], layers: tuple[int, ...]) -> list[dict]:
    if not checkpoints:
        return []
    gold = np.concatenate([item["gold"] for item in checkpoints])
    languages = np.concatenate(
        [np.full(len(item["gold"]), str(item["language"].item())) for item in checkpoints]
    )
    return [
        {
            "layer": layer,
            **score_predictions(
                gold,
                np.concatenate([item["features"][:, layer_index, 0] for item in checkpoints]),
                languages,
            ),
        }
        for layer_index, layer in enumerate(layers)
    ]


def _write_metric_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_scan(config: ScanConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if config.force:
        for path in checkpoint_dir.glob("*.npz"):
            path.unlink()
    examples, resolved = load_examples(
        config.dataset_name, config.dataset_revision, config.split
    )
    examples = stratified_subset(examples, config.max_documents, config.seed)
    train_ids, validation_ids = document_split(
        examples, config.validation_per_language, config.seed
    )
    fingerprint = _fingerprint(config, resolved)
    pending, checkpoints = [], []
    for index, example in enumerate(examples):
        path = checkpoint_dir / f"{index:06d}-{_safe(example.example_id)}.npz"
        existing = _load(path, fingerprint)
        if existing is None:
            pending.append((example, path))
        else:
            checkpoints.append(existing)
    linguistic_cache = config.output_dir / "linguistic_cache"
    ner_cache = prepare_ner_cache(examples, config, linguistic_cache / "ner")
    pos_cache = prepare_pos_cache(examples, config, linguistic_cache / "pos")
    extractor = ApertusExtractor(
        config.model_name, None, config.max_length, config.dtype, config.storage_dtype
    ) if pending else None
    errors = []
    for example, path in tqdm(pending, desc="Apertus layer scan"):
        try:
            payload = scan_example(example, extractor, fingerprint, ner_cache, pos_cache, config)
            _save(path, payload)
            checkpoints.append(payload)
        except Exception as exc:
            LOGGER.exception("Failed %s", example.example_id)
            errors.append({"document_id": example.example_id, "error": str(exc)})
    with (config.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    if not checkpoints:
        raise RuntimeError("No successful layer-scan documents")
    layers = tuple(int(layer) for layer in checkpoints[0]["layers"])
    train_checkpoints = [
        item for item in checkpoints if str(item["document_id"].item()) in train_ids
    ]
    validation_checkpoints = [
        item for item in checkpoints if str(item["document_id"].item()) in validation_ids
    ]
    score_groups = {
        "train": _score_layers(train_checkpoints, layers),
        "validation": _score_layers(validation_checkpoints, layers),
        "all_dev": _score_layers(checkpoints, layers),
    }
    for name, rows in score_groups.items():
        _write_metric_rows(config.output_dir / f"layer_metrics_{name}.csv", rows)
    combined_rows = []
    for name, rows in score_groups.items():
        combined_rows.extend({"split": name, **row} for row in rows)
    _write_metric_rows(config.output_dir / "layer_spearman_scores.csv", combined_rows)
    _plot_layers(score_groups["train"], config.output_dir / "layer_spearman.png")
    selected = _select_layers(score_groups["train"], config.top_layers)
    selection = {
        "selection_source": "training_documents_only",
        "representation_layer": selected[0],
        "vector_layers": selected,
        "ranked_layers": [
            {"layer": int(row["layer"]), "spearman_macro": float(row["spearman_macro"])}
            for row in sorted(score_groups["train"], key=lambda row: row["spearman_macro"], reverse=True)
        ],
    }
    (config.output_dir / "promising_layers.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "seed": config.seed,
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
    }
    (config.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    schema = {"base_features": list(BASE_FEATURES), "static_features": list(STATIC_FEATURES)}
    (config.output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "requested_split": config.split,
        "resolved_split": resolved,
        "documents": len(examples),
        "training_documents": len(train_ids),
        "validation_documents": len(validation_ids),
        "processed_this_run": len(pending),
        "successful_documents": len(checkpoints),
        "failed_documents": len(errors),
        "model_name": config.model_name,
        "model_layers": len(layers),
        "representation_layer": selected[0],
        "vector_layers": selected,
        "best_train_macro_spearman": selection["ranked_layers"][0]["spearman_macro"],
        "layer_score_files": {
            "combined": "layer_spearman_scores.csv",
            "train": "layer_metrics_train.csv",
            "validation": "layer_metrics_validation.csv",
            "all_dev": "layer_metrics_all_dev.csv",
            "plot": "layer_spearman.png",
        },
        "layer_selection_uses": "train only; validation and all_dev are diagnostics",
        "model_forward_passes_this_run": 0 if extractor is None else extractor.forward_passes,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return summary
