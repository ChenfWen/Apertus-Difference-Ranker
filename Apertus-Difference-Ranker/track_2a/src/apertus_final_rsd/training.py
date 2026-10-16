from __future__ import annotations

import copy
import csv
import gzip
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .evaluation import score_predictions
from .features import BASE_FEATURES, STATIC_FEATURES, vector_matrix
from .linguistic import (
    LINGUISTIC_DIMENSION,
    encode_linguistic_features,
    linguistic_feature_names,
)

LOGGER = logging.getLogger(__name__)


def build_pair_regressor(
    dimension: int,
    hidden_width: int,
    dropout: float,
    torch,
    linguistic_dimension: int = 0,
):
    """Construct the reloadable lightweight head used by representation mode."""
    class PairRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            input_size = 4 * dimension + linguistic_dimension
            self.network = torch.nn.Sequential(
                torch.nn.LayerNorm(input_size),
                torch.nn.Linear(input_size, hidden_width),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_width, 1),
            )

        def forward(self, source, target, linguistic=None):
            source = torch.nn.functional.normalize(source, p=2, dim=1, eps=1e-12)
            target = torch.nn.functional.normalize(target, p=2, dim=1, eps=1e-12)
            parts = [source, target, (source - target).abs(), source * target]
            if linguistic_dimension:
                if linguistic is None or linguistic.shape[1] != linguistic_dimension:
                    raise ValueError("Linguistic feature tensor is missing or has the wrong width")
                parts.append(linguistic)
            features = torch.cat(parts, dim=1)
            return torch.sigmoid(self.network(features).squeeze(1))

    return PairRegressor()


def build_feature_regressor(dimension: int, hidden_width: int, dropout: float, torch):
    """Construct an MLP matched to the raw-representation head."""
    class FeatureRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.LayerNorm(dimension),
                torch.nn.Linear(dimension, hidden_width),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_width, 1),
            )

        def forward(self, features):
            return torch.sigmoid(self.network(features).squeeze(1))

    return FeatureRegressor()


@dataclass(slots=True)
class VectorTrainConfig:
    scan_dir: Path
    output_dir: Path
    positive_weight: float = 4.0
    max_iterations: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    seed: int = 42
    linguistic: bool = False


@dataclass(slots=True)
class RepresentationTrainConfig:
    scan_dir: Path
    representation_dir: Path
    output_dir: Path
    hidden_width: int = 128
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    positive_weight: float = 4.0
    pairwise_weight: float = 0.2
    pairwise_temperature: float = 0.1
    patience: int = 5
    seed: int = 42
    linguistic: bool = False


@dataclass(slots=True)
class FeatureMLPTrainConfig:
    scan_dir: Path
    output_dir: Path
    hidden_width: int = 128
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 4096
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    positive_weight: float = 4.0
    pairwise_weight: float = 0.2
    pairwise_temperature: float = 0.1
    patience: int = 5
    seed: int = 42


def _manifest(scan_dir: Path):
    value = json.loads((scan_dir / "split_manifest.json").read_text(encoding="utf-8"))
    return set(value["train_ids"]), set(value["validation_ids"])


def _scan_documents(scan_dir: Path):
    for path in sorted((scan_dir / "checkpoints").glob("*.npz")):
        with np.load(path, allow_pickle=False) as item:
            yield {key: item[key].copy() for key in item.files}


def _baseline_validation(scan_dir: Path, validation_ids: set[str], layer: int):
    gold, prediction, languages = [], [], []
    for item in _scan_documents(scan_dir):
        document_id = str(item["document_id"].item())
        if document_id not in validation_ids:
            continue
        layers = [int(value) for value in item["layers"]]
        index = layers.index(layer)
        gold.append(item["gold"])
        prediction.append(item["features"][:, index, 0])
        languages.append(np.full(len(item["gold"]), str(item["language"].item())))
    return score_predictions(
        np.concatenate(gold), np.concatenate(prediction), np.concatenate(languages)
    )


def _load_vector_data(scan_dir: Path, selected_layers: list[int], linguistic: bool = False):
    train_ids, validation_ids = _manifest(scan_dir)
    parts = {"train": ([], [], [], [], []), "validation": ([], [], [], [], [])}
    feature_names = None
    for item in _scan_documents(scan_dir):
        document_id = str(item["document_id"].item())
        split = "train" if document_id in train_ids else "validation"
        if split == "validation" and document_id not in validation_ids:
            continue
        layers = [int(value) for value in item["layers"]]
        selected_indices = [layers.index(layer) for layer in selected_layers]
        matrix = vector_matrix(item["features"], item["static"], selected_indices, item["sides"])
        if linguistic:
            matrix = np.concatenate(
                (
                    matrix,
                    encode_linguistic_features(item["ner_labels"], item["pos_labels"]),
                ),
                axis=1,
            ).astype(np.float32)
        valid = item["gold"] != -1
        x, y, language, documents, sides = parts[split]
        x.append(matrix[valid])
        y.append(item["gold"][valid])
        language.append(np.full(int(valid.sum()), str(item["language"].item())))
        documents.append(np.full(int(valid.sum()), document_id))
        sides.append(item["sides"][valid])
        if feature_names is None:
            feature_names = []
            for layer in selected_layers:
                feature_names.extend(f"layer_{layer}_{name}" for name in BASE_FEATURES)
            feature_names.extend(
                f"layer_{layer}_previous_top1_distance" for layer in selected_layers
            )
            feature_names.extend(
                f"layer_{layer}_next_top1_distance" for layer in selected_layers
            )
            feature_names.extend(STATIC_FEATURES)
            if linguistic:
                feature_names.extend(linguistic_feature_names())
    result = {}
    for split, values in parts.items():
        result[split] = tuple(np.concatenate(part) for part in values)
    return result, feature_names


def _write_validation_predictions(path: Path, gold, prediction, languages, documents, sides):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("document_id", "language", "side", "gold", "prediction"))
        writer.writerows(
            zip(documents, languages, sides, gold.astype(float), prediction.astype(float), strict=True)
        )


def train_vector(config: VectorTrainConfig) -> dict:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    config.output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((config.scan_dir / "promising_layers.json").read_text(encoding="utf-8"))
    selected_layers = [int(layer) for layer in selection["vector_layers"]]
    data, feature_names = _load_vector_data(
        config.scan_dir, selected_layers, linguistic=config.linguistic
    )
    x_train, y_train, language_train, _, _ = data["train"]
    x_validation, y_validation, language_validation, documents_validation, sides_validation = data["validation"]
    weights = 1.0 + config.positive_weight * (y_train > 0).astype(np.float32)
    candidates = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=config.learning_rate,
            max_iter=config.max_iterations, max_leaf_nodes=config.max_leaf_nodes,
            l2_regularization=config.l2_regularization, random_state=config.seed,
            early_stopping=False,
        ),
    }
    results, predictions = {}, {}
    for name, estimator in candidates.items():
        LOGGER.info("Training engineered-vector model: %s", name)
        estimator.fit(x_train, y_train, sample_weight=weights)
        prediction = np.clip(estimator.predict(x_validation), 0.0, 1.0)
        predictions[name] = prediction
        results[name] = score_predictions(y_validation, prediction, language_validation)
    finite = [name for name in results if math.isfinite(results[name]["spearman_macro"])]
    if not finite:
        raise RuntimeError("No vector model produced a finite validation macro Spearman score")
    best_name = max(finite, key=lambda name: results[name]["spearman_macro"])
    best_model = candidates[best_name]
    joblib.dump(best_model, config.output_dir / "model.joblib")
    _write_validation_predictions(
        config.output_dir / "validation_predictions.csv.gz",
        y_validation, predictions[best_name], language_validation,
        documents_validation, sides_validation,
    )
    baseline = _baseline_validation(
        config.scan_dir, _manifest(config.scan_dir)[1], int(selection["representation_layer"])
    )
    summary = {
        "mode": (
            "engineered_linguistic_gradient_boosting"
            if config.linguistic else "engineered_gradient_boosting"
        ),
        "selected_layers": selected_layers,
        "feature_count": int(x_train.shape[1]),
        "feature_names": feature_names,
        "uses_pos_and_ner": config.linguistic,
        "training_tokens": int(len(y_train)),
        "validation_tokens": int(len(y_validation)),
        "positive_training_tokens": int((y_train > 0).sum()),
        "candidate_validation_metrics": results,
        "selected_model": best_name,
        "validation_metrics": results[best_name],
        "diffalign_validation_baseline": baseline,
        "macro_spearman_change_vs_diffalign": (
            results[best_name]["spearman_macro"] - baseline["spearman_macro"]
        ),
        "no_document_leakage": True,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def train_feature_mlp(config: FeatureMLPTrainConfig) -> dict:
    import torch

    config.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    selection = json.loads((config.scan_dir / "promising_layers.json").read_text(encoding="utf-8"))
    selected_layers = [int(layer) for layer in selection["vector_layers"]]
    data, feature_names = _load_vector_data(config.scan_dir, selected_layers)
    x_train, y_train, language_train, _, _ = data["train"]
    x_validation, y_validation, language_validation, documents_validation, sides_validation = data["validation"]

    feature_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    x_train = ((x_train - feature_mean) / feature_scale).astype(np.float32)
    x_validation = ((x_validation - feature_mean) / feature_scale).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_regressor(
        x_train.shape[1], config.hidden_width, config.dropout, torch
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def validate():
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(x_validation), config.batch_size):
                batch = torch.from_numpy(x_validation[start:start + config.batch_size]).to(device)
                values.append(model(batch).cpu().numpy())
        prediction = np.concatenate(values)
        return score_predictions(y_validation, prediction, language_validation), prediction

    history, best_state, best_metrics, best_prediction, best_epoch = [], None, None, None, None
    best_rho, stale = -math.inf, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = np.random.permutation(len(y_train))
        losses = []
        for start in range(0, len(order), config.batch_size):
            chosen = order[start:start + config.batch_size]
            x = torch.from_numpy(x_train[chosen]).to(device)
            y = torch.from_numpy(y_train[chosen].astype(np.float32)).to(device)
            prediction = model(x)
            weights = 1.0 + config.positive_weight * (y > 0).to(torch.float32)
            pointwise = (
                torch.nn.functional.smooth_l1_loss(prediction, y, reduction="none") * weights
            ).mean()
            permutation = torch.randperm(len(y), device=device)
            delta_gold = y - y[permutation]
            comparable = delta_gold.abs() > 1e-6
            if comparable.any():
                sign = delta_gold[comparable].sign()
                delta_prediction = prediction[comparable] - prediction[permutation][comparable]
                pairwise = torch.nn.functional.softplus(
                    -sign * delta_prediction / config.pairwise_temperature
                ).mean()
            else:
                pairwise = pointwise.new_tensor(0.0)
            loss = pointwise + config.pairwise_weight * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, validation_prediction = validate()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        LOGGER.info(
            "Feature MLP epoch %d: loss %.5f, validation macro rho %.5f",
            epoch, np.mean(losses), metrics["spearman_macro"],
        )
        rho = metrics["spearman_macro"]
        if math.isfinite(rho) and rho > best_rho:
            best_rho, stale = rho, 0
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            best_metrics, best_prediction, best_epoch = metrics, validation_prediction, epoch
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Engineered-feature MLP produced no finite validation score")

    torch.save(
        {
            "state_dict": best_state,
            "feature_count": int(x_train.shape[1]),
            "feature_names": feature_names,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "hidden_width": config.hidden_width,
            "dropout": config.dropout,
            "selected_layers": selected_layers,
        },
        config.output_dir / "model.pt",
    )
    _write_validation_predictions(
        config.output_dir / "validation_predictions.csv.gz",
        y_validation, best_prediction, language_validation,
        documents_validation, sides_validation,
    )
    baseline = _baseline_validation(
        config.scan_dir, _manifest(config.scan_dir)[1], int(selection["representation_layer"])
    )
    summary = {
        "mode": "engineered_feature_mlp",
        "selected_layers": selected_layers,
        "feature_count": int(x_train.shape[1]),
        "architecture": f"LayerNorm({x_train.shape[1]}) -> Linear({x_train.shape[1]}, {config.hidden_width}) -> GELU -> Dropout -> Linear({config.hidden_width}, 1) -> Sigmoid",
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_tokens": int(len(y_train)),
        "validation_tokens": int(len(y_validation)),
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "validation_metrics": best_metrics,
        "diffalign_validation_baseline": baseline,
        "macro_spearman_change_vs_diffalign": (
            best_metrics["spearman_macro"] - baseline["spearman_macro"]
        ),
        "training_history": history,
        "comparison_control": "same hidden width, activation, dropout, output, weighting, and losses as paired-state MLP",
        "no_document_leakage": True,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _representation_files(directory: Path):
    return sorted((directory / "documents").glob("*.npz"))


def _document_identifier(path: Path) -> str:
    with np.load(path, allow_pickle=False) as item:
        return str(item["document_id"].item())


def train_representation(config: RepresentationTrainConfig) -> dict:
    import torch

    config.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_ids, validation_ids = _manifest(config.scan_dir)
    files = _representation_files(config.representation_dir)
    train_files = [path for path in files if _document_identifier(path) in train_ids]
    validation_files = [path for path in files if _document_identifier(path) in validation_ids]
    if not train_files or not validation_files:
        raise RuntimeError("Representation data does not cover both train and validation documents")
    with np.load(train_files[0], allow_pickle=False) as item:
        dimension = int(item["source"].shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linguistic_dimension = LINGUISTIC_DIMENSION if config.linguistic else 0
    model = build_pair_regressor(
        dimension, config.hidden_width, config.dropout, torch, linguistic_dimension
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def load_valid(path):
        with np.load(path, allow_pickle=False) as item:
            valid = item["gold"] != -1
            linguistic = encode_linguistic_features(item["ner_labels"], item["pos_labels"])
            return (
                item["source"][valid].astype(np.float32),
                item["target"][valid].astype(np.float32),
                linguistic[valid].astype(np.float32),
                item["gold"][valid].astype(np.float32),
                np.full(int(valid.sum()), str(item["language"].item())),
                np.full(int(valid.sum()), str(item["document_id"].item())),
                item["sides"][valid],
            )

    def validate():
        model.eval()
        gold, predicted, languages, documents, sides = [], [], [], [], []
        with torch.inference_mode():
            for path in validation_files:
                source, target, linguistic, labels, language, document, side = load_valid(path)
                values = []
                for start in range(0, len(labels), config.batch_size):
                    end = start + config.batch_size
                    values.append(
                        model(
                            torch.from_numpy(source[start:end]).to(device),
                            torch.from_numpy(target[start:end]).to(device),
                            torch.from_numpy(linguistic[start:end]).to(device)
                            if config.linguistic else None,
                        ).cpu().numpy()
                    )
                gold.append(labels); predicted.append(np.concatenate(values))
                languages.append(language); documents.append(document); sides.append(side)
        arrays = tuple(np.concatenate(values) for values in (gold, predicted, languages, documents, sides))
        return score_predictions(arrays[0], arrays[1], arrays[2]), arrays

    history, best_state, best_metrics, best_arrays, best_epoch = [], None, None, None, None
    best_rho, stale = -math.inf, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        random.shuffle(train_files)
        losses = []
        for path in train_files:
            source, target, linguistic, labels, *_ = load_valid(path)
            order = np.random.permutation(len(labels))
            for start in range(0, len(order), config.batch_size):
                chosen = order[start:start + config.batch_size]
                x_source = torch.from_numpy(source[chosen]).to(device)
                x_target = torch.from_numpy(target[chosen]).to(device)
                y = torch.from_numpy(labels[chosen]).to(device)
                x_linguistic = (
                    torch.from_numpy(linguistic[chosen]).to(device)
                    if config.linguistic else None
                )
                prediction = model(x_source, x_target, x_linguistic)
                weights = 1.0 + config.positive_weight * (y > 0).to(torch.float32)
                pointwise = (
                    torch.nn.functional.smooth_l1_loss(prediction, y, reduction="none") * weights
                ).mean()
                permutation = torch.randperm(len(y), device=device)
                delta_gold = y - y[permutation]
                comparable = delta_gold.abs() > 1e-6
                if comparable.any():
                    sign = delta_gold[comparable].sign()
                    delta_prediction = prediction[comparable] - prediction[permutation][comparable]
                    pairwise = torch.nn.functional.softplus(
                        -sign * delta_prediction / config.pairwise_temperature
                    ).mean()
                else:
                    pairwise = pointwise.new_tensor(0.0)
                loss = pointwise + config.pairwise_weight * pairwise
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        metrics, arrays = validate()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        LOGGER.info("Epoch %d: loss %.5f, validation macro rho %.5f", epoch, np.mean(losses), metrics["spearman_macro"])
        rho = metrics["spearman_macro"]
        if math.isfinite(rho) and rho > best_rho:
            best_rho, stale = rho, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_metrics, best_arrays, best_epoch = metrics, arrays, epoch
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Representation training produced no finite validation score")
    selection = json.loads((config.scan_dir / "promising_layers.json").read_text(encoding="utf-8"))
    torch.save(
        {
            "state_dict": best_state, "hidden_size": dimension,
            "hidden_width": config.hidden_width, "dropout": config.dropout,
            "representation_layer": int(selection["representation_layer"]),
            "linguistic_dimension": linguistic_dimension,
        },
        config.output_dir / "model.pt",
    )
    _write_validation_predictions(
        config.output_dir / "validation_predictions.csv.gz",
        best_arrays[0], best_arrays[1], best_arrays[2], best_arrays[3], best_arrays[4],
    )
    baseline = _baseline_validation(
        config.scan_dir, validation_ids, int(selection["representation_layer"])
    )
    summary = {
        "mode": (
            "paired_apertus_representation_with_pos_ner"
            if config.linguistic else "paired_apertus_representation"
        ),
        "representation_layer": int(selection["representation_layer"]),
        "input_definition": "[source, nearest_target, abs_difference, elementwise_product]",
        "uses_pos_and_ner": config.linguistic,
        "linguistic_dimension": linguistic_dimension,
        "apertus_hidden_size": dimension,
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_documents": len(train_files),
        "validation_documents": len(validation_files),
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "validation_metrics": best_metrics,
        "diffalign_validation_baseline": baseline,
        "macro_spearman_change_vs_diffalign": (
            best_metrics["spearman_macro"] - baseline["spearman_macro"]
        ),
        "training_history": history,
        "no_document_leakage": True,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return summary
