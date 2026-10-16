from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .final_evaluation import FinalEvaluationConfig, run_final_evaluation
from .representation import RepresentationConfig, run_prepare
from .scan import ScanConfig, run_scan
from .training import (
    RepresentationTrainConfig,
    VectorTrainConfig,
    train_representation,
    train_vector,
)


def _common_dataset(parser):
    parser.add_argument("--dataset-name", default="ZurichNLP/SwissGov-RSD")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split", default="dev")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Final frozen-Apertus lightweight SwissGov-RSD experiment"
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="Scan all layers and select promising layers")
    _common_dataset(scan)
    scan.add_argument("--output-dir", type=Path, required=True)
    scan.add_argument("--model-name", default="swiss-ai/Apertus-v1.5-8B")
    scan.add_argument("--max-documents", type=int)
    scan.add_argument("--validation-per-language", type=int, default=17)
    scan.add_argument("--seed", type=int, default=42)
    scan.add_argument("--top-layers", type=int, default=5)
    scan.add_argument("--max-length", type=int, default=8192)
    scan.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    scan.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    scan.add_argument("--ner-model-name", default="Davlan/xlm-roberta-base-ner-hrl")
    scan.add_argument("--pos-model-name", default="wietsedv/xlm-roberta-base-ft-udpos28-en")
    scan.add_argument("--tagger-max-length", type=int, default=512)
    scan.add_argument("--tagger-stride", type=int, default=128)
    scan.add_argument("--tagger-batch-size", type=int, default=8)
    scan.add_argument("--force", action="store_true")

    prepare = commands.add_parser("prepare-representations", help="Cache raw paired hidden states")
    _common_dataset(prepare)
    prepare.add_argument("--scan-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model-name", default="swiss-ai/Apertus-v1.5-8B")
    prepare.add_argument("--max-length", type=int, default=8192)
    prepare.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    prepare.add_argument("--force", action="store_true")

    vector = commands.add_parser(
        "train-gradient-boosting", help="Train an engineered-feature histogram gradient booster"
    )
    vector.add_argument("--scan-dir", type=Path, required=True)
    vector.add_argument("--output-dir", type=Path, required=True)
    vector.add_argument("--positive-weight", type=float, default=4.0)
    vector.add_argument("--max-iterations", type=int, default=300)
    vector.add_argument("--learning-rate", type=float, default=0.05)
    vector.add_argument("--max-leaf-nodes", type=int, default=31)
    vector.add_argument("--l2-regularization", type=float, default=1.0)
    vector.add_argument("--seed", type=int, default=42)
    vector.add_argument("--linguistic", action="store_true")

    representation = commands.add_parser("train-representation", help="Train small paired-state MLP")
    representation.add_argument("--scan-dir", type=Path, required=True)
    representation.add_argument("--representation-dir", type=Path, required=True)
    representation.add_argument("--output-dir", type=Path, required=True)
    representation.add_argument("--hidden-width", type=int, default=128)
    representation.add_argument("--dropout", type=float, default=0.1)
    representation.add_argument("--epochs", type=int, default=20)
    representation.add_argument("--batch-size", type=int, default=1024)
    representation.add_argument("--learning-rate", type=float, default=3e-4)
    representation.add_argument("--weight-decay", type=float, default=1e-4)
    representation.add_argument("--positive-weight", type=float, default=4.0)
    representation.add_argument("--pairwise-weight", type=float, default=0.2)
    representation.add_argument("--pairwise-temperature", type=float, default=0.1)
    representation.add_argument("--patience", type=int, default=5)
    representation.add_argument("--seed", type=int, default=42)
    representation.add_argument("--linguistic", action="store_true")

    comparison = commands.add_parser(
        "evaluate-final",
        help="Compare raw DiffAlign and four lightweight models on a held-out split",
    )
    _common_dataset(comparison)
    comparison.set_defaults(split="test")
    comparison.add_argument("--scan-dir", type=Path, required=True)
    comparison.add_argument("--representation-model-dir", type=Path, required=True)
    comparison.add_argument("--gradient-boosting-model-dir", type=Path, required=True)
    comparison.add_argument("--representation-ling-model-dir", type=Path, required=True)
    comparison.add_argument("--gradient-boosting-ling-model-dir", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.add_argument("--model-name", default="swiss-ai/Apertus-v1.5-8B")
    comparison.add_argument("--max-documents", type=int)
    comparison.add_argument("--max-length", type=int, default=8192)
    comparison.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    comparison.add_argument("--batch-size", type=int, default=1024)
    comparison.add_argument("--ner-model-name", default="Davlan/xlm-roberta-base-ner-hrl")
    comparison.add_argument("--pos-model-name", default="wietsedv/xlm-roberta-base-ft-udpos28-en")
    comparison.add_argument("--tagger-max-length", type=int, default=512)
    comparison.add_argument("--tagger-stride", type=int, default=128)
    comparison.add_argument("--tagger-batch-size", type=int, default=8)
    comparison.add_argument("--force", action="store_true")
    return parser


def main():
    arguments = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if arguments.command == "scan":
        summary = run_scan(ScanConfig(
            output_dir=arguments.output_dir, model_name=arguments.model_name,
            dataset_name=arguments.dataset_name, dataset_revision=arguments.dataset_revision,
            split=arguments.split, max_documents=arguments.max_documents,
            validation_per_language=arguments.validation_per_language, seed=arguments.seed,
            top_layers=arguments.top_layers, max_length=arguments.max_length,
            dtype=arguments.dtype, storage_dtype=arguments.storage_dtype,
            ner_model_name=arguments.ner_model_name, pos_model_name=arguments.pos_model_name,
            tagger_max_length=arguments.tagger_max_length,
            tagger_stride=arguments.tagger_stride,
            tagger_batch_size=arguments.tagger_batch_size, force=arguments.force,
        ))
    elif arguments.command == "prepare-representations":
        summary = run_prepare(RepresentationConfig(
            scan_dir=arguments.scan_dir, output_dir=arguments.output_dir,
            model_name=arguments.model_name, dataset_name=arguments.dataset_name,
            dataset_revision=arguments.dataset_revision, split=arguments.split,
            max_length=arguments.max_length, dtype=arguments.dtype, force=arguments.force,
        ))
    elif arguments.command == "train-gradient-boosting":
        summary = train_vector(VectorTrainConfig(
            scan_dir=arguments.scan_dir, output_dir=arguments.output_dir,
            positive_weight=arguments.positive_weight,
            max_iterations=arguments.max_iterations, learning_rate=arguments.learning_rate,
            max_leaf_nodes=arguments.max_leaf_nodes,
            l2_regularization=arguments.l2_regularization, seed=arguments.seed,
            linguistic=arguments.linguistic,
        ))
    elif arguments.command == "train-representation":
        summary = train_representation(RepresentationTrainConfig(
            scan_dir=arguments.scan_dir, representation_dir=arguments.representation_dir,
            output_dir=arguments.output_dir, hidden_width=arguments.hidden_width,
            dropout=arguments.dropout, epochs=arguments.epochs,
            batch_size=arguments.batch_size, learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay, positive_weight=arguments.positive_weight,
            pairwise_weight=arguments.pairwise_weight,
            pairwise_temperature=arguments.pairwise_temperature,
            patience=arguments.patience, seed=arguments.seed,
            linguistic=arguments.linguistic,
        ))
    elif arguments.command == "evaluate-final":
        summary = run_final_evaluation(FinalEvaluationConfig(
            scan_dir=arguments.scan_dir,
            representation_model_dir=arguments.representation_model_dir,
            gradient_boosting_model_dir=arguments.gradient_boosting_model_dir,
            representation_ling_model_dir=arguments.representation_ling_model_dir,
            gradient_boosting_ling_model_dir=arguments.gradient_boosting_ling_model_dir,
            output_dir=arguments.output_dir, model_name=arguments.model_name,
            dataset_name=arguments.dataset_name, dataset_revision=arguments.dataset_revision,
            split=arguments.split, max_documents=arguments.max_documents,
            max_length=arguments.max_length, dtype=arguments.dtype,
            batch_size=arguments.batch_size,
            ner_model_name=arguments.ner_model_name, pos_model_name=arguments.pos_model_name,
            tagger_max_length=arguments.tagger_max_length,
            tagger_stride=arguments.tagger_stride,
            tagger_batch_size=arguments.tagger_batch_size, force=arguments.force,
        ))
    else:
        raise AssertionError(f"Unhandled command: {arguments.command}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
