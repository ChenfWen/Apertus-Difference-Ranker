from __future__ import annotations

import json
import math
import sys
from pathlib import Path


EXPECTED_METHODS = (
    "diffalign_raw",
    "representation_mlp",
    "engineered_gradient_boosting",
    "representation_ling_mlp",
    "engineered_ling_gradient_boosting",
)
LANGUAGES = ("de", "fr", "it")
EXPECTED_DOCUMENTS = 168
EXPECTED_TOKENS = 132_080


def validate(payload: dict) -> list[dict]:
    if payload.get("test_data_used_for_training_or_selection") is not False:
        raise ValueError("Result does not assert test isolation")
    if payload.get("requested_split") != "test" or payload.get("resolved_split") != "test":
        raise ValueError("Frozen artifact is not a test-split result")
    if payload.get("input_documents") != EXPECTED_DOCUMENTS:
        raise ValueError("Unexpected test-document count")
    if payload.get("successful_documents") != EXPECTED_DOCUMENTS:
        raise ValueError("Not every test document completed")
    if payload.get("failed_documents") != 0:
        raise ValueError("Frozen evaluation contains failed documents")
    if tuple(payload.get("controlled_scores", ())) != EXPECTED_METHODS:
        raise ValueError("Unexpected method set")

    rows = []
    for method in EXPECTED_METHODS:
        metrics = payload["metrics"][method]
        macro = sum(float(metrics[f"spearman_{language}"]) for language in LANGUAGES) / 3.0
        if not math.isclose(macro, float(metrics["spearman_macro"]), abs_tol=1e-12):
            raise ValueError(f"Macro mismatch for {method}")
        if int(metrics["n_tokens_pooled"]) != EXPECTED_TOKENS:
            raise ValueError(f"Unexpected token count for {method}")
        rows.append({"method": method, **metrics})
    return rows


def replay(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = validate(payload)
    print("Apertus Difference Ranker — frozen held-out test result")
    print(f"Model: {payload['model_name']}")
    print(f"Documents: {payload['successful_documents']}/{payload['input_documents']}")
    print(f"Scored tokens: {rows[0]['n_tokens_pooled']:,}")
    print()
    print(f"{'method':40s} {'DE':>8s} {'FR':>8s} {'IT':>8s} {'macro':>8s} {'pooled':>8s}")
    for row in rows:
        print(
            f"{row['method']:40s} {row['spearman_de']:8.4f} "
            f"{row['spearman_fr']:8.4f} {row['spearman_it']:8.4f} "
            f"{row['spearman_macro']:8.4f} {row['spearman_pooled']:8.4f}"
        )
    winner = max(rows, key=lambda row: row["spearman_macro"])
    improvement = winner["spearman_macro"] - payload["metrics"]["diffalign_raw"]["spearman_macro"]
    print(f"\nBest macro Spearman: {winner['method']} = {winner['spearman_macro']:.4f}")
    print(f"Absolute improvement over raw DiffAlign: {improvement:.4f}")
    print("Result integrity checks passed.")


def main() -> None:
    paths = [Path(value) for value in sys.argv[1:]] or [
        Path("data/results/final_test_8b_summary.json"),
        Path("data/results/final_test_70b_summary.json"),
    ]
    for index, path in enumerate(paths):
        if index:
            print("\n" + "=" * 88 + "\n")
        replay(path)


if __name__ == "__main__":
    main()

