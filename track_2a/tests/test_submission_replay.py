import json
from pathlib import Path

from submission_replay import EXPECTED_METHODS, validate


def test_frozen_result_integrity():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "data/results/final_test_8b_summary.json").read_text(encoding="utf-8")
    )
    rows = validate(payload)
    assert tuple(row["method"] for row in rows) == EXPECTED_METHODS
    assert max(rows, key=lambda row: row["spearman_macro"])["method"] == (
        "engineered_ling_gradient_boosting"
    )
    assert payload["metrics"]["engineered_ling_gradient_boosting"]["spearman_macro"] > (
        payload["metrics"]["diffalign_raw"]["spearman_macro"]
    )


def test_frozen_70b_result_integrity():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "data/results/final_test_70b_summary.json").read_text(encoding="utf-8")
    )
    rows = validate(payload)
    assert tuple(row["method"] for row in rows) == EXPECTED_METHODS
    assert max(rows, key=lambda row: row["spearman_macro"])["method"] == (
        "representation_ling_mlp"
    )
    assert payload["metrics"]["diffalign_raw"]["spearman_macro"] > 0.27

