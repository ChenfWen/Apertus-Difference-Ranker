from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

LANGUAGES = ("de", "fr", "it")


def safe_spearman(gold, prediction) -> float:
    gold = np.asarray(gold, np.float64)
    prediction = np.asarray(prediction, np.float64)
    valid = np.isfinite(gold) & np.isfinite(prediction) & (gold != -1)
    gold, prediction = gold[valid], prediction[valid]
    if len(gold) < 2 or np.ptp(gold) == 0 or np.ptp(prediction) == 0:
        return float("nan")
    return float(spearmanr(gold, prediction).statistic)


def score_predictions(gold, prediction, languages) -> dict:
    gold, prediction, languages = np.asarray(gold), np.asarray(prediction), np.asarray(languages)
    result, macro = {}, []
    valid_all = np.isfinite(gold) & np.isfinite(prediction) & (gold != -1)
    for language in LANGUAGES:
        selected = valid_all & (languages == language)
        rho = safe_spearman(gold[selected], prediction[selected])
        result[f"spearman_{language}"] = rho
        result[f"n_tokens_{language}"] = int(selected.sum())
        if math.isfinite(rho):
            macro.append(rho)
    result["spearman_macro"] = float(np.mean(macro)) if macro else float("nan")
    result["spearman_pooled"] = safe_spearman(gold[valid_all], prediction[valid_all])
    result["n_tokens_pooled"] = int(valid_all.sum())
    return result
