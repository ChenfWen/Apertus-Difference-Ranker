import json

import numpy as np

from apertus_final_rsd.data import parse_example, resolve_split_name
from apertus_final_rsd.evaluation import safe_spearman, score_predictions
from apertus_final_rsd.features import BASE_FEATURES, STATIC_FEATURES, vector_matrix
from apertus_final_rsd.linguistic import (
    LINGUISTIC_DIMENSION,
    encode_linguistic_features,
    normalize_linguistic_labels,
    observable_punctuation_mask,
)
from apertus_final_rsd.ner import coarse_ner_label
from apertus_final_rsd.pos import coarse_pos_label
from apertus_final_rsd.split import document_split, stratified_subset
from apertus_final_rsd.scan import _score_layers
from apertus_final_rsd.training import build_feature_regressor, build_pair_regressor


class Example:
    def __init__(self, identifier, language):
        self.example_id = identifier
        self.language = language


def test_resolve_dev_alias():
    assert resolve_split_name("dev", ["train", "validation"]) == "validation"


def test_parse_example_and_language():
    value = parse_example({
        "id": "admin_de_1", "text_a": "Hello .", "text_b": "Hallo .",
        "labels_a": json.dumps([0, -1]), "labels_b": [0, -1],
    }, 0)
    assert value.language == "de"
    assert len(value.labels_a) == 2


def test_document_split_has_no_overlap_and_is_stratified():
    examples = [Example(f"{language}_{index}", language) for language in ("de", "fr", "it") for index in range(20)]
    train, validation = document_split(examples, validation_per_language=3, seed=42)
    assert train.isdisjoint(validation)
    assert len(validation) == 9
    assert {identifier.split("_")[0] for identifier in validation} == {"de", "fr", "it"}


def test_stratified_subset_contains_all_languages():
    examples = [Example(f"{language}_{index}", language) for language in ("de", "fr", "it") for index in range(10)]
    selected = stratified_subset(examples, 9, seed=42)
    assert len(selected) == 9
    assert {example.language for example in selected} == {"de", "fr", "it"}


def test_vector_matrix_shape_and_side_local_context():
    # 4 tokens, 2 layers, 8 base features.
    features = np.zeros((4, 2, len(BASE_FEATURES)), np.float32)
    features[:, :, 0] = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40]])
    static = np.zeros((4, len(STATIC_FEATURES)), np.float32)
    result = vector_matrix(features, static, [0, 1], np.asarray(["a", "a", "b", "b"]))
    assert result.shape == (4, 2 * len(BASE_FEATURES) + 2 * 2 + len(STATIC_FEATURES))
    # Previous context resets at the beginning of side B.
    previous_start = 2 * len(BASE_FEATURES)
    np.testing.assert_array_equal(result[2, previous_start:previous_start + 2], [3, 30])


def test_spearman_excludes_punctuation():
    assert np.isclose(safe_spearman([0, 1, -1], [0.1, 0.9, 100]), 1.0)


def test_macro_spearman():
    result = score_predictions(
        [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1], ["de", "de", "fr", "fr", "it", "it"]
    )
    assert np.isclose(result["spearman_macro"], 1.0)


def test_every_layer_receives_a_plotting_row():
    checkpoint = {
        "gold": np.asarray([0.0, 1.0], np.float32),
        "language": np.asarray("de"),
        "features": np.asarray(
            [
                [[0.0] * len(BASE_FEATURES), [1.0] + [0.0] * (len(BASE_FEATURES) - 1)],
                [[1.0] + [0.0] * (len(BASE_FEATURES) - 1), [0.0] * len(BASE_FEATURES)],
            ],
            np.float32,
        ),
    }
    rows = _score_layers([checkpoint], (1, 2))
    assert [row["layer"] for row in rows] == [1, 2]
    assert np.isclose(rows[0]["spearman_de"], 1.0)
    assert np.isclose(rows[1]["spearman_de"], -1.0)


def test_representation_head_shape():
    import pytest

    torch = pytest.importorskip("torch")
    model = build_pair_regressor(16, 8, 0.1, torch)
    output = model(torch.randn(4, 16), torch.randn(4, 16))
    assert output.shape == (4,)
    assert torch.all((output >= 0) & (output <= 1))


def test_feature_head_shape_and_parameter_count():
    import pytest

    torch = pytest.importorskip("torch")
    model = build_feature_regressor(54, 128, 0.1, torch)
    output = model(torch.randn(4, 54))
    assert output.shape == (4,)
    assert torch.all((output >= 0) & (output <= 1))
    assert sum(parameter.numel() for parameter in model.parameters()) == 7277


def test_linguistic_encoding_is_fixed_and_punctuation_is_explicit():
    punctuation = observable_punctuation_mask(["energy", ".", "works"], ["NOUN", "PUNCT", "VERB"])
    ner, pos = normalize_linguistic_labels(
        ["O", "ORG", "B-PER"], ["NOUN", "PUNCT", "VERB"], punctuation
    )
    assert ner.tolist() == ["O", "PUNCT", "MISC"]
    assert pos.tolist() == ["NOUN", "PUNCT", "VERB"]
    encoded = encode_linguistic_features(ner, pos)
    assert encoded.shape == (3, LINGUISTIC_DIMENSION)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.full(3, 2.0))


def test_coarse_tag_normalization():
    assert coarse_ner_label("B-PER") == "PER"
    assert coarse_ner_label("I-ORGANIZATION") == "ORG"
    assert coarse_pos_label("B-PROPN") == "PROPN"
    assert coarse_pos_label("CONJ") == "CCONJ"


def test_linguistic_representation_head_shape():
    import pytest

    torch = pytest.importorskip("torch")
    model = build_pair_regressor(16, 8, 0.1, torch, LINGUISTIC_DIMENSION)
    output = model(
        torch.randn(4, 16),
        torch.randn(4, 16),
        torch.randn(4, LINGUISTIC_DIMENSION),
    )
    assert output.shape == (4,)
    with pytest.raises(ValueError):
        model(torch.randn(4, 16), torch.randn(4, 16), None)
