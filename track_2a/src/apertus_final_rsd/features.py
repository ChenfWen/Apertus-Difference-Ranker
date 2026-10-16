from __future__ import annotations

import numpy as np

BASE_FEATURES = (
    "top1_distance",
    "top5_distance",
    "top1_top2_margin",
    "alignment_entropy",
    "row_mean_cosine",
    "row_std_cosine",
    "mutual_nearest_fraction",
    "source_hidden_norm",
)

STATIC_FEATURES = (
    "subword_count",
    "relative_position",
    "log_character_length",
    "side_b",
)


def _pool(values: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    return np.asarray([values[group].mean(axis=0) for group in groups], np.float32)


def direction_features(source, target, layers, torch, temperature: float = 0.05):
    all_layers = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for layer in layers:
            raw_source = source.by_layer[layer].to(device=device, dtype=torch.float32)
            raw_target = target.by_layer[layer].to(device=device, dtype=torch.float32)
            a = torch.nn.functional.normalize(raw_source, p=2, dim=1, eps=1e-12)
            b = torch.nn.functional.normalize(raw_target, p=2, dim=1, eps=1e-12)
            cosine = (a @ b.T).clamp(-1.0, 1.0)
            k = min(5, cosine.shape[1])
            top = cosine.topk(k, dim=1).values
            top1 = top[:, 0]
            top2 = top[:, 1] if k > 1 else top1
            source_choice = cosine.argmax(dim=1)
            target_choice = cosine.argmax(dim=0)
            mutual = target_choice[source_choice] == torch.arange(len(a), device=cosine.device)
            probabilities = torch.softmax(cosine / temperature, dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            if cosine.shape[1] > 1:
                entropy = entropy / np.log(cosine.shape[1])
            subword_features = torch.stack(
                (
                    0.5 * (1.0 - top1),
                    0.5 * (1.0 - top.mean(dim=1)),
                    top1 - top2,
                    entropy,
                    cosine.mean(dim=1),
                    cosine.std(dim=1, unbiased=False),
                    mutual.to(torch.float32),
                    raw_source.norm(dim=1),
                ),
                dim=1,
            ).cpu().numpy()
            all_layers.append(_pool(subword_features, source.subwords_by_word))
    return np.stack(all_layers, axis=1)


def static_features(representations, side_b: bool) -> np.ndarray:
    count = len(representations.words)
    denominator = max(1, count - 1)
    return np.asarray(
        [
            (
                len(representations.subwords_by_word[index]),
                index / denominator,
                np.log1p(len(word)),
                float(side_b),
            )
            for index, word in enumerate(representations.words)
        ],
        np.float32,
    )


def raw_word_pairs(source, target, layer: int, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        source_words = torch.stack(
            [source.by_layer[layer][indices].to(device=device, dtype=torch.float32).mean(0) for indices in source.subwords_by_word]
        )
        target_words = torch.stack(
            [target.by_layer[layer][indices].to(device=device, dtype=torch.float32).mean(0) for indices in target.subwords_by_word]
        )
        a = torch.nn.functional.normalize(source_words, p=2, dim=1, eps=1e-12)
        b = torch.nn.functional.normalize(target_words, p=2, dim=1, eps=1e-12)
        counterpart = (a @ b.T).argmax(dim=1)
        matched = target_words[counterpart]
    return source_words.cpu().numpy(), matched.cpu().numpy()


def vector_matrix(layer_features: np.ndarray, static: np.ndarray, selected_indices: list[int], sides):
    selected = layer_features[:, selected_indices, :]
    flattened = selected.reshape(len(selected), -1)
    # Local context is derived only within each document side.
    top1_columns = selected[:, :, 0]
    previous = np.zeros_like(top1_columns)
    following = np.zeros_like(top1_columns)
    sides = np.asarray(sides)
    for side in ("a", "b"):
        indices = np.flatnonzero(sides == side)
        if len(indices):
            previous[indices[1:]] = top1_columns[indices[:-1]]
            previous[indices[0]] = top1_columns[indices[0]]
            following[indices[:-1]] = top1_columns[indices[1:]]
            following[indices[-1]] = top1_columns[indices[-1]]
    return np.concatenate((flattened, previous, following, static), axis=1).astype(np.float32)
