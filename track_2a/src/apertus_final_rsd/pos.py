from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .ner import text_cache_key


LOGGER = logging.getLogger(__name__)
UPOS_LABELS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)


def coarse_pos_label(label: str) -> str:
    """Normalize token-classifier labels to Universal POS categories."""
    value = str(label).upper()
    if value.startswith("B-") or value.startswith("I-"):
        value = value[2:]
    aliases = {
        "CONJ": "CCONJ",
        "PRT": "PART",
        ".": "PUNCT",
    }
    value = aliases.get(value, value)
    return value if value in UPOS_LABELS else "X"


class MultilingualPosTagger:
    """Word-aligned multilingual UPOS tagging with overlapping windows."""

    def __init__(self, model_name: str, max_length: int, stride: int, batch_size: int):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the project inside the PyTorch uenv.") from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading multilingual POS tokenizer %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("The POS model must provide a fast tokenizer for word alignment.")
        LOGGER.info("Loading multilingual POS model %s", model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device).eval()
        self.id2label = {
            int(index): str(label) for index, label in self.model.config.id2label.items()
        }

    def tag_words(self, words: list[str]) -> list[str]:
        if not words:
            return []
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        word_ids = [encoding.word_ids(batch_index=index) for index in range(len(encoding["input_ids"]))]
        n_labels = int(self.model.config.num_labels)
        probability_sum = np.zeros((len(words), n_labels), dtype=np.float64)
        observation_count = np.zeros(len(words), dtype=np.int64)
        model_keys = {"input_ids", "attention_mask", "token_type_ids"}

        for start in range(0, len(word_ids), self.batch_size):
            stop = min(start + self.batch_size, len(word_ids))
            inputs = {
                key: value[start:stop].to(self.device)
                for key, value in encoding.items()
                if key in model_keys
            }
            with self.torch.inference_mode():
                probabilities = self.model(**inputs).logits.softmax(dim=-1).to("cpu").numpy()
            for local_index, chunk_index in enumerate(range(start, stop)):
                for token_index, word_index in enumerate(word_ids[chunk_index]):
                    if word_index is None or not (0 <= word_index < len(words)):
                        continue
                    probability_sum[word_index] += probabilities[local_index, token_index]
                    observation_count[word_index] += 1

        missing = np.flatnonzero(observation_count == 0)
        if len(missing):
            raise RuntimeError(f"POS tokenizer omitted words: {missing[:10].tolist()}")
        predicted = probability_sum / observation_count[:, None]
        return [coarse_pos_label(self.id2label[int(index)]) for index in predicted.argmax(axis=1)]

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def prepare_pos_cache(examples, config, cache_dir: Path) -> dict[str, list[str]]:
    """Load cached UPOS labels and tag only unique documents still missing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts: dict[str, tuple[str, list[str]]] = {}
    for example in examples:
        for text in (example.text_a, example.text_b):
            key = text_cache_key(config.pos_model_name, text)
            texts.setdefault(key, (text, text.split()))

    result: dict[str, list[str]] = {}
    missing: list[tuple[str, str, list[str], Path]] = []
    for key, (text, words) in texts.items():
        path = cache_dir / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model_name") != config.pos_model_name or payload.get("words") != words:
                raise RuntimeError(f"Invalid POS cache entry: {path}")
            labels = list(payload.get("labels", []))
            if len(labels) != len(words):
                raise RuntimeError(f"POS cache length mismatch: {path}")
            result[key] = labels
        else:
            missing.append((key, text, words, path))

    if missing:
        tagger = MultilingualPosTagger(
            config.pos_model_name,
            config.pos_max_length,
            config.pos_stride,
            config.pos_batch_size,
        )
        try:
            for key, _text, words, path in tqdm(missing, desc="Multilingual POS"):
                labels = tagger.tag_words(words)
                payload = {"model_name": config.pos_model_name, "words": words, "labels": labels}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                temporary.replace(path)
                result[key] = labels
        finally:
            tagger.close()
    return result
