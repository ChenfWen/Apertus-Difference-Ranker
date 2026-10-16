from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Representations:
    words: list[str]
    by_layer: dict[int, Any]
    subwords_by_word: list[list[int]]
    subword_to_word: list[int]
    model_tokens: int


def _nested_config(config):
    return getattr(config, "text_config", config)


def _hidden_states(output):
    direct = getattr(output, "hidden_states", None)
    if direct is not None:
        return direct
    for name in ("language_model_output", "text_model_output", "model_output"):
        nested = getattr(output, name, None)
        hidden = getattr(nested, "hidden_states", None) if nested is not None else None
        if hidden is not None:
            return hidden
    return None


class ApertusExtractor:
    def __init__(self, model_name: str, layers: tuple[int, ...] | None, max_length: int,
                 dtype: str = "auto", storage_dtype: str = "float32"):
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Run inside the configured PyTorch uenv.") from exc
        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "float32" or self.device.type == "cpu":
            model_dtype = torch.float32
        elif dtype == "float16":
            model_dtype = torch.float16
        elif dtype == "bfloat16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.storage_dtype = torch.float16 if storage_dtype == "float16" else torch.float32
        LOGGER.info("Loading tokenizer %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for word/subword mapping")
        kwargs = {"dtype": model_dtype, "low_cpu_mem_usage": True}
        if self.device.type == "cuda":
            kwargs["device_map"] = "auto"
        LOGGER.info("Loading model %s", model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs).eval()
        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.num_layers = int(_nested_config(self.model.config).num_hidden_layers)
        self.layers = tuple(range(1, self.num_layers + 1)) if layers is None else tuple(layers)
        invalid = [layer for layer in self.layers if layer < 1 or layer > self.num_layers]
        if invalid:
            raise ValueError(f"Invalid layers {invalid}; model has {self.num_layers}")
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except (AttributeError, NotImplementedError):
            self.input_device = next(self.model.parameters()).device
        self.forward_passes = 0
        self.hidden_size = int(_nested_config(self.model.config).hidden_size)

    def clear_cache(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    @staticmethod
    def _layout(word_ids, n_words):
        positions, groups, reverse = [], [[] for _ in range(n_words)], []
        for model_position, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            lexical_position = len(positions)
            positions.append(model_position)
            groups[word_id].append(lexical_position)
            reverse.append(word_id)
        missing = [index for index, group in enumerate(groups) if not group]
        if missing:
            raise RuntimeError(f"Tokenizer omitted words: {missing[:10]}")
        return positions, groups, reverse

    def extract(self, text: str) -> Representations:
        words = text.split()
        if not words:
            raise ValueError("Cannot encode empty text")
        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=True,
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        word_ids = encoding.word_ids(0)
        retained = [word_id for word_id in word_ids if word_id is not None]
        if not retained or max(retained) + 1 != len(words):
            raise RuntimeError(f"max_length={self.max_length} truncated {len(words)} words")
        positions, groups, reverse = self._layout(word_ids, len(words))
        inputs = {
            key: value.to(self.input_device) for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with self.torch.inference_mode():
            output = self.model(
                **inputs, output_hidden_states=True, return_dict=True, use_cache=False
            )
        self.forward_passes += 1
        hidden_states = _hidden_states(output)
        if hidden_states is None or len(hidden_states) != self.num_layers + 1:
            observed = None if hidden_states is None else len(hidden_states)
            raise RuntimeError(f"Expected {self.num_layers + 1} hidden states; received {observed}")
        by_layer = {}
        for layer in self.layers:
            hidden = hidden_states[layer][0]
            index = self.torch.tensor(positions, device=hidden.device, dtype=self.torch.long)
            values = hidden.index_select(0, index).to(dtype=self.torch.float32)
            if not self.torch.isfinite(values).all():
                raise RuntimeError(f"Non-finite hidden state at layer {layer}")
            by_layer[layer] = values.to("cpu", dtype=self.storage_dtype)
        result = Representations(words, by_layer, groups, reverse, int(encoding["input_ids"].shape[1]))
        del output, hidden_states, inputs, encoding
        self.clear_cache()
        return result
