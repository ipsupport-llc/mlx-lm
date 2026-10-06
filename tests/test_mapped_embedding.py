# Copyright © 2026 Apple Inc.

"""Lookup tables read from the weights file (mmap_lookup_tables) and towers
loaded on first use (lazy_towers) give the same outputs as a normal load."""

import copy
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import gemma4
from mlx_lm.models.mapped_embedding import MappedEmbedding
from mlx_lm.utils import load_model, save_config
from tests.test_gemma4_checkpoint_layouts import CONFIG


def _config(quantized):
    config = copy.deepcopy(CONFIG)
    text = config["text_config"]
    text["vocab_size"] = text["vocab_size_per_layer_input"] = 1024
    text["hidden_size_per_layer_input"] = 16
    for key in (
        "audio_token_id",
        "boa_token_id",
        "boi_token_id",
        "eoi_token_id",
        "image_token_id",
    ):
        config[key] = 1000
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 4}
    return config


def _save(path, config, dtype):
    mx.random.seed(0)
    model = gemma4.Model(gemma4.ModelArgs.from_dict(config))
    model.set_dtype(dtype)
    if "quantization" in config:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda p, m: p.startswith("language_model")
            and hasattr(m, "to_quantized")
            and m.weight.shape[-1] % 32 == 0,
        )
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    save_config(config, path / "config.json")


class TestMappedEmbedding(unittest.TestCase):
    def _check(self, quantized, dtype):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            config = _config(quantized)
            _save(path, config, dtype)
            ids = mx.array([[5, 17, 1023, 5, 0, 512]])
            base, _ = load_model(path)
            mapped, _ = load_model(
                path, model_config={"mmap_lookup_tables": True, "lazy_towers": True}
            )
            table = mapped.language_model.model.embed_tokens_per_layer
            self.assertIsInstance(table, MappedEmbedding)
            names = [k for k, _ in tree_flatten(mapped.parameters())]
            self.assertFalse(any("embed_tokens_per_layer" in k for k in names))
            self.assertTrue(mx.array_equal(base(ids), mapped(ids)).item())
            expected = base.language_model.model.embed_tokens_per_layer(ids)
            self.assertTrue(mx.array_equal(expected, table(ids)).item())
            self.assertEqual(expected.dtype, table(ids).dtype)

    def test_quantized_table(self):
        self._check(quantized=True, dtype=mx.bfloat16)

    def test_bf16_table(self):
        self._check(quantized=False, dtype=mx.bfloat16)

    def test_float16_table(self):
        self._check(quantized=False, dtype=mx.float16)

    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            _save(path, _config(True), mx.bfloat16)
            model, _ = load_model(path)
            self.assertNotIsInstance(
                model.language_model.model.embed_tokens_per_layer, MappedEmbedding
            )

    def test_changed_table_stays_loaded(self):
        # A table sanitize rewrote is no longer the file's: it stays loaded.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            _save(path, _config(False), mx.bfloat16)
            key = "language_model.model.embed_tokens_per_layer.weight"

            def sanitize(self, weights, original=gemma4.Model.sanitize):
                weights = original(self, weights)
                weights[key] = weights[key] * 1
                return weights

            gemma4.Model.sanitize, saved = sanitize, gemma4.Model.sanitize
            try:
                model, _ = load_model(path, model_config={"mmap_lookup_tables": True})
            finally:
                gemma4.Model.sanitize = saved
            self.assertNotIsInstance(
                model.language_model.model.embed_tokens_per_layer, MappedEmbedding
            )


if __name__ == "__main__":
    unittest.main()
