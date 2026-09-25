"""JANG repacks of prism-ml's Hadamard-rotated ternary Bonsai 2 checkpoints
(prism.hadamard.v1 in config.json) load through prism_hadamard_qwen35, and
nothing else is routed there."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_lm import utils
from mlx_lm.models import prism_hadamard_qwen35 as prism


def repack_config():
    return {
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text"},
        "hadamard": {
            "contract": "prism.hadamard.v1",
            "block_size": 1024,
            "forward_modules": ["language_model.lm_head", "language_model.model.layers.0.mlp.up_proj"],
            "inverse_modules": ["language_model.model.embed_tokens"],
            "gdn_v_grouped": True,
        },
        "quantization": {
            "group_size": 128,
            "bits": 2,
            "mode": "affine",
            "language_model.lm_head": {"group_size": 128, "bits": 2, "mode": "affine", "storage_bits": 2},
            "vision_tower.blocks.0.attn.qkv": {"group_size": 128, "bits": 6, "mode": "affine", "storage_bits": 6},
        },
    }


class TestPrismHadamardRepack(unittest.TestCase):
    def adapt(self, config, jang_layout=None):
        with tempfile.TemporaryDirectory() as d:
            if jang_layout is not None:
                Path(d, "jang_config.json").write_text(json.dumps({"layout": jang_layout}))
            utils._adapt_prism_hadamard_repack(config, Path(d))
        return config

    def test_repack_routed_to_prism(self):
        config = self.adapt(repack_config(), {"language_norms": "zero-centered-runtime-plus-one"})
        self.assertEqual(config["model_type"], "prism_hadamard_qwen35")
        self.assertEqual(
            config["modules"],
            [
                {"path": "lm_head", "block": 1024, "embedding": False},
                {"path": "model.layers.0.mlp.up_proj", "block": 1024, "embedding": False},
                {"path": "model.embed_tokens", "block": 1024, "embedding": True},
            ],
        )
        # Rotated modules are built quantized by the model; no per-layer
        # entries (and no storage_bits) reach nn.quantize.
        self.assertEqual(config["quantization"], {"group_size": 128, "bits": 2, "mode": "affine"})
        self.assertTrue(config["zero_centered_norms"])

    def test_norm_convention_only_when_declared(self):
        self.assertFalse(self.adapt(repack_config(), {}).get("zero_centered_norms"))
        self.assertNotIn("zero_centered_norms", self.adapt(repack_config()))

    def test_other_configs_untouched(self):
        plain = {"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}}
        self.assertEqual(self.adapt(dict(plain)), plain)
        prism_native = {"model_type": "prism_hadamard_qwen35", "modules": [{"path": "lm_head", "block": 1024}]}
        self.assertEqual(self.adapt(dict(prism_native)), prism_native)
        other_contract = repack_config()
        other_contract["hadamard"]["contract"] = "something.else"
        self.assertEqual(self.adapt(other_contract)["model_type"], "qwen3_5")

    def test_unsupported_variant_refused(self):
        config = repack_config()
        config["model_type"] = "llama"
        with self.assertRaises(ValueError):
            self.adapt(config)
        config = repack_config()
        config["hadamard"]["gdn_v_grouped"] = False
        with self.assertRaises(ValueError):
            self.adapt(config)

    def test_zero_centered_norms_shifted_in_sanitize(self):
        weights = {
            "language_model.model.layers.0.input_layernorm.weight": mx.zeros((4,)),
            "language_model.model.layers.3.self_attn.q_norm.weight": mx.zeros((4,)),
            "language_model.model.norm.weight": mx.zeros((4,)),
            "language_model.model.layers.0.linear_attn.norm.weight": mx.zeros((4,)),
            "vision_tower.blocks.0.attn.qkv.weight": mx.zeros((4,)),
        }

        def run(zero_centered):
            fake = SimpleNamespace(
                args=SimpleNamespace(zero_centered_norms=zero_centered),
                language_model=SimpleNamespace(sanitize=lambda w: w),
            )
            return prism.Model.sanitize(fake, dict(weights))

        shifted = run(True)
        self.assertNotIn("vision_tower.blocks.0.attn.qkv.weight", shifted)
        for key in (
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.3.self_attn.q_norm.weight",
            "language_model.model.norm.weight",
        ):
            self.assertEqual(shifted[key].tolist(), [1.0] * 4, key)
        # GatedDeltaNet's own norm has no +1 offset.
        self.assertEqual(shifted["language_model.model.layers.0.linear_attn.norm.weight"].tolist(), [0.0] * 4)
        # prism's own checkpoints: untouched.
        self.assertEqual(run(False)["language_model.model.norm.weight"].tolist(), [0.0] * 4)


class TestPrismRepackRoundTrip(unittest.TestCase):
    """A loaded repack saved again (fuse / convert / dwq save the config
    load returned) must load back the same: it used to fail on the kept
    `hadamard` block, and to add the norms' 1 a second time."""

    def test_adapted_config_is_stable(self):
        config = TestPrismHadamardRepack.adapt(None, repack_config(), {"language_norms": "zero-centered-runtime-plus-one"})
        self.assertNotIn("hadamard", config, "translated into modules")
        again = json.loads(json.dumps(config))
        TestPrismHadamardRepack.adapt(None, again)
        self.assertEqual(again, config)
        # Saved by an mlx-lm that kept the block: its norms already hold the 1.
        old_save = dict(config, hadamard=repack_config()["hadamard"], zero_centered_norms=True)
        TestPrismHadamardRepack.adapt(None, old_save)
        self.assertEqual(old_save, dict(config, zero_centered_norms=False))

    def test_hf_layout_conv1d_norms_shifted_once(self):
        from mlx_lm.models import qwen3_5

        weights = {
            "language_model.model.norm.weight": mx.zeros((4,)),
            "language_model.model.layers.0.input_layernorm.weight": mx.zeros((4,)),
            # HF layout (C, 1, K): qwen3_5's sanitize adds the norms' 1 itself.
            "language_model.model.layers.0.linear_attn.conv1d.weight": mx.zeros((8, 1, 4)),
        }
        text = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False))
        fake = SimpleNamespace(
            args=SimpleNamespace(zero_centered_norms=True),
            language_model=SimpleNamespace(sanitize=lambda w: qwen3_5.TextModel.sanitize(text, w)),
        )
        out = prism.Model.sanitize(fake, dict(weights))
        self.assertEqual(out["language_model.model.norm.weight"].tolist(), [1.0] * 4)
        self.assertEqual(out["language_model.model.layers.0.input_layernorm.weight"].tolist(), [1.0] * 4)
        # MLX layout: prism's sanitize adds it (qwen3_5's doesn't).
        weights["language_model.model.layers.0.linear_attn.conv1d.weight"] = mx.zeros((8, 4, 1))
        out = prism.Model.sanitize(fake, dict(weights))
        self.assertEqual(out["language_model.model.norm.weight"].tolist(), [1.0] * 4)


class TestLayerQuantization(unittest.TestCase):
    def test_equal_storage_bits_dropped(self):
        self.assertEqual(
            utils._layer_quantization("x", {"bits": 2, "group_size": 128, "storage_bits": 2}),
            {"bits": 2, "group_size": 128},
        )

    def test_different_storage_bits_refused(self):
        with self.assertRaises(ValueError):
            utils._layer_quantization("x", {"bits": 3, "group_size": 64, "storage_bits": 4})

    def test_other_entries_pass_through(self):
        self.assertIs(utils._layer_quantization("x", False), False)
        entry = {"bits": 4, "group_size": 64}
        self.assertIs(utils._layer_quantization("x", entry), entry)


if __name__ == "__main__":
    unittest.main()
