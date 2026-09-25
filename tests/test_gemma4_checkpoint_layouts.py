# Copyright © 2026 Apple Inc.

"""Gemma 4 checkpoints in the layouts they actually come in: HF (torch
conv layout), saved by mlx (converted models, mlx-vlm's mlx-community
E2B/E4B), and upstream mlx-lm's text-only conversions (towers described in
the config, their weights gone). Each used to fail or silently lose a
modality; see the fork review."""

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import gemma4
from mlx_lm.utils import save_config

CONFIG = json.loads(r'''{"audio_config": {"attention_chunk_size": 12, "attention_context_left": 13, "attention_context_right": 0, "attention_invalid_logits_value": -1000000000.0, "attention_logit_cap": 50.0, "conv_kernel_size": 5, "gradient_clipping": 10000000000.0, "hidden_act": "silu", "hidden_size": 64, "model_type": "gemma4_audio", "num_attention_heads": 2, "num_hidden_layers": 1, "output_proj_dims": 64, "residual_weight": 0.5, "rms_norm_eps": 1e-06, "subsampling_conv_channels": [16, 8], "use_clipped_linears": true}, "audio_token_id": 258881, "boa_token_id": 256000, "boi_token_id": 255999, "eoi_token_id": 258882, "image_token_id": 258880, "model_type": "gemma4", "text_config": {"attention_bias": false, "attention_dropout": 0.0, "attention_k_eq_v": false, "bos_token_id": 2, "enable_moe_block": false, "eos_token_id": 1, "final_logit_softcapping": 30.0, "global_head_dim": 64, "head_dim": 32, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64, "hidden_size_per_layer_input": 8, "intermediate_size": 128, "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"], "max_position_embeddings": 131072, "model_type": "gemma4_text", "moe_intermediate_size": null, "num_attention_heads": 2, "num_experts": null, "num_hidden_layers": 6, "num_key_value_heads": 1, "num_kv_shared_layers": 0, "pad_token_id": 0, "rms_norm_eps": 1e-06, "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"}, "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}, "sliding_window": 512, "tie_word_embeddings": true, "top_k_experts": null, "use_bidirectional_attention": null, "use_cache": true, "use_double_wide_mlp": false, "vocab_size": 262144, "vocab_size_per_layer_input": 262144}, "tie_word_embeddings": true, "vision_config": {"attention_bias": false, "attention_dropout": 0.0, "head_dim": 32, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64, "intermediate_size": 128, "max_position_embeddings": 131072, "model_type": "gemma4_vision", "num_attention_heads": 2, "num_hidden_layers": 1, "num_key_value_heads": 2, "patch_size": 8, "pooling_kernel_size": 2, "position_embedding_size": 64, "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"}, "standardize": false, "use_clipped_linears": true}}''')


def _model():
    mx.random.seed(0)
    model = gemma4.Model(gemma4.ModelArgs.from_dict(CONFIG))
    mx.eval(model.parameters())
    return model


def _checkpoint(model, torch_layout):
    """The model's own weights, as a checkpoint would carry them."""
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if torch_layout and k.startswith("audio_tower.") and k.endswith(("layer0.conv.weight", "layer1.conv.weight")):
            v = v.transpose(0, 3, 1, 2)          # mlx (O, kh, kw, I) -> torch (O, I, kh, kw)
        if torch_layout and k.startswith("audio_tower.") and k.endswith("depthwise_conv1d.conv.weight"):
            k = k.replace("depthwise_conv1d.conv.weight", "depthwise_conv1d.weight")
            v = v.transpose(0, 2, 1)             # mlx (O, K, I) -> torch (O, I, K)
        weights[k] = v
    return weights


class TestGemma4CheckpointLayouts(unittest.TestCase):
    def test_audio_conv_weights_in_either_layout(self):
        model = _model()
        own = dict(tree_flatten(model.parameters()))
        for torch_layout in (True, False):
            sanitized = _model().sanitize(_checkpoint(model, torch_layout))
            for k, v in own.items():
                if k.startswith("audio_tower."):
                    self.assertEqual(sanitized[k].shape, v.shape, (torch_layout, k))
                    self.assertTrue(mx.array_equal(sanitized[k], v).item(), (torch_layout, k))
        # The mlx-vlm naming (no ".conv" in the depthwise key), mlx layout.
        vlm = {k.replace("depthwise_conv1d.conv.weight", "depthwise_conv1d.weight"): v
               for k, v in _checkpoint(model, False).items()}
        target = _model()
        target.load_weights(list(target.sanitize(vlm).items()), strict=True)

    def test_towers_without_weights_are_dropped(self):
        model = _model()
        text_only = {k: v for k, v in _checkpoint(model, False).items()
                     if not k.startswith(("audio_tower.", "embed_audio.", "vision_tower.", "embed_vision."))}
        target = _model()
        target.load_weights(list(target.sanitize(text_only).items()), strict=True)
        self.assertIsNone(target.audio_tower)
        self.assertIsNone(target.vision_tower)
        no_audio = {k: v for k, v in _checkpoint(model, False).items() if not k.startswith(("audio_tower.", "embed_audio."))}
        target = _model()
        target.load_weights(list(target.sanitize(no_audio).items()), strict=True)
        self.assertIsNone(target.audio_tower)
        self.assertIsNotNone(target.vision_tower)

    def test_save_config_keeps_gemma4_vision_config(self):
        with tempfile.TemporaryDirectory() as d:
            import json
            save_config(dict(CONFIG), Path(d) / "config.json")
            self.assertIn("vision_config", json.loads((Path(d) / "config.json").read_text()))
            save_config({"model_type": "llava", "vision_config": {}}, Path(d) / "c2.json")
            self.assertNotIn("vision_config", json.loads((Path(d) / "c2.json").read_text()))

    def test_quantized_patch_projection_keeps_the_image(self):
        model = _model()
        pe = model.vision_tower.patch_embedder
        nn.quantize(pe, group_size=64, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear))
        self.assertEqual(pe.input_proj.weight.dtype, mx.uint32)
        dim = CONFIG["vision_config"]["patch_size"] ** 2 * 3
        pos = mx.array([[[0, 0], [0, 1], [1, 0], [1, 1]]])
        pad = mx.zeros((1, 4), dtype=mx.bool_)
        a = pe(mx.random.uniform(0, 1, (1, 4, dim)), pos, pad)   # pixels in [0, 1]
        b = pe(mx.random.uniform(0, 1, (1, 4, dim)), pos, pad)
        self.assertGreater(mx.abs(a - b).max().item(), 0.1, "different images, different embeddings")


if __name__ == "__main__":
    unittest.main()
