# Copyright © 2026 Apple Inc.

"""Qwen3.5 MoE (qwen3_5_moe) keeps its vision tower: an HF-layout
checkpoint (fused experts, model.visual.*) loads strictly with it."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import qwen3_5_moe
from mlx_lm.multimodal import Qwen35ImageInputs, load_image_inputs

TEXT = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": 1000,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 3,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_expert_intermediate_size": 128,
    "moe_intermediate_size": 64,
    "rms_norm_eps": 1e-5,
    "head_dim": 64,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.25,
    "max_position_embeddings": 1000,
    "full_attention_interval": 4,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [3, 3, 2],
        "rope_theta": 1000.0,
    },
}
VISION = {
    "depth": 1,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 2,
    "out_hidden_size": 128,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 64,
    "deepstack_visual_indexes": [],
}
CONFIG = {
    "model_type": "qwen3_5_moe",
    "text_config": TEXT,
    "vision_config": VISION,
    "image_token_id": 999,
    "vision_start_token_id": 997,
    "vision_end_token_id": 998,
}


def _model():
    return qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(CONFIG))


def _hf_checkpoint(model):
    """The model's weights the way an HF qwen3_5_moe checkpoint names them."""
    out = {}
    params = dict(tree_flatten(model.parameters()))
    for k, v in params.items():
        if k.startswith("vision_tower."):
            out["model.visual." + k[len("vision_tower.") :]] = v
        elif ".switch_mlp." in k:
            continue
        else:
            out[k.replace("language_model.model", "model.language_model", 1)] = v
    for l in range(TEXT["num_hidden_layers"]):
        p = f"language_model.model.layers.{l}.mlp.switch_mlp"
        hf = f"model.language_model.layers.{l}.mlp.experts"
        out[f"{hf}.gate_up_proj"] = mx.concatenate(
            [params[f"{p}.gate_proj.weight"], params[f"{p}.up_proj.weight"]], axis=-2
        )
        out[f"{hf}.down_proj"] = params[f"{p}.down_proj.weight"]
    return params, out


class TestQwen35MoeVision(unittest.TestCase):
    def test_vision_tower_is_built_and_kept(self):
        model = _model()
        self.assertIsNotNone(model.vision_tower)
        params, hf = _hf_checkpoint(model)
        target = _model()
        sanitized = target.sanitize(hf)
        target.load_weights(list(sanitized.items()), strict=True)
        loaded = dict(tree_flatten(target.parameters()))
        self.assertEqual(loaded.keys(), params.keys())
        for k, v in params.items():
            self.assertTrue(mx.array_equal(loaded[k], v).item(), k)

    def test_mtp_dropped(self):
        model = _model()
        _, hf = _hf_checkpoint(model)
        hf["mtp.fc.weight"] = mx.zeros((4, 4))
        self.assertNotIn("mtp.fc.weight", model.sanitize(hf))

    def test_image_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsInstance(
                load_image_inputs(_model(), Path(d)), Qwen35ImageInputs
            )


if __name__ == "__main__":
    unittest.main()
