# Copyright © 2026 IPSupport LLC

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.expert_offload import OffloadedSwitchGLU
from mlx_lm.models import qwen3_moe
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model


def _tiny_moe(path):
    config = {
        "model_type": "qwen3_moe",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "moe_intermediate_size": 64,
        "rms_norm_eps": 1e-5,
        "vocab_size": 100,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
        "max_position_embeddings": 256,
        "norm_topk_prob": True,
        "rope_theta": 10000.0,
    }
    mx.random.seed(0)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    config["quantization"] = {"group_size": 32, "bits": 4}
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))


def _run(model, prompt, steps):
    cache = make_prompt_cache(model)
    logits = model(prompt, cache=cache)
    outs = [logits[:, -1]]
    y = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(steps):
        logits = model(y[:, None], cache=cache)
        outs.append(logits[:, -1])
        y = mx.argmax(logits[:, -1], axis=-1)
    return mx.stack(outs)


class TestExpertOffload(unittest.TestCase):
    def test_same_logits_as_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            _tiny_moe(path)
            prompt = mx.array([[1, 5, 9, 13, 2, 7, 3, 11, 4, 6]])
            ref, _ = load_model(path)
            expected = _run(ref, prompt, 6)
            for share in (1.0, 0.25):
                model, _ = load_model(path, model_config={"experts_on_disk": share})
                layers = [m for _, m in model.named_modules() if isinstance(m, OffloadedSwitchGLU)]
                self.assertEqual(len(layers), 2)
                got = _run(model, prompt, 6)
                self.assertTrue(mx.allclose(got, expected, atol=1e-3, rtol=1e-3).item(), f"share {share}")
                # Never more experts kept than the share allows.
                for m in layers:
                    self.assertLessEqual(len(m._expert_store.cache), max(1, round(8 * share)))


class TestFilePrefix(unittest.TestCase):
    def test_names(self):
        from mlx_lm.expert_offload import _file_prefix

        index = {
            "model.language_model.layers.3.experts.switch_glu.gate_proj.weight": 0,
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": 0,
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": 0,
        }
        # Wrapper names in another order.
        self.assertEqual(
            _file_prefix("language_model.model.layers.3.experts.switch_glu", index),
            "model.language_model.layers.3.experts.switch_glu",
        )
        # An MTP head's layer 0 is not the model's.
        self.assertEqual(_file_prefix("model.layers.0.mlp.switch_mlp", index), "model.layers.0.mlp.switch_mlp")
        self.assertEqual(_file_prefix("mtp.layers.0.mlp.switch_mlp", index), "mtp.layers.0.mlp.switch_mlp")
        self.assertIsNone(_file_prefix("other.layers.0.mlp.switch_mlp", index))
        # A repeated name is not the same as one.
        self.assertIsNone(_file_prefix("model.model.layers.0.mlp.switch_mlp", index))


if __name__ == "__main__":
    unittest.main()
