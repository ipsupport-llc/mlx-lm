# Copyright © 2026 IPSupport LLC

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models import lfm3


def _args(window=4):
    return lfm3.ModelArgs.from_dict(
        {
            "model_type": "lfm3",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "norm_eps": 1e-5,
            "conv_bias": False,
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention", "sliding_attention", "conv"],
            "sliding_window": window,
            "rope_parameters": {
                "full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 1e4},
            },
        }
    )


class TestLfm3(unittest.TestCase):
    def test_partial_rope_leaves_the_tail(self):
        attn = lfm3.Attention(_args(), "full_attention")
        x = mx.random.normal((1, 1, 3, 8))
        y = attn._rope(x, 5)
        # Head dim 8, factor 0.5: pairs (0,4) and (1,5) rotate, (2,6) and (3,7) don't.
        self.assertTrue(mx.allclose(y[..., [2, 3, 6, 7]], x[..., [2, 3, 6, 7]]).item())
        self.assertFalse(mx.allclose(y[..., [0, 4]], x[..., [0, 4]]).item())

    def test_cached_steps_match_one_pass(self):
        mx.random.seed(1)
        model = lfm3.Model(_args(window=4))
        tokens = mx.array([[3, 9, 1, 7, 12, 5, 30, 2, 8, 4]])
        full = model(tokens)
        cache = model.make_cache()
        first = model(tokens[:, :3], cache=cache)
        steps = [model(tokens[:, i : i + 1], cache=cache) for i in range(3, tokens.shape[1])]
        stepped = mx.concatenate([first] + steps, axis=1)
        # Past the sliding window too.
        np.testing.assert_allclose(np.array(stepped), np.array(full), atol=1e-4, rtol=1e-4)

    def test_sanitize(self):
        model = lfm3.Model(_args())
        w = {
            "language_model.model.embed_tokens.weight": mx.zeros((64, 32)),
            "language_model.model.layers.0.conv.conv.weight": mx.zeros((32, 3, 1)),
            "lm_head.weight": mx.zeros((64, 32)),
        }
        out = model.sanitize(w)
        self.assertEqual(sorted(out), ["model.embed_tokens.weight", "model.layers.0.conv.conv.weight"])


if __name__ == "__main__":
    unittest.main()
