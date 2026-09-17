# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import nemotron_h_mtp_generate_step
from mlx_lm.models.nemotron_h import Model, ModelArgs


def tiny_args(**overrides):
    kwargs = dict(
        model_type="nemotron_h",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        max_position_embeddings=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=16,
        mamba_proj_bias=False,
        ssm_state_size=16,
        conv_kernel=3,
        n_groups=2,
        time_step_limit=(0.0, float("inf")),
        mlp_bias=False,
        layer_norm_epsilon=1e-4,
        use_bias=True,
        use_conv_bias=True,
        hybrid_override_pattern=["*", "M", "*", "M"],
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "attention"],
    )
    kwargs.update(overrides)
    return ModelArgs(**kwargs)


class TestNemotronHMTPGenerate(unittest.TestCase):
    def test_matches_greedy_decoding(self):
        """MTP self-speculative decoding must be a pure speedup: identical
        token stream to plain greedy backbone-only decoding."""
        mx.random.seed(0)
        model = Model(tiny_args())
        prompt = mx.random.randint(0, 64, (6,))
        n_tokens = 12

        cache = model.make_cache()
        hidden = model.backbone(prompt[None], cache=cache)
        tok = mx.argmax(model.lm_head(hidden[:, -1, :]), axis=-1)
        baseline = [tok.item()]
        for _ in range(n_tokens - 1):
            hidden = model.backbone(tok.reshape(1, 1), cache=cache)
            tok = mx.argmax(model.lm_head(hidden[:, -1, :]), axis=-1)
            baseline.append(tok.item())

        spec = [
            tok
            for tok, _, _ in nemotron_h_mtp_generate_step(
                prompt, model, max_tokens=n_tokens
            )
        ]
        self.assertEqual(spec, baseline)

    def test_from_draft_flags_are_consistent(self):
        """Every yielded token is real regardless of from_draft, and
        acceptance never emits two draft-flagged tokens in a row (each
        accept immediately yields one non-draft bonus token)."""
        mx.random.seed(1)
        model = Model(tiny_args())
        prompt = mx.random.randint(0, 64, (5,))

        flags = [
            from_draft
            for _, _, from_draft in nemotron_h_mtp_generate_step(
                prompt, model, max_tokens=20
            )
        ]
        for i, f in enumerate(flags):
            if f:
                self.assertFalse(
                    i + 1 < len(flags) and flags[i + 1],
                    "a draft accept must be followed by a bonus (non-draft) token",
                )


if __name__ == "__main__":
    unittest.main()
