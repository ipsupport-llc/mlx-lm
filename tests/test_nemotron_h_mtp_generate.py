# Copyright © 2026 Apple Inc.

import functools
import unittest

import mlx.core as mx

from mlx_lm.generate import maybe_quantize_kv_cache, nemotron_h_mtp_generate_step
from mlx_lm.models.cache import QuantizedKVCache
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

    def test_matches_greedy_decoding_with_quantized_kv_cache(self):
        """kv_bits must produce the exact same token stream as plain
        backbone-only decoding quantized the same way -- NOT vs a
        full-precision baseline, since 4-bit KV quantization is itself
        lossy and would diverge from full precision regardless of MTP.
        The invariant this checks is narrower and more important: MTP's
        speculative accept/reject must track the backbone's own quantized-
        cache forward exactly, same as the full-precision case."""
        mx.random.seed(2)
        # mx.quantize only supports group_size in {32, 64, 128} -- the
        # shared tiny_args() head_dim (8) is too small for any of them, so
        # this test uses a wider hidden_size to get head_dim=32.
        model = Model(tiny_args(hidden_size=128, intermediate_size=256))
        prompt = mx.random.randint(0, 64, (6,))
        n_tokens = 16
        quantize_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=0,
            kv_group_size=32,
            kv_bits=4,
        )

        cache = model.make_cache()
        hidden = model.backbone(prompt[None], cache=cache)
        quantize_fn(cache)
        tok = mx.argmax(model.lm_head(hidden[:, -1, :]), axis=-1)
        baseline = [tok.item()]
        for _ in range(n_tokens - 1):
            hidden = model.backbone(tok.reshape(1, 1), cache=cache)
            quantize_fn(cache)
            tok = mx.argmax(model.lm_head(hidden[:, -1, :]), axis=-1)
            baseline.append(tok.item())

        quant_cache = model.make_cache()
        spec = [
            tok
            for tok, _, _ in nemotron_h_mtp_generate_step(
                prompt,
                model,
                max_tokens=n_tokens,
                prompt_cache=quant_cache,
                kv_bits=4,
                kv_group_size=32,
                quantized_kv_start=0,
            )
        ]
        self.assertEqual(spec, baseline)
        full_attention_caches = [
            c for c in quant_cache if c is not None and hasattr(c, "offset")
        ]
        self.assertTrue(
            any(isinstance(c, QuantizedKVCache) for c in full_attention_caches),
            "quantized_kv_start=0 should have converted the backbone's "
            "full-attention cache to QuantizedKVCache",
        )

    def test_rollback_after_quantization_matches_fresh_forward(self):
        """Forces a reject (mismatched draft) right after the cache has
        already been converted to QuantizedKVCache, confirming
        rollback_speculative_cache's generic is_trimmable()/trim() path
        handles a quantized cache correctly -- not just the accept path
        exercised implicitly by the test above."""
        mx.random.seed(3)
        model = Model(tiny_args(hidden_size=128, intermediate_size=256))
        prompt = mx.random.randint(0, 64, (6,))

        quant_cache = model.make_cache()
        flags = []
        for _, _, from_draft in nemotron_h_mtp_generate_step(
            prompt,
            model,
            max_tokens=24,
            prompt_cache=quant_cache,
            kv_bits=4,
            kv_group_size=32,
            quantized_kv_start=0,
        ):
            flags.append(from_draft)
        # A draft is "rejected" whenever a non-bonus, non-first token is
        # yielded with from_draft=False -- i.e. some token besides index 0
        # and the token right after an accept is a verify-path token.
        i = 1
        saw_reject = False
        while i < len(flags):
            if flags[i]:
                i += 2  # accept always followed by one bonus token
            else:
                saw_reject = True
                i += 1
        self.assertTrue(
            saw_reject, "test setup should produce at least one rejected draft"
        )

        quantize_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=0,
            kv_group_size=32,
            kv_bits=4,
        )
        cache = model.make_cache()
        hidden = model.backbone(prompt[None], cache=cache)
        quantize_fn(cache)
        tok = mx.argmax(model.lm_head(hidden[:, -1, :]), axis=-1)
        baseline = [tok.item()]
        for _ in range(23):
            hidden = model.backbone(tok.reshape(1, 1), cache=cache)
            quantize_fn(cache)
            tok = mx.argmax(model.lm_head(hidden[:, -1, :]), axis=-1)
            baseline.append(tok.item())

        quant_cache2 = model.make_cache()
        spec = [
            tok
            for tok, _, _ in nemotron_h_mtp_generate_step(
                prompt,
                model,
                max_tokens=24,
                prompt_cache=quant_cache2,
                kv_bits=4,
                kv_group_size=32,
                quantized_kv_start=0,
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
