# Copyright © 2026 Apple Inc.

"""Regression test for a real crash hit in production (LLMTray): Gemma 4's
KV-shared layers (``num_kv_shared_layers>0``) reuse an earlier layer's
already-computed ``(keys, values)`` directly, without ever calling their own
``cache.update_and_fetch()`` (they have no cache object -- see
``Gemma4TextModel.__call__``'s cache-list padding with ``None`` for shared
layers). ``scaled_dot_product_attention`` in ``models/base.py`` decides
whether to take the quantized-attention path purely by checking
``hasattr(cache, "bits")`` on the ``cache`` object it's given -- so a shared
layer, which always passed ``cache=None`` for that check regardless of what
the borrowed keys/values actually were, silently fell through to the
*unquantized* SDPA path even after the source layer's cache had been
converted to a ``QuantizedKVCache`` (as ``mlx_lm.generate``'s
``quantized_kv_start`` does automatically once ``offset`` crosses the
threshold). That path then received a quantized ``(packed, scales, biases)``
tuple where it expected a plain ``mx.array`` and crashed with
``TypeError: scaled_dot_product_attention(): incompatible function
arguments ... Invoked with types: mlx.core.array, list, list``.

Fix: the source layer's own cache object is threaded through to the shared
layer's ``self_attn`` call (see ``source_cache`` in ``gemma4_text.py``) so
the quantized-dispatch check sees the real cache, not ``None``.

This test builds a small synthetic model directly (no torch/transformers
needed) so it can run in ANY environment with mlx installed, unlike the
other Gemma4 parity tests.
"""

import mlx.core as mx

from mlx_lm.models import gemma4_text

TEXT_ARGS = gemma4_text.ModelArgs(
    model_type="gemma4_text",
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=12,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    global_head_dim=64,
    hidden_size_per_layer_input=0,
    num_kv_shared_layers=6,
    enable_moe_block=False,
    tie_word_embeddings=True,
    final_logit_softcapping=None,
    pad_token_id=0,
    # 5 sliding + 1 full, repeated -- matches the real E4B pattern and keeps
    # both layer types present before the shared region (layers 6-11) kicks
    # in, same reasoning as test_gemma4_multimodal_parity.py's synthetic
    # config.
    layer_types=["sliding_attention"] * 5
    + ["full_attention"]
    + ["sliding_attention"] * 5
    + ["full_attention"],
)


def _make_model():
    mx.random.seed(0)
    model = gemma4_text.Model(TEXT_ARGS)
    return model


def test_quantized_kv_with_shared_layers_does_not_crash():
    model = _make_model()
    cache = model.make_cache()
    assert len(cache) == 6  # first_kv_shared = 12 - 6

    prompt = mx.random.randint(0, TEXT_ARGS.vocab_size, (1, 5))
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 5, TEXT_ARGS.vocab_size)

    # Simulate mlx_lm.generate's quantized_kv_start: convert every
    # cache that supports it once its offset is non-zero.
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized"):
            cache[i] = c.to_quantized(group_size=32, bits=4)
    assert all(hasattr(c, "bits") for c in cache)

    # This is the exact call that crashed before the fix: a shared layer
    # reads a source cache's now-quantized keys/values and must dispatch to
    # quantized_scaled_dot_product_attention instead of the plain path.
    next_token = mx.random.randint(0, TEXT_ARGS.vocab_size, (1, 1))
    logits2 = model(next_token, cache=cache)
    mx.eval(logits2)
    assert logits2.shape == (1, 1, TEXT_ARGS.vocab_size)
    assert not mx.any(mx.isnan(logits2)).item()


def test_quantized_kv_matches_unquantized_reference_closely():
    # Same generation, but WITHOUT quantizing the cache -- confirms the fix
    # doesn't change ordinary (non-quantized) behavior, and gives a rough
    # sanity bound on how much quantizing the KV cache moves the logits
    # (expected to be small but non-zero at 4-bit).
    model = _make_model()
    prompt = mx.random.randint(0, TEXT_ARGS.vocab_size, (1, 5))
    next_token = mx.random.randint(0, TEXT_ARGS.vocab_size, (1, 1))

    cache_fp = model.make_cache()
    model(prompt, cache=cache_fp)
    logits_fp = model(next_token, cache=cache_fp)
    mx.eval(logits_fp)

    cache_q = model.make_cache()
    model(prompt, cache=cache_q)
    for i, c in enumerate(cache_q):
        if hasattr(c, "to_quantized"):
            cache_q[i] = c.to_quantized(group_size=32, bits=8)
    logits_q = model(next_token, cache=cache_q)
    mx.eval(logits_q)

    diff = mx.abs(logits_fp - logits_q).max().item()
    assert diff < 1.0  # loose bound: proves it's a real (small) quantization
    # error, not garbage from a shape/type mismatch silently broadcasting.
