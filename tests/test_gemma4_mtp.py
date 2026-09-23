# Copyright © 2026 Apple Inc.

"""Gemma 4 MTP drafter (gemma4_assistant) speculative decoding.

Tiny random float32 models, no weights needed. Parity of the drafter module
itself against transformers' Gemma4AssistantForCausalLM was checked
separately on the real google/gemma-4-26B-A4B-it-assistant weights (fp32,
synthetic main-model KV: max logit diff 0.012 on a logit scale of 33, same
argmax).
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import (
    _gemma4_trim_cache,
    gemma4_mtp_generate_step,
    generate_step,
)
from mlx_lm.models import gemma4_assistant, gemma4_text
from mlx_lm.models.cache import make_prompt_cache

VOCAB = 97
WINDOW = 8


def _main_model():
    args = gemma4_text.ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=WINDOW,
        attention_k_eq_v=True,
        use_double_wide_mlp=False,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    mx.random.seed(0)
    model = gemma4_text.Model(args)
    # Random init gives near-uniform logits; sharpen so greedy ties are
    # vanishingly unlikely and batched-vs-single numerics can't flip a pick.
    model.model.embed_tokens.weight = model.model.embed_tokens.weight * 8
    mx.eval(model.parameters())
    return model


def _drafter():
    args = gemma4_assistant.ModelArgs(
        backbone_hidden_size=64,
        text_config=dict(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            head_dim=16,
            global_head_dim=32,
            vocab_size=VOCAB,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            num_kv_shared_layers=2,
            hidden_size_per_layer_input=0,
            sliding_window=WINDOW,
            attention_k_eq_v=True,
            use_double_wide_mlp=False,
            final_logit_softcapping=None,
            layer_types=["sliding_attention", "full_attention"],
        ),
    )
    mx.random.seed(1)
    d = gemma4_assistant.Model(args)
    mx.eval(d.parameters())
    return d


class OracleDrafter(nn.Module):
    """Proposes the true greedy continuation, except every `wrong_every`-th
    proposal, so accept / partial-accept / reject all get exercised."""

    model_type = "gemma4_assistant"
    sliding_window = WINDOW

    def __init__(self, reference, prompt_len, wrong_every):
        super().__init__()
        self.reference = reference
        self.prompt_len = prompt_len
        self.wrong_every = wrong_every
        self._pos = None
        self._j = 0
        self.calls = 0

    def __call__(self, inputs_embeds, shared_kv, position):
        # Shapes the real drafter would see.
        k, v = shared_kv["sliding_attention"]
        assert k.shape[2] <= WINDOW and k.shape == v.shape
        k, v = shared_kv["full_attention"]
        assert k.shape[2] == position, (k.shape, position)
        if position != self._pos:
            self._pos, self._j = position, 0
        idx = position - self.prompt_len + 1 + self._j
        self._j += 1
        self.calls += 1
        tok = self.reference[idx] if idx < len(self.reference) else 0
        if self.calls % self.wrong_every == 0:
            tok = (tok + 1) % VOCAB
        logits = mx.zeros((1, 1, VOCAB)).at[0, 0, tok].add(1.0)
        return logits, mx.zeros((1, 1, 64))


class TestGemma4MTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _main_model()
        cls.prompt = mx.array([(i * 7 + 3) % VOCAB for i in range(19)])  # > window
        cls.ref = [
            t for t, _ in generate_step(cls.prompt, cls.model, max_tokens=60)
        ]

    def _mtp(self, drafter, k, **kw):
        return [
            (t, d)
            for t, _, d in gemma4_mtp_generate_step(
                self.prompt,
                self.model,
                drafter,
                num_draft_tokens=k,
                max_tokens=60,
                **kw,
            )
        ]

    def test_oracle_drafter_matches_plain_greedy(self):
        for k in (1, 2, 3, 5):
            for wrong_every in (2, 3, 7):
                oracle = OracleDrafter(self.ref, len(self.prompt), wrong_every)
                out = self._mtp(oracle, k)
                self.assertEqual([t for t, _ in out], self.ref, (k, wrong_every))
                n_draft = sum(d for _, d in out)
                self.assertGreater(n_draft, 0)
                self.assertLess(n_draft, len(out))

    def test_small_prefill_steps(self):
        oracle = OracleDrafter(self.ref, len(self.prompt), 3)
        out = self._mtp(oracle, 3, prefill_step_size=4)
        self.assertEqual([t for t, _ in out], self.ref)

    def test_real_drafter_module_matches_plain_greedy(self):
        out = self._mtp(_drafter(), 3)
        self.assertEqual([t for t, _ in out], self.ref)

    def test_cache_matches_emitted_tokens_wherever_generation_stops(self):
        # The server stores the cache keyed by prompt + emitted tokens and
        # trims / extends it for the next request, so however the consumer
        # stops (mid-accepted-drafts or at the main model's own token), the
        # cache must hold exactly those tokens -- as plain decoding leaves it.
        for stop_after in range(1, 25):
            oracle = OracleDrafter(self.ref, len(self.prompt), 4)
            cache = make_prompt_cache(self.model)
            gen = gemma4_mtp_generate_step(
                self.prompt,
                self.model,
                oracle,
                num_draft_tokens=3,
                max_tokens=60,
                prompt_cache=cache,
            )
            out = [next(gen)[0] for _ in range(stop_after)]
            gen.close()
            self.assertEqual(out, self.ref[:stop_after])
            n = len(self.prompt) + stop_after
            self.assertEqual([c.offset for c in cache], [n] * len(cache))
            # Continuing from that cache == a fresh forward over everything.
            nxt = mx.array([[self.ref[stop_after]]])
            got = self.model(nxt, cache=cache)[0, -1]
            full = mx.concatenate([self.prompt, mx.array(self.ref[: stop_after + 1])])
            want = self.model(full[None])[0, -1]
            self.assertTrue(mx.allclose(got, want, atol=0.05).item(), stop_after)

    def test_rotating_cache_rollback_past_window(self):
        # Verify-then-trim must leave the caches exactly as if only the kept
        # tokens had been processed, also after the sliding caches wrapped.
        tokens = mx.array([(i * 5 + 1) % VOCAB for i in range(3 * WINDOW)])
        a = make_prompt_cache(self.model)
        b = make_prompt_cache(self.model)
        self.model(tokens[None, :20], cache=a)
        self.model(tokens[None, :20], cache=b)
        self.model(tokens[None, 20:24], cache=a)  # verify block of 4
        _gemma4_trim_cache(a, 2)  # keep 2
        self.model(tokens[None, 20:22], cache=b)
        nxt = tokens[None, 22:23]
        la = self.model(nxt, cache=a)
        lb = self.model(nxt, cache=b)
        self.assertTrue(mx.allclose(la, lb, atol=1e-5).item())
        self.assertEqual([c.offset for c in a], [c.offset for c in b])


if __name__ == "__main__":
    unittest.main()
