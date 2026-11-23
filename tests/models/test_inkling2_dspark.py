# Copyright © 2026 IPSupport LLC

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.generate import dspark_generate_step, generate_step
from mlx_lm.models.cache import make_prompt_cache

from .test_inkling2 import LOOP_CONFIG, create_model

# 8 layers, layers 3-5 looped 3 times: 14 layer visits, 14 is the output.
MTP_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 96,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 8,
    "block_size": 5,
    "markov_rank": 8,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [2, 6, 11, 14],
}
CONFIG = {**LOOP_CONFIG, "mtp_config": MTP_CONFIG}


def plain(model, prompt, n):
    out = []
    for tok, _ in generate_step(prompt, model, max_tokens=n):
        out.append(tok)
    return out


class TestInkling2DSpark(unittest.TestCase):
    def setUp(self):
        self.model = create_model(config=CONFIG)
        self.prompt = mx.random.randint(0, 120, (20,), key=mx.random.key(1))

    def test_features(self):
        cache = make_prompt_cache(self.model)
        logits, features = self.model.forward_features(self.prompt[None], cache)
        self.assertEqual(features.shape, (1, 20, 4, 64))
        hidden = self.model.model.language_model(self.prompt[None])
        self.assertTrue(mx.allclose(features[:, :, -1], hidden, atol=1e-5))
        self.assertTrue(
            mx.allclose(logits[0, -1], self.model(self.prompt[None])[0, -1], atol=1e-4)
        )

    def test_greedy_matches_plain(self):
        want = plain(self.model, self.prompt, 30)
        got = [
            t
            for t, _, _ in dspark_generate_step(
                self.prompt, self.model, max_tokens=30, adaptive=False
            )
        ]
        self.assertEqual(got, want)

    def test_partial_acceptance(self):
        want = plain(self.model, self.prompt, 40)
        calls = []

        def oracle(token, position, last_features, draft_cache, k):
            # The true continuation, wrong from a varying position on.
            start = position - len(self.prompt) + 1
            drafts = want[start : start + k] + [0] * k
            drafts = drafts[:k]
            wrong = len(calls) % (k + 1)
            if wrong < k:
                drafts[wrong] = (drafts[wrong] + 1) % 120
            calls.append(wrong)
            return mx.array([drafts])

        self.model.draft = oracle
        got, from_draft = [], 0
        for t, _, d in dspark_generate_step(
            self.prompt, self.model, max_tokens=40, adaptive=False
        ):
            got.append(t)
            from_draft += d
        self.assertEqual(got, want)
        self.assertGreater(from_draft, 10)

    def test_cache_after_early_stop(self):
        want = plain(self.model, self.prompt, 40)

        def oracle(token, position, last_features, draft_cache, k):
            start = position - len(self.prompt) + 1
            return mx.array([(want[start : start + k] + [0] * k)[:k]])

        self.model.draft = oracle
        for n in (5, 9, 14):
            cache = make_prompt_cache(self.model)
            got = [
                t
                for t, _, _ in dspark_generate_step(
                    self.prompt,
                    self.model,
                    max_tokens=n,
                    prompt_cache=cache,
                    adaptive=False,
                )
            ]
            self.assertEqual(got, want[:n])
            # The cache holds prompt + yielded tokens: the next step is plain's.
            nxt = self.model(mx.array([[want[n]]]), cache=cache)
            full = mx.concatenate([self.prompt, mx.array(want[: n + 1])])
            self.assertTrue(
                mx.allclose(nxt[0, -1], self.model(full[None])[0, -1], atol=1e-3),
                f"stop after {n}",
            )

    def test_sanitize_drops_or_keeps_draft(self):
        weights = dict(tree_flatten(self.model.parameters()))
        draft = {k: v for k, v in weights.items() if k.startswith("dspark.")}
        raw = {"mtp." + k.split(".", 1)[1]: v for k, v in draft.items()}
        model = create_model(config=CONFIG)
        out = model.sanitize(raw)
        self.assertEqual(set(out), set(draft))
        model = create_model(config=CONFIG)
        model.sanitize({})
        self.assertNotIn("dspark", model)


if __name__ == "__main__":
    unittest.main()
