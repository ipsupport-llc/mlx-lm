# Copyright © 2026 Apple Inc.

import types
import unittest

import mlx.core as mx

from mlx_lm.generate import (
    MIN_PREFILL_STEP,
    adaptive_prefill_step,
    attention_heads,
    gemma4_mtp_generate_step,
    generate_step,
    nemotron_h_mtp_generate_step,
)
from mlx_lm.models import llama, nemotron_h

MB = 1 << 20


class TestAdaptivePrefillStep(unittest.TestCase):
    def test_no_budget_keeps_the_step(self):
        for budget in (None, 0):
            self.assertEqual(adaptive_prefill_step(2048, 100_000, 16, budget), 2048)

    def test_short_prompt_keeps_the_step(self):
        self.assertEqual(adaptive_prefill_step(2048, 0, 16, 512 * MB), 2048)

    def test_scores_fit_the_budget(self):
        # Gemma 4 26B (16 heads) that ran out of memory at 28672 of 59720.
        for offset in (8192, 28672, 59720, 200_000):
            step = adaptive_prefill_step(2048, offset, 16, 512 * MB)
            self.assertLess(step, 2048)
            if step > MIN_PREFILL_STEP:
                self.assertLessEqual(16 * 2 * step * (offset + step), 512 * MB)
                # The largest such step.
                self.assertGreater(16 * 2 * (step + 1) * (offset + step + 1), 512 * MB)

    def test_step_shrinks_as_the_cache_grows(self):
        steps = [
            adaptive_prefill_step(2048, o, 16, 512 * MB)
            for o in range(0, 120_000, 4096)
        ]
        self.assertEqual(steps, sorted(steps, reverse=True))

    def test_bounds(self):
        self.assertEqual(adaptive_prefill_step(2048, 10**7, 64, MB), MIN_PREFILL_STEP)
        # Never above the configured step, even when that is below the minimum.
        self.assertEqual(adaptive_prefill_step(64, 10**7, 64, MB), 64)
        self.assertEqual(adaptive_prefill_step(512, 0, 1, 1 << 40), 512)


class TestAttentionHeads(unittest.TestCase):
    def test_model_args(self):
        model = types.SimpleNamespace(
            args=types.SimpleNamespace(num_attention_heads=16)
        )
        self.assertEqual(attention_heads(model), 16)

    def test_language_model_and_text_config(self):
        wrapper = types.SimpleNamespace(
            args=types.SimpleNamespace(text_config={"num_attention_heads": 8}),
        )
        self.assertEqual(attention_heads(wrapper), 8)
        nested = types.SimpleNamespace(
            args=types.SimpleNamespace(),
            language_model=types.SimpleNamespace(
                args=types.SimpleNamespace(num_attention_heads=12)
            ),
        )
        self.assertEqual(attention_heads(nested), 12)

    def test_unknown(self):
        self.assertEqual(attention_heads(types.SimpleNamespace()), 32)


class TestGenerateStepBudget(unittest.TestCase):
    def test_chunks_follow_the_budget(self):
        mx.random.seed(0)
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
        model = llama.Model(args)
        mx.eval(model.parameters())
        prompt = mx.random.randint(0, 64, (1200,))
        budget = 4 * 2 * 256 * 512

        def run(budget):
            seen = []
            tokens = [
                t
                for t, _ in generate_step(
                    prompt,
                    model,
                    max_tokens=8,
                    prefill_step_size=512,
                    prompt_progress_callback=lambda p, n: seen.append((p, n)),
                    prefill_memory_budget=budget,
                )
            ]
            return tokens, seen

        ref, _ = run(None)
        out, seen = run(budget)
        self.assertEqual(out, ref)
        self.assertEqual(seen[0], (0, 1200))
        self.assertEqual(seen[-1], (1200, 1200))
        done = [p for p, _ in seen[:-1]]
        steps = [b - a for a, b in zip(done, done[1:])]
        expected = [adaptive_prefill_step(512, o, 4, budget) for o in done[:-1]]
        self.assertEqual(steps[:-1], expected[:-1])
        self.assertLess(min(steps[:-1]), 512)


def _steps(progress):
    done = [p for p, _ in progress]
    return [b - a for a, b in zip(done, done[1:]) if b > a]


class TestMTPStepBudget(unittest.TestCase):
    """The MTP prefill loops shrink the step too; tokens don't change."""

    BUDGET = 4 * 2 * 256 * 512

    def test_nemotron_h(self):
        from test_nemotron_h_mtp_generate import tiny_args

        mx.random.seed(0)
        model = nemotron_h.Model(tiny_args(max_position_embeddings=4096))
        prompt = mx.random.randint(0, 64, (900,))

        def run(budget):
            seen = []
            gen = nemotron_h_mtp_generate_step(
                prompt,
                model,
                max_tokens=8,
                prefill_step_size=512,
                prompt_progress_callback=lambda p, n: seen.append((p, n)),
                prefill_memory_budget=budget,
            )
            return [t for t, _, _ in gen], seen

        ref, _ = run(None)
        out, seen = run(self.BUDGET)
        self.assertEqual(out, ref)
        self.assertEqual(seen[-1], (900, 900))
        self.assertLess(max(_steps(seen)), 512)

    def test_gemma4(self):
        from test_gemma4_mtp import VOCAB, _drafter, _main_model

        model = _main_model()
        prompt = mx.array([(i * 7 + 3) % VOCAB for i in range(900)])
        ref = [t for t, _ in generate_step(prompt, model, max_tokens=12)]
        seen = []
        out = [
            t
            for t, _, _ in gemma4_mtp_generate_step(
                prompt,
                model,
                _drafter(),
                max_tokens=12,
                prefill_step_size=512,
                prompt_progress_callback=lambda p, n: seen.append((p, n)),
                prefill_memory_budget=self.BUDGET,
            )
        ]
        self.assertEqual(out, ref)
        self.assertEqual(seen[-1], (900, 900))
        self.assertLess(max(_steps(seen)), 512)


if __name__ == "__main__":
    unittest.main()
