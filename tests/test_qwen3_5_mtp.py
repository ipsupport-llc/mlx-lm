# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.generate import (
    DraftLength,
    generate_step,
    mtp_generate_step,
    stream_generate,
)
from mlx_lm.models import qwen3_5, qwen3_5_moe
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

VOCAB = 64


def text_config(**overrides):
    config = dict(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        tie_word_embeddings=True,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )
    config.update(overrides)
    return config


def make_model(moe=False, **overrides):
    mx.random.seed(0)
    if moe:
        overrides = dict(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            **overrides,
        )
        module = qwen3_5_moe
    else:
        module = qwen3_5
    args = qwen3_5.ModelArgs(
        model_type="qwen3_5_moe" if moe else "qwen3_5",
        text_config=text_config(**overrides),
    )
    model = module.Model(args)
    mx.eval(model.parameters())
    return model


def greedy_plain(model, prompt, n):
    return [
        t for t, _ in zip((t for t, _ in generate_step(prompt, model)), range(n))
    ]


def close(a, b):
    # Two plain forwards that split the tokens differently differ by ~1e-3.
    return mx.allclose(a, b, rtol=1e-2, atol=5e-3)


def cache_state(cache):
    out = []
    for c in cache:
        if c.is_trimmable():
            out += [c.keys[..., : c.offset, :], c.values[..., : c.offset, :]]
        else:
            out += [c[0], c[1]]
    return out


class Tokenizer(TokenizerWrapper):
    """What stream_generate uses."""

    class Detokenizer:
        def __init__(self):
            self.tokens, self.text, self.last_segment = [], "", ""

        def reset(self):
            self.tokens, self.text = [], ""

        def add_token(self, t):
            self.tokens.append(t)
            self.last_segment = f"<{t}>"

        def finalize(self):
            self.last_segment = ""

    def __init__(self):
        self._detokenizer = self.Detokenizer()
        self._eos_token_ids = set()

    @property
    def detokenizer(self):
        return self._detokenizer

    @property
    def eos_token_ids(self):
        return self._eos_token_ids


class OracleHead:
    """Drafts the true next tokens (from a reference sequence), except where
    `wrong(position)` says to draft a wrong one: drives accepted, partially
    accepted and rejected iterations on a random model whose own head would
    almost never be right."""

    def __init__(self, model, sequence, wrong):
        self.lm = model.language_model
        self.sequence = sequence
        self.wrong = wrong
        self.real_step = self.lm.mtp_step
        self.real_backbone = self.lm.backbone
        self.chain = 0
        self.cache = None

    def backbone(self, inputs, cache=None, ssm_sink=None):
        self.chain = 0
        self.cache = cache
        return self.real_backbone(inputs, cache=cache, ssm_sink=ssm_sink)

    def mtp_step(self, hidden, tokens, mtp_cache):
        logits, h = self.real_step(hidden, tokens, mtp_cache)
        # The draft loop's first call follows the token at the cache's end.
        offset = next(c.offset for c in self.cache if c.is_trimmable())
        position = offset + 1 + self.chain
        self.chain += 1
        if position >= len(self.sequence):
            return logits, h
        token = self.sequence[position]
        if self.wrong(position):
            token = (token + 1) % VOCAB
        forced = mx.full(logits.shape, -1e4).at[..., token].add(2e4)
        return forced, h

    def install(self):
        self.lm.backbone = self.backbone
        self.lm.mtp_step = self.mtp_step


class TestQwen35MTP(unittest.TestCase):
    def test_head_loads_only_with_its_weights(self):
        model = make_model()
        self.assertIsNotNone(model.mtp)
        weights = dict(tree_flatten(model.parameters()))
        self.assertTrue(any(".mtp." in k for k in weights))

        stripped = {k: v for k, v in weights.items() if ".mtp." not in k}
        fresh = make_model()
        fresh.load_weights(list(fresh.sanitize(stripped).items()))
        self.assertIsNone(fresh.mtp)

        again = make_model()
        again.load_weights(list(again.sanitize(dict(weights)).items()))
        self.assertIsNotNone(again.mtp)

    def test_moe_head_experts_are_stacked(self):
        model = make_model(moe=True)
        weights = dict(tree_flatten(model.parameters()))
        raw = {}
        for k, v in weights.items():
            if ".mtp." in k and ".switch_mlp." in k:
                prefix, rest = k.split(".switch_mlp.")
                for e in range(v.shape[0]):
                    raw[f"{prefix}.experts.{e}.{rest}"] = v[e]
            else:
                raw[k] = v
        out = model.sanitize(raw)
        for k, v in weights.items():
            self.assertTrue(mx.array_equal(out[k], v), k)

    def test_rollback_matches_a_fresh_forward(self):
        model = make_model()
        lm = model.language_model
        prompt = mx.random.randint(0, VOCAB, (7,))
        block = mx.random.randint(0, VOCAB, (1, 4))
        for keep in range(5):
            cache = model.make_cache()
            lm.backbone(prompt[None], cache=cache)
            sink = []
            lm.backbone(block, cache=cache, ssm_sink=sink)
            lm.rollback_speculative_cache(cache, sink, keep, 4)

            ref = model.make_cache()
            lm.backbone(prompt[None], cache=ref)
            if keep:
                lm.backbone(block[:, :keep], cache=ref)
            for a, b in zip(cache_state(cache), cache_state(ref)):
                self.assertTrue(close(a, b), f"keep={keep}")

    def check_matches_plain(self, moe):
        model = make_model(moe=moe)
        prompt = mx.random.randint(0, VOCAB, (9,))
        n = 20
        plain = greedy_plain(model, prompt, n)
        sequence = prompt.tolist() + plain
        for k in (1, 2, 3):
            for wrong in (lambda p: False, lambda p: p % 3 == 0, lambda p: True):
                m = make_model(moe=moe)
                OracleHead(m, sequence, wrong).install()
                out = [
                    t
                    for t, _, _ in mtp_generate_step(
                        prompt, m, num_draft_tokens=k, max_tokens=n
                    )
                ]
                self.assertEqual(out, plain, f"k={k}")

    def test_matches_plain_greedy_decoding(self):
        self.check_matches_plain(moe=False)

    def test_matches_plain_greedy_decoding_moe(self):
        self.check_matches_plain(moe=True)

    def test_own_head_matches_plain_greedy_decoding(self):
        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (9,))
        plain = greedy_plain(model, prompt, 16)
        for k in (1, 3):
            out = [
                t
                for t, _, _ in mtp_generate_step(
                    prompt, model, num_draft_tokens=k, max_tokens=16
                )
            ]
            self.assertEqual(out, plain)

    def test_cache_holds_the_emitted_tokens_wherever_generation_stops(self):
        """The server stores the cache keyed by prompt + emitted tokens: it
        must hold exactly those, whether generation stopped at an accepted
        draft or at the backbone's own token."""
        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (9,))
        plain = greedy_plain(model, prompt, 16)
        sequence = prompt.tolist() + plain
        for n in range(1, 14):
            m = make_model()
            OracleHead(m, sequence, lambda p: p % 4 == 0).install()
            cache = m.make_cache()
            out = [
                t
                for t, _, _ in mtp_generate_step(
                    prompt, m, num_draft_tokens=3, max_tokens=n, prompt_cache=cache
                )
            ]
            self.assertEqual(out, plain[:n])
            ref = model.make_cache()
            model.language_model.backbone(mx.array(sequence[: len(prompt) + n])[None], cache=ref)
            for a, b in zip(cache_state(cache), cache_state(ref)):
                self.assertTrue(close(a, b), f"n={n}")

    def test_chunked_prefill_matches_one_shot(self):
        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (23,))
        plain = greedy_plain(model, prompt, 8)
        out = [
            t
            for t, _, _ in mtp_generate_step(
                prompt, model, max_tokens=8, prefill_step_size=5
            )
        ]
        self.assertEqual(out, plain)

    def test_quantized_kv_cache(self):
        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (9,))
        plain = [
            t
            for t, _ in zip(
                (
                    t
                    for t, _ in generate_step(
                        prompt, model, kv_bits=8, kv_group_size=32, quantized_kv_start=0
                    )
                ),
                range(12),
            )
        ]
        sequence = prompt.tolist() + plain
        m = make_model()
        oracle = OracleHead(m, sequence, lambda p: p % 3 == 0)
        oracle.install()
        caches = []
        real_make = m.language_model.make_mtp_cache
        m.language_model.make_mtp_cache = lambda: caches.append(real_make()) or caches[-1]
        out = [
            t
            for t, _, _ in mtp_generate_step(
                prompt,
                m,
                num_draft_tokens=2,
                max_tokens=12,
                kv_bits=8,
                kv_group_size=32,
                quantized_kv_start=0,
            )
        ]
        self.assertEqual(out, plain)
        # The head's cache is quantized too.
        self.assertTrue(all(hasattr(c, "bits") for c in caches[0]))

    def test_sampling_with_temperature(self):
        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (9,))
        sampler = make_sampler(temp=1.0, top_p=0.95)
        mx.random.seed(3)
        out = list(
            mtp_generate_step(
                prompt, model, num_draft_tokens=2, max_tokens=12, sampler=sampler
            )
        )
        self.assertEqual(len(out), 12)
        for _, logprobs, _ in out:
            self.assertEqual(logprobs.shape, (VOCAB,))

    def test_stream_generate_uses_the_head(self):
        """stream_generate dispatches to the head (as the server calls it),
        with the same tokens."""
        import sys

        gen = sys.modules["mlx_lm.generate"]

        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (9,))
        plain = greedy_plain(model, prompt, 10)
        calls = []
        real = gen.mtp_generate_step

        def spy(*args, **kwargs):
            calls.append(kwargs["num_draft_tokens"])
            return real(*args, **kwargs)

        gen.mtp_generate_step = spy
        try:
            out = [
                r.token
                for r in stream_generate(
                    model,
                    Tokenizer(),
                    prompt,
                    max_tokens=10,
                    num_draft_tokens=2,
                    input_embeddings=None,
                )
            ]
        finally:
            gen.mtp_generate_step = real
        self.assertEqual(out, plain)
        self.assertEqual(calls, [2])

    def test_adaptive_matches_plain_greedy_decoding(self):
        model = make_model()
        prompt = mx.random.randint(0, VOCAB, (9,))
        plain = greedy_plain(model, prompt, 40)
        sequence = prompt.tolist() + plain
        for wrong in (lambda p: False, lambda p: p % 2 == 0, lambda p: True):
            m = make_model()
            OracleHead(m, sequence, wrong).install()
            cache = m.make_cache()
            out = [
                t
                for t, _, _ in mtp_generate_step(
                    prompt, m, num_draft_tokens=3, max_tokens=40, prompt_cache=cache
                )
            ]
            self.assertEqual(out, plain)
            ref = model.make_cache()
            model.language_model.backbone(mx.array(sequence[: len(prompt) + 40])[None], cache=ref)
            for a, b in zip(cache_state(cache), cache_state(ref)):
                self.assertTrue(close(a, b))


class TestDraftLength(unittest.TestCase):
    def feed(self, lengths, p, seconds, n=200):
        # Each draft accepted with probability p (deterministic pattern).
        for i in range(n):
            k = lengths.choose()
            n_acc = 0
            while n_acc < k and ((i * 7 + n_acc * 3) % 100) < p * 100:
                n_acc += 1
            lengths.update(k, n_acc, seconds(k))

    def test_good_drafts_use_the_longest(self):
        lengths = DraftLength(3)
        self.feed(lengths, 0.9, lambda k: 1.0 + 0.1 * k)
        self.assertEqual(lengths.best(), 3)

    def test_bad_drafts_fall_back_to_plain_decoding(self):
        lengths = DraftLength(3)
        self.feed(lengths, 0.1, lambda k: 1.0 + 0.3 * k)
        self.assertEqual(lengths.best(), 0)

    def test_probes_keep_measuring_after_falling_back(self):
        lengths = DraftLength(3, probe_every=4)
        self.feed(lengths, 0.0, lambda k: 1.0 + 0.3 * k, n=40)
        self.assertEqual(lengths.best(), 0)
        chosen = [lengths.choose() for _ in range(8)]
        self.assertIn(1, chosen)

    def test_no_drafts(self):
        self.assertEqual(DraftLength(0).choose(), 0)


if __name__ == "__main__":
    unittest.main()
