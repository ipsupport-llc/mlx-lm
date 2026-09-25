# Copyright © 2026 Apple Inc.

"""A classic (non-Gemma-assistant) draft model through stream_generate the
way mlx_lm.server calls it: always with input_embeddings (None for text).
That used to raise TypeError in speculative_generate_step for every request."""

import types
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models import llama
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper


def _tiny_llama(seed):
    mx.random.seed(seed)
    args = llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=64,
        rope_theta=10000.0, tie_word_embeddings=True,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


class _Tokenizer(TokenizerWrapper):
    """The bits stream_generate uses (a TokenizerWrapper, so it isn't
    wrapped again)."""

    class _Detok:
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
        self._detokenizer = self._Detok()
        self._eos_token_ids = set()

    @property
    def detokenizer(self):
        return self._detokenizer

    @property
    def eos_token_ids(self):
        return self._eos_token_ids


class TestClassicDraftServerPath(unittest.TestCase):
    def test_text_request_with_input_embeddings_none(self):
        model, draft = _tiny_llama(0), _tiny_llama(1)
        prompt = mx.array([1, 5, 9, 3])
        ref = [t for t, _ in generate_step(prompt, model, max_tokens=12)]
        out = [
            r.token
            for r in stream_generate(
                model, _Tokenizer(), prompt, max_tokens=12, draft_model=draft,
                input_embeddings=None,
            )
        ]
        self.assertEqual(out, ref)

    def test_image_style_request_falls_back_without_the_drafter(self):
        model, draft = _tiny_llama(0), _tiny_llama(1)
        prompt = mx.array([1, 5, 9, 3])
        emb = model.model.embed_tokens(prompt)
        ref = [t for t, _ in generate_step(prompt, model, max_tokens=8, input_embeddings=emb)]
        out = [
            r.token
            for r in stream_generate(
                model, _Tokenizer(), prompt, max_tokens=8, draft_model=draft,
                input_embeddings=emb,
            )
        ]
        self.assertEqual(out, ref)

    def test_cache_holds_prompt_and_yielded_tokens(self):
        # The server stores the cache under prompt + yielded tokens and
        # reuses it for the next request: it was a token short.
        model, draft = _tiny_llama(0), _tiny_llama(1)
        prompt = mx.array([1, 5, 9, 3])
        follow = mx.array([7, 2, 11])
        for max_tokens, stop_after in [(1, None), (2, None), (5, None), (9, None), (20, 3), (20, 6)]:
            with self.subTest(max_tokens=max_tokens, stop_after=stop_after):
                cache = make_prompt_cache(model) + make_prompt_cache(draft)
                gen = stream_generate(
                    model, _Tokenizer(), prompt, max_tokens=max_tokens, draft_model=draft,
                    num_draft_tokens=3, prompt_cache=cache, input_embeddings=None,
                )
                out = []
                for r in gen:
                    out.append(r.token)
                    if len(out) == stop_after:
                        break
                gen.close()
                n = len(model.layers)
                self.assertEqual([cache[0].offset, cache[n].offset], [prompt.size + len(out)] * 2)
                # The reused caches continue like a fresh run over everything.
                seen = mx.concatenate([prompt, mx.array(out, mx.uint32), follow])
                for m, c in ((model, cache[:n]), (draft, cache[n:])):
                    reused = [t for t, _ in generate_step(follow, m, max_tokens=4, prompt_cache=c)]
                    fresh = [t for t, _ in generate_step(seen, m, max_tokens=4)]
                    self.assertEqual(reused, fresh)


class TestServerDraftSlots(unittest.TestCase):
    """An image request runs without the draft model and caches no draft
    slots: a text request reusing that prefix crashed (draft cache [])."""

    def _generator(self, draft):
        from mlx_lm.server import ResponseGenerator

        gen = object.__new__(ResponseGenerator)
        gen.model_provider = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[0, 0]), draft_model=draft,
        )
        return gen

    def test_slots_match_the_request(self):
        gen = self._generator(object())
        main, both = ["m1", "m2"], ["m1", "m2", "d1", "d2"]
        key, rest = [1, 2, 3, 4], [4]
        self.assertEqual(gen._match_draft_slots(main, rest, key, True), (None, key))
        self.assertEqual(gen._match_draft_slots(both, rest, key, True), (both, rest))
        self.assertEqual(gen._match_draft_slots(both, rest, key, False), (main, rest))
        self.assertEqual(gen._match_draft_slots(main, rest, key, False), (main, rest))
        self.assertEqual(gen._match_draft_slots(None, key, key, True), (None, key))

    def test_no_draft_model_unchanged(self):
        gen = self._generator(None)
        self.assertEqual(gen._match_draft_slots(["m1", "m2"], [4], [1, 4], False), (["m1", "m2"], [4]))


if __name__ == "__main__":
    unittest.main()
