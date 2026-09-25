# Copyright © 2026 Apple Inc.

"""A classic (non-Gemma-assistant) draft model through stream_generate the
way mlx_lm.server calls it: always with input_embeddings (None for text).
That used to raise TypeError in speculative_generate_step for every request."""

import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models import llama
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


if __name__ == "__main__":
    unittest.main()
