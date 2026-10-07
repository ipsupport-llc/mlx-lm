# Copyright © 2026 Apple Inc.

"""--warm-next-turn: after a chat request, the server prefills the
conversation with the answer, as the client will send it back, while idle,
so a model whose cache can't be trimmed (here a tiny Qwen3.5: delta-rule
layers) reuses it on the next request."""

import json
import time
import types
import unittest

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.server import CompletionRequest, ResponseGenerator

TOKENIZER = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


def tiny_hybrid(vocab):
    mx.random.seed(0)
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6, vocab_size=vocab,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, full_attention_interval=2, tie_word_embeddings=True,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1], "rope_theta": 10000,
                         "partial_rotary_factor": 0.5}))
    model = qwen3_5.Model(args)
    mx.eval(model.parameters())
    return model


class Provider:
    def __init__(self):
        _, self.tokenizer = load(TOKENIZER)
        self.model = tiny_hybrid(len(self.tokenizer.vocab) + 1000)
        self.model_key = ("tiny-hybrid", None)
        self.draft_model = None
        self.cli_args = types.SimpleNamespace(
            chat_template_args={}, prefill_step_size=16, kv_bits=None, kv_group_size=64,
            quantized_kv_start=0, prompt_cache_bytes=None, warm_next_turn=True,
            warm_next_turn_max_tokens=16384, max_kv_size=None)

    def load_default(self):
        pass

    def reset(self):
        pass


def state(cache):
    out = []
    for c in cache:
        if c.is_trimmable():
            out += [c.keys[..., : c.offset, :], c.values[..., : c.offset, :]]
        else:
            out += [c[0], c[1]]
    return out


class TestWarmNextTurn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = Provider()
        cls.rg = ResponseGenerator(cls.provider, LRUPromptCache())

    @classmethod
    def tearDownClass(cls):
        cls.rg.stop_and_join()

    def args(self):
        return types.SimpleNamespace(chat_template_kwargs=None)

    def test_next_turn_tokens_start_the_next_request(self):
        tok = self.provider.tokenizer
        messages = [{"role": "user", "content": "hi"}]
        request = CompletionRequest("chat", "", messages, None, None)
        answer = {"role": "assistant", "content": "Hello there"}
        warm = self.rg._next_turn_tokens(tok, request, self.args(), answer)
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        nxt = tok.apply_chat_template(messages + [answer, {"role": "user", "content": "and you?"}],
                                      add_generation_prompt=True)
        self.assertEqual(list(nxt[: len(warm)]), warm)
        self.assertGreater(len(warm), len(prompt))

    def test_warmed_entry_is_the_conversation_so_far(self):
        tok, model = self.provider.tokenizer, self.provider.model
        messages = [{"role": "user", "content": "tell me a story"}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        # The checkpoint a request leaves at its prompt's end.
        cache = make_prompt_cache(model)
        model(mx.array(prompt)[None], cache=cache)
        mx.eval([c.state for c in cache])
        self.rg.prompt_cache.insert_cache(self.provider.model_key, list(prompt), cache, cache_type="user")

        request = CompletionRequest("chat", "", json.loads(json.dumps(messages)), None, None)
        answer = "Once upon a time there was a small model."
        self.rg.warm_next_turn(self.provider.model_key, request, self.args(), answer, [])
        warm = self.rg._next_turn_tokens(tok, request, self.args(), {"role": "assistant", "content": answer})

        deadline = time.time() + 30
        while time.time() < deadline:
            got, rest = self.rg.prompt_cache.fetch_nearest_cache(self.provider.model_key, warm + [0])
            if got is not None and len(rest) == 1:
                break
            time.sleep(0.2)
        self.assertEqual(len(rest), 1, "no warmed entry for the next turn")

        ref = make_prompt_cache(model)
        model(mx.array(warm)[None], cache=ref)
        for a, b in zip(state(got), state(ref)):
            self.assertTrue(mx.allclose(a, b, rtol=1e-2, atol=5e-3))


    def test_a_failing_warm_leaves_the_server_running(self):
        tok, model = self.provider.tokenizer, self.provider.model
        messages = [{"role": "user", "content": "a failing warm"}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        cache = make_prompt_cache(model)
        model(mx.array(prompt)[None], cache=cache)
        mx.eval([c.state for c in cache])
        self.rg.prompt_cache.insert_cache(self.provider.model_key, list(prompt), cache, cache_type="user")
        request = CompletionRequest("chat", "", messages, None, None)
        job = (self.provider.model_key, request, self.args(), {"role": "assistant", "content": "a long enough answer"})

        class Broken:
            def __getattr__(self, name):
                return getattr(model, name)

            def __call__(self, *a, **k):
                raise RuntimeError("out of memory")

        self.provider.model = Broken()
        try:
            with self.assertLogs(level="WARNING") as logs:
                self.rg._warm_next(job, mx.default_stream(mx.default_device()))   # no exception
        finally:
            self.provider.model = model
        self.assertIn("out of memory", "\n".join(logs.output))

    def test_a_job_for_another_model_is_dropped(self):
        tok = self.provider.tokenizer
        messages = [{"role": "user", "content": "another model"}]
        request = CompletionRequest("chat", "", messages, None, None)
        answer = {"role": "assistant", "content": "ok"}
        warm = self.rg._next_turn_tokens(tok, request, self.args(), answer)
        self.rg._warm_next((("other", None), request, self.args(), answer), mx.default_stream(mx.default_device()))
        got, rest = self.rg.prompt_cache.fetch_nearest_cache(self.provider.model_key, warm + [0])
        self.assertTrue(got is None or len(rest) > 1)


if __name__ == "__main__":
    unittest.main()
