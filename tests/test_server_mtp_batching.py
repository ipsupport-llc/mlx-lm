# Copyright © 2026 Apple Inc.

"""Which requests a Nemotron-H model with an MTP head keeps out of batching:
only those stream_generate would serve with MTP (greedy, no logits
processors). A sampled or penalized request never uses MTP and must batch
like on any other model (it used to be serialized)."""

import types
import unittest

from mlx_lm.server import LogitsProcessorArguments, ResponseGenerator, SamplingArguments


def _args(temperature=0.0, repetition_penalty=0.0, logit_bias=None):
    return types.SimpleNamespace(
        seed=None,
        sampling=SamplingArguments(
            temperature=temperature, top_p=1.0, top_k=0, min_p=0.0,
            xtc_probability=0.0, xtc_threshold=0.0,
        ),
        logits=LogitsProcessorArguments(
            logit_bias=logit_bias, repetition_penalty=repetition_penalty, repetition_context_size=20,
            presence_penalty=0.0, presence_context_size=20, frequency_penalty=0.0, frequency_context_size=20,
        ),
    )


def _generator(supports_mtp):
    gen = object.__new__(ResponseGenerator)
    gen.model_provider = types.SimpleNamespace(
        is_batchable=True, supports_mtp=supports_mtp, tokenizer=types.SimpleNamespace(encode=lambda s: [0], eos_token_ids=[]),
        cli_args=types.SimpleNamespace(kv_bits=None),
    )
    return gen


class TestMTPBatching(unittest.TestCase):
    def test_only_mtp_requests_stay_unbatched(self):
        mtp = _generator(True)
        self.assertFalse(mtp._is_batchable(_args(0.0)), "greedy: served by MTP")
        self.assertTrue(mtp._is_batchable(_args(0.7)), "sampled: never MTP")
        self.assertTrue(mtp._is_batchable(_args(0.0, repetition_penalty=1.1)), "penalized: never MTP")
        self.assertTrue(mtp._is_batchable(_args(0.0, logit_bias={1: 2.0})))

    def test_models_without_mtp_unchanged(self):
        plain = _generator(False)
        self.assertTrue(plain._is_batchable(_args(0.0)))
        self.assertTrue(plain._is_batchable(_args(0.7)))


if __name__ == "__main__":
    unittest.main()
