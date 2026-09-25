# Copyright © 2026 Apple Inc.

"""Bidirectional attention within an image (use_bidirectional_attention =
"vision": Gemma 4 26B-A4B, 12B), without torch: chunked prefill -- also
cut mid-image by a small prefill_step_size, and after a cached prefix --
must equal one pass, the spans must be cleared after, and the helpers that
find them. HF parity: tests/test_gemma4_multimodal_parity.py."""

import json
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.generate import generate_step
from mlx_lm.models import gemma4
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.multimodal import _IMAGE_KEY_BASE, vision_spans

def setUpModule():
    # CPU: on the GPU, a 1-token pass and a multi-token one already differ
    # by ~2e-2 on this tiny random model (upstream too) -- kernel numerics,
    # not what's tested here. On the CPU they agree to ~2e-5.
    global _device
    _device = mx.default_device()
    mx.set_default_device(mx.cpu)


def tearDownModule():
    mx.set_default_device(_device)


CONFIG = json.loads(r'''{"audio_token_id": 258881, "boa_token_id": 256000, "boi_token_id": 255999, "eoi_token_id": 258882, "image_token_id": 258880, "model_type": "gemma4", "text_config": {"attention_bias": false, "attention_dropout": 0.0, "attention_k_eq_v": false, "bos_token_id": 2, "enable_moe_block": false, "eos_token_id": 1, "final_logit_softcapping": 30.0, "global_head_dim": 64, "head_dim": 32, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64, "hidden_size_per_layer_input": 8, "intermediate_size": 128, "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"], "max_position_embeddings": 131072, "model_type": "gemma4_text", "moe_intermediate_size": null, "num_attention_heads": 2, "num_experts": null, "num_hidden_layers": 6, "num_key_value_heads": 1, "num_kv_shared_layers": 0, "pad_token_id": 0, "rms_norm_eps": 1e-06, "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"}, "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}, "sliding_window": 512, "tie_word_embeddings": true, "top_k_experts": null, "use_bidirectional_attention": "vision", "use_cache": true, "use_double_wide_mlp": false, "vocab_size": 262144, "vocab_size_per_layer_input": 262144}, "tie_word_embeddings": true, "vision_config": {"attention_bias": false, "attention_dropout": 0.0, "head_dim": 32, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64, "intermediate_size": 128, "max_position_embeddings": 131072, "model_type": "gemma4_vision", "num_attention_heads": 2, "num_hidden_layers": 1, "num_key_value_heads": 2, "patch_size": 8, "pooling_kernel_size": 2, "position_embedding_size": 64, "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"}, "standardize": false, "use_clipped_linears": true}}''')
IMAGE = CONFIG["image_token_id"]


def _model():
    mx.random.seed(0)
    model = gemma4.Model(gemma4.ModelArgs.from_dict(CONFIG))
    mx.eval(model.parameters())
    return model


def _prompt(model):
    """Two 'images' (random embeddings at image-token positions) in text."""
    ids = [5, 6, CONFIG["boi_token_id"]] + [IMAGE] * 6 + [CONFIG["eoi_token_id"], 7, CONFIG["boi_token_id"]] + [IMAGE] * 5 + [CONFIG["eoi_token_id"], 8, 9, 10]
    pad = model.language_model.model.config.pad_token_id
    gen_ids = mx.array([pad if t == IMAGE else t for t in ids])
    emb = model.language_model.model.embed_tokens(gen_ids)
    rng = np.random.default_rng(1)
    img = mx.array(np.array(ids) == IMAGE)[:, None]
    emb = mx.where(img, mx.array(rng.normal(size=emb.shape).astype(np.float32)), emb)
    return ids, gen_ids, emb


class TestVisionBidirectional(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(gemma4.Model.image_spans([1, IMAGE, IMAGE, 2, IMAGE, 3], IMAGE), [(1, 3), (4, 5)])
        key = [1, 2, _IMAGE_KEY_BASE + 5, _IMAGE_KEY_BASE + 5, 3, 4, _IMAGE_KEY_BASE + 9, 7]
        self.assertEqual(vision_spans(key), [(2, 4), (6, 7)])
        model = _model()
        model.set_vision_spans([(10, 20)])
        self.assertEqual(model.prefill_chunk_size(0, 15), 10, "stops before the image")
        self.assertEqual(model.prefill_chunk_size(10, 4), 10, "or takes it whole")
        self.assertEqual(model.prefill_chunk_size(0, 25), 25, "an image inside the chunk is fine")
        model.set_vision_spans(None)
        self.assertEqual(model.prefill_chunk_size(0, 15), 15)

    def _logits(self, model, gen_ids, emb, spans, split, cache=None):
        model.set_vision_spans(spans)
        try:
            cache = cache or make_prompt_cache(model)
            out, i = [], cache[0].offset
            for n in split:
                out.append(model(gen_ids[None, i:i + n], cache=cache, input_embeddings=emb[None, i:i + n]))
                i += n
        finally:
            model.set_vision_spans(None)
        return mx.concatenate(out, axis=1)

    def test_chunks_that_respect_images_equal_one_pass(self):
        model = _model()
        ids, gen_ids, emb = _prompt(model)
        spans = gemma4.Model.image_spans(ids, IMAGE)
        L = len(ids)
        one = self._logits(model, gen_ids, emb, spans, [L])
        for step in (1, 2, 3, 5):
            split, i = [], 0
            model.set_vision_spans(spans)
            while i < L:
                n = model.prefill_chunk_size(i, min(step, L - i))
                split.append(n)
                i += n
            model.set_vision_spans(None)
            chunked = self._logits(model, gen_ids, emb, spans, split)
            self.assertLess(mx.abs(chunked - one).max().item(), 1e-3, (step, split))
        causal = self._logits(model, gen_ids, emb, None, [L])
        self.assertGreater(mx.abs(causal - one).max().item(), 1e-3, "the images' tokens do see each other")

    def test_generate_step_small_prefill_steps_and_cached_prefix(self):
        model = _model()
        ids, gen_ids, emb = _prompt(model)
        spans = gemma4.Model.image_spans(ids, IMAGE)

        def first(prefill_step, cached=0):
            cache = make_prompt_cache(model)
            model.set_vision_spans(spans)
            try:
                if cached:
                    model(gen_ids[None, :cached], cache=cache, input_embeddings=emb[None, :cached])
                tok, lp = next(generate_step(gen_ids[cached:], model, input_embeddings=emb[cached:],
                                             prefill_step_size=prefill_step, prompt_cache=cache, max_tokens=1,
                                             stream=mx.cpu))
                mx.eval(lp)
            finally:
                model.set_vision_spans(None)
            return tok, lp

        ref_tok, ref = first(4096)
        for step, cached in ((2, 0), (3, 0), (4, 2), (3, 1)):
            tok, lp = first(step, cached)
            self.assertEqual(tok, ref_tok, (step, cached))
            self.assertLess(mx.abs(lp - ref).max().item(), 1e-3, (step, cached))

    def test_spans_dont_outlive_the_prefill(self):
        model = _model()
        ids, gen_ids, emb = _prompt(model)
        causal = model(gen_ids[None], input_embeddings=emb[None])
        model.set_vision_spans(gemma4.Model.image_spans(ids, IMAGE))
        model.set_vision_spans(None)
        again = model(gen_ids[None], input_embeddings=emb[None])
        self.assertTrue(mx.array_equal(causal, again).item())

    def test_models_without_it_ignore_spans(self):
        cfg = json.loads(json.dumps(CONFIG))
        cfg["text_config"]["use_bidirectional_attention"] = None
        mx.random.seed(0)
        model = gemma4.Model(gemma4.ModelArgs.from_dict(cfg))
        ids, gen_ids, emb = _prompt(model)
        model.set_vision_spans(gemma4.Model.image_spans(ids, IMAGE))
        self.assertIsNone(model.language_model.model.vision_spans)


if __name__ == "__main__":
    unittest.main()
