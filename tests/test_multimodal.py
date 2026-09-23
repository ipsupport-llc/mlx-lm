# Copyright © 2026 Apple Inc.

"""Unit tests for mlx_lm/multimodal.py (image inputs in mlx_lm.server).

No model weights needed: the fusion step is stubbed, so this checks the
parts the server owns -- extracting images from OpenAI-style content parts,
expanding the chat template's single image placeholder into the model's
soft-token span, the pad-substituted generation ids, and the image-hashed
prompt-cache key. End-to-end correctness against HF's Gemma4Processor and a
real model was verified separately (input_ids identical to HF's processor;
logits through the input_embeddings path identical to the direct
pixel_values path; "two cats" on the COCO cats photo via the server).
"""

import base64
import io
import types
import unittest

import mlx.core as mx
import numpy as np

try:
    from PIL import Image

    from mlx_lm.multimodal import _IMAGE_KEY_BASE, ImageInputs, extract_images
    from mlx_lm.server import process_message_content

    HAVE_DEPS = True
except ImportError:  # pillow / transformers' Gemma 4 PIL processor missing
    HAVE_DEPS = False

IMAGE, BOI, EOI, PAD = 258880, 255999, 258882, 0


def _png(h, w, seed):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _data_uri(blob):
    return "data:image/png;base64," + base64.b64encode(blob).decode()


def _stub_model():
    captured = {}

    def fuse(ids, pixel_values, position_ids, feats, mask):
        captured["n"] = ids.shape[1]
        return mx.zeros((1, ids.shape[1], 8)), None

    model = types.SimpleNamespace(
        args=types.SimpleNamespace(
            image_token_id=IMAGE, boi_token_id=BOI, eoi_token_id=EOI
        ),
        language_model=types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(pad_token_id=PAD)
            )
        ),
        _fuse_multimodal_inputs=fuse,
    )
    return model, captured


PROCESSOR_CONFIG = {
    "image_processor": {"patch_size": 16, "max_soft_tokens": 280, "pooling_kernel_size": 3}
}


@unittest.skipUnless(HAVE_DEPS, "needs pillow + transformers")
class TestMultimodal(unittest.TestCase):
    def test_extract_images_rewrites_parts_in_order(self):
        a, b = _png(32, 48, 0), _png(40, 24, 1)
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": _data_uri(a)}},
                    {"type": "image_url", "image_url": _data_uri(b)},
                ],
            },
        ]
        self.assertEqual(extract_images(messages), [a, b])
        self.assertEqual(
            messages[1]["content"],
            [{"type": "text", "text": "compare"}, {"type": "image"}, {"type": "image"}],
        )
        # The server must now leave that list for the chat template instead
        # of rejecting non-text parts.
        process_message_content(messages)
        self.assertEqual(messages[1]["content"][1], {"type": "image"})
        self.assertEqual(messages[0]["content"], "sys")

    def test_bad_image_url_is_a_value_error(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}
                ],
            }
        ]
        with self.assertRaises(ValueError):
            extract_images(messages)
        with self.assertRaises(ValueError):
            extract_images(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ftp://x"}}]}]
            )

    def test_text_only_content_still_flattened(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        process_message_content(messages)
        self.assertEqual(messages[0]["content"], "ab")

    def test_build_expands_placeholders_and_keys_cache_by_image(self):
        model, captured = _stub_model()
        ii = ImageInputs(model, PROCESSOR_CONFIG)
        a, b = _png(480, 640, 0), _png(900, 300, 1)
        prompt = [2, 7, IMAGE, 9, IMAGE, 11]

        ids, emb, key = ii.build(prompt, [a, b])

        # boi + N soft tokens + eoi per image, text tokens untouched.
        self.assertEqual(ids[:2], [2, 7])
        self.assertEqual(ids[2], BOI)
        self.assertEqual(ids[-1], 11)
        self.assertEqual(ids.count(BOI), 2)
        self.assertEqual(ids.count(EOI), 2)
        # Generation ids carry pad (not the image id) at image positions --
        # required for per-layer-embedding models to match HF.
        self.assertNotIn(IMAGE, ids)
        self.assertEqual(len(ids), len(key))
        self.assertEqual(emb.shape[0], len(ids))
        self.assertEqual(captured["n"], len(ids))

        # Cache key: pseudo ids at image positions, one value per image.
        pseudo = [t for t in key if t >= _IMAGE_KEY_BASE]
        self.assertEqual(len(pseudo), ids.count(PAD))
        self.assertEqual(len(set(pseudo)), 2)

        # Same image -> same key; different image -> key diverges at the
        # first image position (only the text before it can be reused).
        _, _, key_same = ii.build(prompt, [a, b])
        self.assertEqual(key, key_same)
        _, _, key_other = ii.build(prompt, [_png(480, 640, 5), b])
        first_diff = next(i for i, (x, y) in enumerate(zip(key, key_other)) if x != y)
        self.assertEqual(first_diff, 3)  # right after [2, 7, BOI]

    def test_build_rejects_placeholder_count_mismatch(self):
        model, _ = _stub_model()
        ii = ImageInputs(model, PROCESSOR_CONFIG)
        with self.assertRaises(ValueError):
            ii.build([2, IMAGE, 3], [_png(64, 64, 0), _png(64, 64, 1)])


if __name__ == "__main__":
    unittest.main()
