# Copyright © 2026 Apple Inc.

import unittest

import numpy as np
from mlx import nn

from mlx_lm.processors import load_processor, muse_spark


class TestMuseSparkProcessor(unittest.TestCase):
    def test_preprocess_image(self):
        processor = muse_spark.Processor(max_num_image_tokens=16)
        # (height, width), pixel shape, number of tokens
        test_cases = [
            # A multiple of 28, no resize.
            ((56, 84), (3, 56, 84), 6),
            # Resized to the nearest grid with the same aspect ratio.
            ((100, 50), (3, 112, 56), 8),
            # Resized down to max_num_image_tokens.
            ((280, 280), (3, 112, 112), 16),
        ]
        for size, shape, num_tokens in test_cases:
            with self.subTest(size=size):
                image = np.zeros((*size, 3), dtype=np.uint8)
                pixels, n = processor.preprocess_image(image)
                self.assertEqual(pixels.shape, shape)
                self.assertEqual(n, num_tokens)

    def test_call(self):
        processor = muse_spark.Processor()
        images = [
            np.zeros((56, 84, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        ]
        text, pixel_values = processor("A <|patch|> and B <|patch|>.", images=images)
        expected = (
            "A <|image_start|>" + "<|patch|>" * 6 + "<|image_end|>"
            " and B <|image_start|><|patch|><|image_end|>."
        )
        self.assertEqual(text, expected)
        self.assertEqual([p.shape for p in pixel_values], [(3, 56, 84), (3, 28, 28)])

    def test_call_errors(self):
        processor = muse_spark.Processor()
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("<|patch|> <|patch|>", images=[image])
        with self.assertRaisesRegex(ValueError, "for 1 images"):
            processor("No image", images=[image])

    def test_load_processor(self):
        model = nn.Module()
        self.assertIsNone(load_processor(model, {"model_type": "muse_spark"}))
        self.assertIsNone(load_processor(model, {"vision_config": {}}))
        config = {
            "vision_config": {"patch_size": 16},
            "processor_type": "muse_spark",
        }
        processor = load_processor(model, config)
        self.assertEqual(processor.patch_size, 16)


class TestMuseSparkImageInputs(unittest.TestCase):
    def test_build_expands_each_placeholder(self):
        import io
        from types import SimpleNamespace

        import mlx.core as mx
        from PIL import Image

        from mlx_lm.multimodal import MuseSparkImageInputs

        calls = {}

        class FakeModel:
            args = SimpleNamespace(image_token_id=7)

            def embed_inputs(self, inputs, *, pixel_values):
                calls["inputs"] = inputs.tolist()
                calls["shapes"] = [p.shape for p in pixel_values]
                return mx.zeros((inputs.shape[0], 4))

        def png(h, w):
            buf = io.BytesIO()
            Image.new("RGB", (w, h)).save(buf, format="PNG")
            return buf.getvalue()

        inputs = MuseSparkImageInputs(
            FakeModel(), {"vision_config": {"patch_size": 14}}, {"<|image_start|>": 5, "<|image_end|>": 6}
        )
        expanded, embeddings, key = inputs.build([1, 7, 2, 7], [png(56, 84), png(28, 28)])
        self.assertEqual(expanded, [1, 5] + [7] * 6 + [6, 2, 5, 7, 6])
        self.assertEqual(calls["inputs"], expanded)
        self.assertEqual(calls["shapes"], [(3, 56, 84), (3, 28, 28)])
        self.assertEqual(embeddings.shape, (len(expanded), 4))
        # The two images get their own cache ids, the markers keep theirs.
        self.assertEqual(len(key), len(expanded))
        self.assertNotEqual(key[2], key[11])
        with self.assertRaises(ValueError):
            inputs.build([1, 7], [png(28, 28), png(28, 28)])


if __name__ == "__main__":
    unittest.main()
