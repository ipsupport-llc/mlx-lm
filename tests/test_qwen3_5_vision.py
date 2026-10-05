# Copyright © 2026 Apple Inc.

"""Qwen3.5 image inputs: mRoPE, preprocessing, positions, config."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_lm.models.qwen3_5 import MRoPE, MRoPEState
from mlx_lm.multimodal import Qwen35ImageInputs
from mlx_lm.utils import save_config


class TestMRoPE(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.x = mx.random.normal((1, 4, 7, 256))
        self.state = MRoPEState()
        import mlx.nn as nn

        self.rope = MRoPE(
            64, 1e7, [11, 11, 10], self.state, nn.RoPE(64, traditional=False, base=1e7)
        )

    def reference(self, offset):
        return mx.fast.rope(
            self.x, 64, traditional=False, base=1e7, scale=1.0, offset=offset
        )

    def test_without_images_is_plain_rope(self):
        self.assertTrue(
            mx.allclose(self.rope(self.x, offset=5), self.reference(5)).item()
        )

    def test_equal_axes_match_plain_rope(self):
        # Text-only positions (the same on t, h, w) must give plain RoPE: the
        # rotation and the channel layout are right.
        pos = mx.broadcast_to(mx.arange(20, dtype=mx.int32)[None], (3, 20))
        self.state.positions, self.state.delta = pos, 0
        # Exactly: a text chunk of an image prompt uses the text RoPE itself.
        self.assertTrue(
            mx.array_equal(self.rope(self.x, offset=3), self.reference(3)).item()
        )

    def test_image_positions_differ_from_plain_rope(self):
        pos = mx.array([[0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 1]], dtype=mx.int32)
        self.state.positions, self.state.delta = pos, 0
        self.assertFalse(
            mx.allclose(
                self.rope(self.x[:, :, :4], offset=0), self.reference(0)[:, :, :4]
            ).item()
        )

    def test_positions_past_the_prompt_continue_with_delta(self):
        pos = mx.broadcast_to(mx.arange(4, dtype=mx.int32)[None], (3, 4))
        self.state.positions, self.state.delta = pos, -2
        window = self.state.window(2, 5)
        self.assertEqual(window[0].tolist(), [2, 3, 2, 3, 4])  # 4, 5, 6 shifted by -2

    def test_axes_interleave(self):
        # Frequencies 1, 4, 7, ... follow h; 2, 5, 8, ... follow w (HF's
        # apply_interleaved_mrope with section [11, 11, 10]).
        sel = self.rope._selector.tolist()
        self.assertEqual(sel[:6], [0, 1, 2, 0, 1, 2])
        self.assertEqual(sel.count(1), 11)
        self.assertEqual(sel.count(2), 10)


class _FakeModel:
    class args:
        image_token_id = 9

    def embed_with_images(self, ids, pixel_values, grid_thw):
        self.seen = (ids, pixel_values, grid_thw)
        return mx.zeros((1, ids.shape[1], 8))


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, format="PNG")
    return buf.getvalue()


class TestQwen35ImageInputs(unittest.TestCase):
    def setUp(self):
        cfg = {
            "patch_size": 16,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        }
        self.model = _FakeModel()
        self.inputs = Qwen35ImageInputs(self.model, cfg)

    def test_resize_is_a_multiple_of_32_within_limits(self):
        h, w = self.inputs._resize(480, 640)
        self.assertEqual((h % 32, w % 32), (0, 0))
        self.assertLessEqual(h * w, self.inputs.MAX_PIXELS)
        h, w = self.inputs._resize(4000, 4000)
        self.assertLessEqual(h * w, self.inputs.MAX_PIXELS)
        h, w = self.inputs._resize(50, 50)
        self.assertGreaterEqual(h * w, 65536)

    def test_patches(self):
        pv, grid = self.inputs._patches(Image.new("RGB", (640, 480)))
        self.assertEqual(grid, (1, 30, 40))
        self.assertEqual(pv.shape, (1200, 3 * 2 * 16 * 16))

    def test_build_expands_and_positions(self):
        # text(2) <image 640x480 -> 15x20 merged> text(1)
        ids, _, key = self.inputs.build([1, 2, 9, 3], [_png(640, 480)])
        self.assertEqual(len(ids), 2 + 300 + 1)
        self.assertEqual(ids.count(9), 300)
        positions, delta = self.inputs.media_positions
        pos = np.array(positions)
        self.assertEqual(pos[:, :2].tolist(), [[0, 1]] * 3)
        block = pos[:, 2:302]
        self.assertEqual(block[0].max(), 2)  # t: one frame
        self.assertEqual(block[1].max(), 2 + 14)  # h: 15 rows
        self.assertEqual(block[2].max(), 2 + 19)  # w: 20 columns
        self.assertEqual(pos[:, -1].tolist(), [22] * 3)  # after the image: max + 1
        self.assertEqual(delta, 23 - len(ids))
        self.assertTrue(all(k >= 1 << 40 for k in key[2:302]))

    def test_thin_images_are_refused_and_the_cap_holds(self):
        with self.assertRaises(ValueError):
            self.inputs._resize(1, 1_000_000)
        h, w = self.inputs._resize(100, 19_000)
        self.assertLessEqual(h * w, self.inputs.MAX_PIXELS)

    def test_placeholder_count_must_match(self):
        with self.assertRaises(ValueError):
            self.inputs.build([1, 9, 9], [_png(64, 64)])


class TestSaveConfig(unittest.TestCase):
    def test_keeps_qwen3_5_vision_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            save_config({"model_type": "qwen3_5", "vision_config": {"depth": 24}}, path)
            self.assertIn("vision_config", json.loads(path.read_text()))
            save_config({"model_type": "qwen3", "vision_config": {"depth": 24}}, path)
            self.assertNotIn("vision_config", json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
