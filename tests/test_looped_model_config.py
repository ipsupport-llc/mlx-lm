# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.server import looped_model_config, make_parser
from mlx_lm.utils import load_model

TINY = dict(
    model_type="dory",
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=16,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    sliding_window=8,
    n_input_layers=4,
    n_recurrent_layers=8,
    n_output_layers=4,
    n_recurrent_loops=2,
)


class TestLoopedModelConfig(unittest.TestCase):
    def parse(self, *argv):
        return make_parser().parse_args(list(argv))

    def test_flags_left_out_keep_the_config(self):
        self.assertEqual(looped_model_config(self.parse()), {})

    def test_flags_become_overrides(self):
        args = self.parse("--recurrent-loops", "4", "--recurrent-kv-cache", "last_loop")
        self.assertEqual(
            looped_model_config(args),
            {"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "last_loop"},
        )

    def test_out_of_range_loops_refused(self):
        for n in ("0", "5"):
            with self.assertRaises(ValueError):
                looped_model_config(self.parse("--recurrent-loops", n))

    def test_override_reaches_the_model(self):
        from mlx_lm.models import dory

        model = dory.Model(dory.ModelArgs(**TINY))
        mx.eval(model.parameters())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "config.json").write_text(json.dumps(TINY))
            mx.save_safetensors(
                str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
            )
            for loops, mode, caches in (
                (1, "per_loop", 8),
                (3, "per_loop", 16),
                (3, "last_loop", 8),
            ):
                cfg = looped_model_config(
                    self.parse(
                        "--recurrent-loops", str(loops), "--recurrent-kv-cache", mode
                    )
                )
                m, _ = load_model(path, model_config=cfg)
                self.assertEqual(m.args.n_recurrent_loops, loops)
                self.assertEqual(m.args.recurrent_kv_cache_mode, mode)
                cache = m.make_cache()
                self.assertEqual(len(cache), caches)
                out = m(mx.array([[1, 2, 3]]), cache=cache)
                self.assertEqual(out.shape, (1, 3, TINY["vocab_size"]))


if __name__ == "__main__":
    unittest.main()
