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


class TestRequestLoops(unittest.TestCase):
    """A request's loop count changes the model without reloading it."""

    @classmethod
    def setUpClass(cls):
        from mlx_lm.models import dory

        cls.dir = tempfile.TemporaryDirectory()
        path = Path(cls.dir.name)
        model = dory.Model(dory.ModelArgs(**TINY))
        mx.eval(model.parameters())
        (path / "config.json").write_text(json.dumps(TINY))
        mx.save_safetensors(
            str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        cls.path = str(path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def provider(self, *argv):
        from mlx_lm.server import ModelProvider

        p = ModelProvider(make_parser().parse_args(["--model", self.path, *argv]))
        p._load = self.fake_load(p)
        return p

    def fake_load(self, provider):
        # The tokenizer needs files the tiny model has none of.
        def _load(model_path, adapter_path=None, draft_model_path=None):
            model, _ = load_model(
                Path(model_path), model_config=looped_model_config(provider.cli_args)
            )
            provider.loads = getattr(provider, "loads", 0) + 1
            provider._set_model_key((model_path, adapter_path, draft_model_path), model)
            provider.model = model

        return _load

    def test_loops_switch_without_reload(self):
        p = self.provider()
        model, _ = p.load("default_model")
        self.assertEqual(p.model_key[3], 2)
        self.assertEqual(len(model.make_cache()), 12)
        p.load("default_model", loops=4)
        self.assertEqual(p.model_key[3], 4)
        self.assertEqual(len(p.model.make_cache()), 20)
        self.assertIs(p.model, model)
        p.load("default_model")
        self.assertEqual(p.model_key[3], 2, "no loops: the default again")
        self.assertEqual(p.loads, 1)

    def test_default_is_the_flag(self):
        p = self.provider("--recurrent-loops", "1")
        p.load("default_model")
        self.assertEqual(p.model_key[3], 1)
        p.load("default_model", loops=3)
        self.assertEqual(p.model_key[3], 3)
        p.load("default_model")
        self.assertEqual(p.model_key[3], 1)

    def test_switched_model_matches_one_loaded_with_that_count(self):
        p = self.provider()
        p.load("default_model", loops=3)
        fresh, _ = load_model(Path(self.path), model_config={"n_recurrent_loops": 3})
        x = mx.array([[1, 2, 3, 4]])
        a = p.model(x, cache=p.model.make_cache())
        b = fresh(x, cache=fresh.make_cache())
        self.assertEqual(float(mx.abs(a - b).max()), 0.0)


class TestRequestLoopsOverHTTP(unittest.TestCase):
    """recurrent_loops in a request body reaches the model and its cache key."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import shutil
        import threading

        from huggingface_hub import snapshot_download

        from mlx_lm.models import dory
        from mlx_lm.server import (
            APIHandler,
            LRUPromptCache,
            ModelProvider,
            ResponseGenerator,
        )

        tok = Path(snapshot_download("mlx-community/Qwen1.5-0.5B-Chat-4bit"))
        cls.dir = tempfile.TemporaryDirectory()
        path = Path(cls.dir.name)
        for f in tok.iterdir():
            if f.name.startswith(("tokenizer", "vocab", "merges", "special")):
                shutil.copy(f, path / f.name)
        cfg = dict(TINY, vocab_size=151936)
        model = dory.Model(dory.ModelArgs(**cfg))
        mx.eval(model.parameters())
        (path / "config.json").write_text(json.dumps(cfg))
        mx.save_safetensors(
            str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        cli = make_parser().parse_args(["--model", str(path)])
        cls.provider = ModelProvider(cli)
        cls.generator = ResponseGenerator(cls.provider, LRUPromptCache())
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *a, **k: APIHandler(cls.generator, *a, **k),
        )
        cls.port = cls.httpd.server_port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()
        cls.generator.stop_and_join()
        cls.dir.cleanup()

    def complete(self, **body):
        import http.client

        conn = http.client.HTTPConnection("localhost", self.port, timeout=60)
        payload = {"prompt": "Hello there", "max_tokens": 3, **body}
        conn.request("POST", "/v1/completions", json.dumps(payload))
        r = conn.getresponse()
        return r.status, r.read()

    def test_request_loops(self):
        status, _ = self.complete(recurrent_loops=3)
        self.assertEqual(status, 200)
        self.assertEqual(self.provider.model_key[3], 3)
        status, _ = self.complete()
        self.assertEqual(status, 200)
        self.assertEqual(self.provider.model_key[3], 2)
        status, _ = self.complete(recurrent_loops=1)
        self.assertEqual(status, 200)
        self.assertEqual(self.provider.model_key[3], 1)
        self.assertEqual(len(self.provider.model.make_cache()), 8)

    def test_bad_loops_refused(self):
        for bad in (0, 5, "2", True, 2.5):
            status, body = self.complete(recurrent_loops=bad)
            self.assertEqual(status, 400, (bad, body))


if __name__ == "__main__":
    unittest.main()
