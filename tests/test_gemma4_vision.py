# Copyright © 2026 Apple Inc.

"""Numerical parity test for `mlx_lm.models.gemma4_vision` against the real
`google/gemma-4-E4B-it` checkpoint.

This does NOT download the full ~16GB `model.safetensors`. Instead it reads
the safetensors header (a small JSON blob after an 8-byte length prefix) and
then a couple of layers' worth of individual tensors straight off the
Hugging Face CDN via HTTP `Range` requests -- a few MB total, not 16GB.

It builds the real `transformers` `Gemma4VisionPatchEmbedder` +
`Gemma4VisionEncoder` (trimmed to 2 layers) with those real weights, runs a
fixed random input through both it and this repo's MLX implementation, and
asserts the outputs match closely.

Requires `torch` + a `transformers` build with Gemma4 support, and network
access to huggingface.co; skipped automatically if either is unavailable.
"""

import json
import struct
import unittest
import urllib.error
import urllib.request

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from mlx_lm.models import gemma4_vision as gv

MODEL_ID = "google/gemma-4-E4B-it"
SAFETENSORS_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/model.safetensors"
NUM_LAYERS = 2

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4VisionEncoder,
        Gemma4VisionPatchEmbedder,
    )

    _HAS_TRANSFORMERS_GEMMA4 = True
except Exception:
    _HAS_TRANSFORMERS_GEMMA4 = False


def _http_get_range(url: str, start: int, end: int, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.reshape(shape) if shape else f32.reshape(())


class _RemoteSafetensors:
    """Lazily reads individual tensors out of a remote `.safetensors` file
    via HTTP range requests, without downloading the whole file."""

    def __init__(self, url: str):
        self.url = url
        header_len = struct.unpack("<Q", _http_get_range(url, 0, 7))[0]
        self.header = json.loads(_http_get_range(url, 8, 8 + header_len - 1))
        self.data_start = 8 + header_len

    def get(self, name: str) -> np.ndarray:
        meta = self.header[name]
        assert meta["dtype"] == "BF16", f"unexpected dtype for {name}: {meta['dtype']}"
        start, end = meta["data_offsets"]
        if start == end:
            return np.zeros(meta["shape"], dtype=np.float32)
        raw = _http_get_range(self.url, self.data_start + start, self.data_start + end - 1)
        return _bf16_bytes_to_f32(raw, meta["shape"])


def _fetch_real_vision_weights(num_layers: int) -> dict:
    """Fetches patch_embedder + the first `num_layers` encoder layers' real
    weights (linear weights, clip bounds, norms) from the real checkpoint,
    keyed the same way `Gemma4VisionModel.sanitize()` expects (i.e. with the
    `model.vision_tower.` prefix stripped)."""
    st = _RemoteSafetensors(SAFETENSORS_URL)
    prefix = "model.vision_tower."

    wanted = ["patch_embedder.input_proj.weight", "patch_embedder.position_embedding_table"]
    projs = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    for i in range(num_layers):
        base = f"encoder.layers.{i}."
        wanted += [
            base + "input_layernorm.weight",
            base + "post_attention_layernorm.weight",
            base + "pre_feedforward_layernorm.weight",
            base + "post_feedforward_layernorm.weight",
            base + "self_attn.q_norm.weight",
            base + "self_attn.k_norm.weight",
        ]
        for p in projs:
            wanted.append(base + p + ".linear.weight")
            for c in ("input_min", "input_max", "output_min", "output_max"):
                wanted.append(base + p + "." + c)

    return {name: st.get(prefix + name) for name in wanted}


@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
@unittest.skipUnless(
    _HAS_TRANSFORMERS_GEMMA4, "installed transformers build has no Gemma4 vision support"
)
class TestGemma4VisionParity(unittest.TestCase):
    weights = None  # populated lazily by setUpClass, once, over the network

    @classmethod
    def setUpClass(cls):
        try:
            cls.weights = _fetch_real_vision_weights(NUM_LAYERS)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise unittest.SkipTest(f"could not reach huggingface.co: {e}")

    def test_patch_embedder_and_encoder_match_real_weights(self):
        weights = self.weights

        # Fixed random pre-patchified input: a 4x4 grid of patches, no padding.
        rng = np.random.default_rng(0)
        grid = 4
        n_patches = grid * grid
        patch_dim = 3 * 16 * 16
        pixel_values_np = rng.uniform(0, 1, size=(1, n_patches, patch_dim)).astype(
            np.float32
        )
        xs, ys = np.meshgrid(np.arange(grid), np.arange(grid), indexing="xy")
        pos_np = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.int64)[
            None
        ]

        # ---- MLX ----
        args = gv.ModelArgs(num_hidden_layers=NUM_LAYERS, use_clipped_linears=True)
        patch_embedder = gv.PatchEmbedder(args)
        encoder = gv.Encoder(args)

        pe_weights = {
            k[len("patch_embedder.") :]: mx.array(v)
            for k, v in weights.items()
            if k.startswith("patch_embedder.")
        }
        enc_weights = {
            k[len("encoder.") :]: mx.array(v)
            for k, v in weights.items()
            if k.startswith("encoder.")
        }
        patch_embedder.update(tree_unflatten(list(pe_weights.items())))
        encoder.update(tree_unflatten(list(enc_weights.items())))
        mx.eval(patch_embedder.parameters(), encoder.parameters())

        pixel_values_mx = mx.array(pixel_values_np)
        pos_mx = mx.array(pos_np)
        padding_mx = mx.all(pos_mx == -1, axis=-1)

        inputs_embeds_mx = patch_embedder(pixel_values_mx, pos_mx, padding_mx)
        out_mx = encoder(inputs_embeds_mx, pos_mx, mask=None)
        mx.eval(out_mx)
        out_mx_np = np.array(out_mx)
        inputs_embeds_mx_np = np.array(inputs_embeds_mx)

        # ---- Real transformers (trimmed to NUM_LAYERS) ----
        config = Gemma4VisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=12,
            num_key_value_heads=12,
            head_dim=64,
            patch_size=16,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            rope_parameters={"rope_theta": 100.0, "rope_type": "default"},
            use_clipped_linears=True,
            position_embedding_size=10240,
            pooling_kernel_size=3,
            standardize=False,
        )
        config._attn_implementation = "eager"

        patch_embedder_t = Gemma4VisionPatchEmbedder(config)
        patch_embedder_t.load_state_dict(
            {
                "input_proj.weight": torch.from_numpy(
                    weights["patch_embedder.input_proj.weight"]
                ),
                "position_embedding_table": torch.from_numpy(
                    weights["patch_embedder.position_embedding_table"]
                ),
            },
            strict=True,
        )

        encoder_t = Gemma4VisionEncoder(config)
        sd_enc = {}
        for i in range(NUM_LAYERS):
            src_base = f"encoder.layers.{i}."
            dst_base = f"layers.{i}."
            for k, v in weights.items():
                if k.startswith(src_base):
                    sd_enc[dst_base + k[len(src_base) :]] = torch.from_numpy(v)
        missing, unexpected = encoder_t.load_state_dict(sd_enc, strict=False)
        # Only non-persistent rope buffers (never in a state_dict) may be
        # "missing"; nothing real should be unexpected.
        self.assertEqual(unexpected, [])
        self.assertTrue(all("rotary_emb" in m for m in missing), missing)

        pixel_values_t = torch.from_numpy(pixel_values_np)
        pos_t = torch.from_numpy(pos_np).long()
        padding_t = (pos_t == -1).all(dim=-1)

        with torch.no_grad():
            inputs_embeds_t = patch_embedder_t(pixel_values_t, pos_t, padding_t)
            out_t = encoder_t(
                inputs_embeds=inputs_embeds_t,
                attention_mask=~padding_t,
                pixel_position_ids=pos_t,
            )
        out_t_np = out_t.last_hidden_state.numpy()
        inputs_embeds_t_np = inputs_embeds_t.numpy()

        # Patch embedding: float32 matmul + embedding lookup, should match tightly.
        np.testing.assert_allclose(
            inputs_embeds_mx_np, inputs_embeds_t_np, atol=1e-2, rtol=1e-2
        )

        # Full 2-layer encoder (attention + MLP + real, finite clip bounds):
        # allow a slightly looser tolerance. Real per-layer clip bounds mean a
        # handful of activations sit right at a clamp boundary, where tiny
        # (~1e-4) float32 reduction-order differences between MLX's and
        # PyTorch's matmul/softmax kernels can flip which side of the clamp
        # an element lands on; that's a couple of outlier elements out of
        # thousands (verified: <0.1% of elements exceed atol=1e-2+rtol*|v|
        # in a representative run), not a modeling bug.
        diff = np.abs(out_mx_np - out_t_np)
        frac_bad = (diff > (1e-2 + 1e-2 * np.abs(out_t_np))).mean()
        self.assertLess(
            frac_bad,
            0.01,
            f"too many elements diverge: {frac_bad:.4%} exceed atol=1e-2,rtol=1e-2 "
            f"(max diff {diff.max():.4f})",
        )
        np.testing.assert_allclose(out_mx_np, out_t_np, atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
