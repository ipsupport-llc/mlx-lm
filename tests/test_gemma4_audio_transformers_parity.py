# Copyright © 2025 Apple Inc.

"""Numerical parity test for mlx_lm.models.gemma4_audio against the real
``transformers`` Gemma4 audio implementation.

This is intentionally *not* exercised by the default mlx-lm test run (via
``importorskip``) since neither ``torch`` nor ``transformers`` are mlx-lm
runtime dependencies. Run explicitly with an environment that has a recent
``transformers`` (with Gemma4 support) and ``torch`` installed, e.g.:

    pytest tests/test_gemma4_audio_transformers_parity.py -v

The test builds small random Gemma4AudioConfig/ModelArgs pairs, copies the
HF model's own randomly-initialized weights (bit for bit, with layout
conversions only where mlx's conv/linear layouts differ from torch's) into
the mlx model, and asserts the forward outputs agree to near float32
precision. Comparisons are run with mlx on CPU (`mx.set_default_device
(mx.cpu)`) to avoid conflating this with ordinary Metal-vs-CPU float32
kernel non-determinism, which independently measured ~3e-4 max abs
difference on this same architecture -- two-to-three orders of magnitude
larger than the ~1e-7 achieved on CPU, and unrelated to correctness.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig,
        Gemma4TextConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4AudioModel as HFAudioModel,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder as HFMultimodalEmbedder,
    )
except ImportError:
    pytest.skip(
        "installed `transformers` has no Gemma4 audio support", allow_module_level=True
    )

import mlx.core as mx

from mlx_lm.models import gemma4_audio as ga

mx.set_default_device(mx.cpu)


COMMON_CONFIG = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    subsampling_conv_channels=[8, 4],
    conv_kernel_size=5,
    attention_chunk_size=6,
    attention_context_left=5,
    attention_context_right=2,
    output_proj_dims=16,
)
MEL_BINS = COMMON_CONFIG["subsampling_conv_channels"][0]  # required by the
# hardcoded `(channels[0] // 4) * channels[1]` sub-sample projection formula


def _remap_name(name: str) -> str:
    # `Gemma4AudioCausalConv1d` subclasses `nn.Conv1d` directly in torch, so
    # its weight has no extra prefix. In mlx it's wrapped as
    # `CausalConv1d.conv` (mlx.nn.Conv1d has no causal padding of its own).
    if name.endswith("depthwise_conv1d.weight"):
        return name.replace("depthwise_conv1d.weight", "depthwise_conv1d.conv.weight")
    return name


def _convert_weight(name: str, tensor: "torch.Tensor") -> mx.array:
    arr = tensor.detach().numpy()
    if name.endswith(("layer0.conv.weight", "layer1.conv.weight")):
        # torch Conv2d: (out, in/groups, kh, kw) -> mlx Conv2d: (out, kh, kw, in/groups)
        arr = arr.transpose(0, 2, 3, 1)
    elif name.endswith("depthwise_conv1d.weight"):
        # torch Conv1d: (out, in/groups, k) -> mlx Conv1d: (out, k, in/groups)
        arr = arr.transpose(0, 2, 1)
    return mx.array(arr)


def _build_models(use_clipped_linears: bool, seed: int):
    torch.manual_seed(seed)
    hf_cfg = Gemma4AudioConfig(use_clipped_linears=use_clipped_linears, **COMMON_CONFIG)
    mlx_cfg = ga.ModelArgs(use_clipped_linears=use_clipped_linears, **COMMON_CONFIG)

    hf_model = HFAudioModel(hf_cfg)
    hf_model.eval()

    mlx_model = ga.AudioModel(mlx_cfg)
    weights = {
        _remap_name(n): _convert_weight(n, t) for n, t in hf_model.state_dict().items()
    }
    mlx_model.load_weights(list(weights.items()), strict=True)

    return hf_cfg, mlx_cfg, hf_model, mlx_model


def _randomize_clip_bounds(hf_model, mlx_model, seed: int):
    """Give the ClippableLinear input/output bounds finite, non-trivial,
    asymmetric values (the untrained default is +/-inf, i.e. a no-op) and
    re-sync them into the mlx model, so the test actually exercises the
    clamp path rather than the trivially-true unclamped path."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for name, buf in hf_model.named_buffers():
            if name.endswith("input_min"):
                buf.copy_(torch.tensor(-abs(float(rng.normal(2.0, 0.5)))))
            elif name.endswith("input_max"):
                buf.copy_(torch.tensor(abs(float(rng.normal(2.0, 0.5)))))
            elif name.endswith("output_min"):
                buf.copy_(torch.tensor(-abs(float(rng.normal(2.0, 0.5)))))
            elif name.endswith("output_max"):
                buf.copy_(torch.tensor(abs(float(rng.normal(2.0, 0.5)))))

    weights = {
        _remap_name(n): _convert_weight(n, t) for n, t in hf_model.state_dict().items()
    }
    mlx_model.load_weights(list(weights.items()), strict=True)


def _run_and_compare(hf_model, mlx_model, feats_np, mask_np, atol=5e-5):
    feats_t = torch.tensor(feats_np)
    mask_t = None if mask_np is None else torch.tensor(mask_np)
    with torch.no_grad():
        hf_out = hf_model(feats_t, mask_t)

    feats_m = mx.array(feats_np)
    mask_m = None if mask_np is None else mx.array(mask_np)
    mlx_hidden, mlx_mask = mlx_model(feats_m, mask_m)

    if hf_out.attention_mask is None:
        assert mlx_mask is None
    else:
        assert np.array_equal(hf_out.attention_mask.numpy(), np.array(mlx_mask))

    hf_hidden = hf_out.last_hidden_state.numpy()
    max_abs = np.max(np.abs(hf_hidden - np.array(mlx_hidden)))
    assert max_abs < atol, f"max abs diff {max_abs} exceeds {atol}"
    return max_abs


@pytest.mark.parametrize("use_clipped_linears", [True, False])
def test_audio_model_parity_with_padding(use_clipped_linears):
    hf_cfg, mlx_cfg, hf_model, mlx_model = _build_models(use_clipped_linears, seed=0)

    batch, seq_len = 2, 37  # not a multiple of attention_chunk_size (6)
    rng = np.random.default_rng(1)
    feats_np = rng.normal(size=(batch, seq_len, MEL_BINS)).astype(np.float32)
    mask_np = np.ones((batch, seq_len), dtype=bool)
    mask_np[1, 30:] = False  # exercise real padding

    _run_and_compare(hf_model, mlx_model, feats_np, mask_np)


def test_audio_model_parity_with_calibrated_clip_bounds():
    """The ClippableLinear clamp bounds are checkpoint-specific calibration
    values, not architecture constants -- verify the mlx port actually
    applies whatever finite bounds are loaded (as opposed to silently
    ignoring them or using a hardcoded value)."""
    hf_cfg, mlx_cfg, hf_model, mlx_model = _build_models(
        use_clipped_linears=True, seed=2
    )
    _randomize_clip_bounds(hf_model, mlx_model, seed=2)

    batch, seq_len = 2, 37
    rng = np.random.default_rng(3)
    feats_np = rng.normal(size=(batch, seq_len, MEL_BINS)).astype(np.float32) * 5.0
    mask_np = np.ones((batch, seq_len), dtype=bool)
    mask_np[1, 30:] = False

    _run_and_compare(hf_model, mlx_model, feats_np, mask_np)

    # Sanity: the clamp must actually be doing something observable, i.e.
    # this isn't passing merely because the bounds are wide/inert. Compare
    # against a second mlx model loaded with the *unclamped* defaults on
    # the same inputs/weights and confirm the outputs differ.
    _, unclamped_cfg, _, _ = _build_models(use_clipped_linears=False, seed=2)
    unclamped_model = ga.AudioModel(unclamped_cfg)
    weights = {
        _remap_name(n): _convert_weight(n, t)
        for n, t in hf_model.state_dict().items()
        if not n.split(".")[-1].startswith(
            ("input_min", "input_max", "output_min", "output_max")
        )
    }
    unclamped_model.load_weights(list(weights.items()), strict=False)
    unclamped_hidden, _ = unclamped_model(mx.array(feats_np), mx.array(mask_np))
    hf_clamped_hidden = hf_model(
        torch.tensor(feats_np), torch.tensor(mask_np)
    ).last_hidden_state
    assert (
        np.max(np.abs(hf_clamped_hidden.detach().numpy() - np.array(unclamped_hidden)))
        > 1e-3
    )


def test_audio_model_parity_without_attention_mask():
    """Regression test: even with no padding mask at all, the reference
    still folds the local sliding-window mask into attention via an
    unconditional `and_mask_function`. Skipping mask construction entirely
    when there is no padding (as an earlier version of this port did) lets
    boundary tokens attend outside their true receptive field -- this was
    caught by this exact test (0.14 max abs diff vs. ~1e-7 once fixed)."""
    _, _, hf_model, mlx_model = _build_models(use_clipped_linears=True, seed=7)

    feats_np = (
        np.random.default_rng(8).normal(size=(2, 24, MEL_BINS)).astype(np.float32)
    )
    _run_and_compare(hf_model, mlx_model, feats_np, mask_np=None)


def test_audio_model_parity_seq_len_multiple_of_chunk_size():
    _, _, hf_model, mlx_model = _build_models(use_clipped_linears=True, seed=11)

    chunk = COMMON_CONFIG["attention_chunk_size"]
    feats_np = (
        np.random.default_rng(12)
        .normal(size=(1, chunk * 3, MEL_BINS))
        .astype(np.float32)
    )
    mask_np = np.ones((1, chunk * 3), dtype=bool)
    _run_and_compare(hf_model, mlx_model, feats_np, mask_np)


def test_multimodal_embedder_parity():
    torch.manual_seed(5)
    hf_cfg = Gemma4AudioConfig(**COMMON_CONFIG)
    mlx_cfg = ga.ModelArgs(**COMMON_CONFIG)
    text_hidden_size = 48

    hf_embed = HFMultimodalEmbedder(
        hf_cfg, Gemma4TextConfig(hidden_size=text_hidden_size)
    )
    mlx_embed = ga.MultimodalEmbedder(mlx_cfg, text_hidden_size=text_hidden_size)

    weights = {
        n: mx.array(t.detach().numpy()) for n, t in hf_embed.state_dict().items()
    }
    mlx_embed.load_weights(list(weights.items()), strict=True)

    x_np = (
        np.random.default_rng(6)
        .normal(size=(2, 5, COMMON_CONFIG["output_proj_dims"]))
        .astype(np.float32)
    )
    with torch.no_grad():
        hf_out = hf_embed(torch.tensor(x_np))
    mlx_out = mlx_embed(mx.array(x_np))

    max_abs = np.max(np.abs(hf_out.numpy() - np.array(mlx_out)))
    assert max_abs < 5e-5, max_abs
