# Copyright © 2026 Apple Inc.

"""Numerical parity test for the top-level multimodal `mlx_lm.models.gemma4`
(text decoder + vision tower + audio tower + fusion) against the real
``transformers`` ``Gemma4ForConditionalGeneration``.

Unlike ``test_gemma4_vision.py`` (which fetches a couple of real layers'
weights from ``google/gemma-4-E4B-it`` over HTTP range requests), this test
builds small, fully-synthetic ``Gemma4Config``/``gemma4.ModelArgs`` pairs
(following ``test_gemma4_audio_transformers_parity.py``'s pattern) and
copies the real HF model's own randomly-initialized weights into the mlx
model via ``gemma4.Model.sanitize()`` -- i.e. this test exercises
``sanitize()`` itself, not a hand-rolled substitute for it, end to end
against a real (if tiny) ``Gemma4ForConditionalGeneration.state_dict()``.

The token ids, sub-config field names, and the overall fusion algorithm
(replace multimodal placeholder ids with ``pad_token_id`` before the main
embedding lookup and before computing per-layer "token identity" inputs,
project each tower's pooled soft tokens through its
``Gemma4MultimodalEmbedder`` (``embed_vision``/``embed_audio``), splice them
into the *scaled* text embedding sequence via a ``masked_scatter``-equivalent
at the placeholder positions, then run the text decoder on the fused
sequence with the precomputed per-layer inputs) are read directly off
``transformers/models/gemma4/{modeling_gemma4,modular_gemma4,
configuration_gemma4}.py`` and a real ``google/gemma-4-E4B-it``
``config.json`` (fetched to confirm real field values: ``image_token_id=
258880``, ``audio_token_id=258881``, ``boi_token_id=255999``,
``eoi_token_id=258882``, ``boa_token_id=256000``,
``eoa_token_index=258883``, ``hidden_size_per_layer_input=256`` (per-layer
inputs are active on this checkpoint), ``enable_moe_block=False`` (E4B has
no MoE layers, so this test's synthetic text config leaves MoE off too --
MoE routing is an orthogonal, pre-existing feature of ``gemma4_text.py`` not
touched by this task)).

Run explicitly with an environment that has ``torch`` + a ``transformers``
build with Gemma4 support, e.g.:

    pytest tests/test_gemma4_multimodal_parity.py -v

Comparisons run with mlx on CPU (``mx.set_default_device(mx.cpu)``) for the
same reason as the other two Gemma4 parity tests: to avoid conflating
ordinary Metal-vs-CPU float32 kernel non-determinism with correctness.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig,
        Gemma4Config,
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration as HFGemma4,
    )
except ImportError:
    pytest.skip(
        "installed `transformers` has no Gemma4 support", allow_module_level=True
    )

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import gemma4

mx.set_default_device(mx.cpu)


# ---- Real token ids (verified against a real google/gemma-4-E4B-it
# config.json), reused verbatim so the mask/splice logic under test runs
# against the same ids the real config uses -- only the *values assigned to
# the small synthetic vocab* below are shrunk, not the ids themselves. ----
IMAGE_TOKEN_ID = 258_880
AUDIO_TOKEN_ID = 258_881
BOI_TOKEN_ID = 255_999
EOI_TOKEN_ID = 258_882
BOA_TOKEN_ID = 256_000
EOA_TOKEN_INDEX = 258_883
PAD_TOKEN_ID = 0
VOCAB_SIZE = 258_900  # just past the highest special id used above

TEXT_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=12,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    global_head_dim=16,
    hidden_size_per_layer_input=8,
    vocab_size_per_layer_input=VOCAB_SIZE,
    num_kv_shared_layers=6,
    use_double_wide_mlp=False,
    enable_moe_block=False,
    tie_word_embeddings=True,
    final_logit_softcapping=30.0,
    pad_token_id=PAD_TOKEN_ID,
    # Real E4B's pattern is 5 sliding + 1 full, repeated; two repeats here
    # keeps enough layers for the kv-sharing region (see num_kv_shared_layers
    # above) to include both layer types before the shared region starts.
    layer_types=["sliding_attention"] * 5
    + ["full_attention"]
    + ["sliding_attention"] * 5
    + ["full_attention"],
)
VISION_CONFIG = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    patch_size=4,
    pooling_kernel_size=2,
    position_embedding_size=16,
    use_clipped_linears=False,  # clipping itself is covered by test_gemma4_vision.py
)
AUDIO_CONFIG = dict(
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    subsampling_conv_channels=[4, 2],
    conv_kernel_size=3,
    attention_chunk_size=4,
    attention_context_left=3,
    attention_context_right=0,
    output_proj_dims=24,
    use_clipped_linears=False,  # clipping itself is covered by test_gemma4_audio_transformers_parity.py
)
MEL_BINS = AUDIO_CONFIG["subsampling_conv_channels"][0]


def _build_models(seed: int):
    torch.manual_seed(seed)
    hf_config = Gemma4Config(
        text_config=Gemma4TextConfig(**TEXT_CONFIG),
        vision_config=Gemma4VisionConfig(**VISION_CONFIG),
        audio_config=Gemma4AudioConfig(**AUDIO_CONFIG),
        image_token_id=IMAGE_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        boi_token_id=BOI_TOKEN_ID,
        eoi_token_id=EOI_TOKEN_ID,
        boa_token_id=BOA_TOKEN_ID,
        eoa_token_index=EOA_TOKEN_INDEX,
        tie_word_embeddings=True,
    )
    hf_model = HFGemma4(hf_config)
    hf_model.eval()

    mlx_args = gemma4.ModelArgs(
        text_config=dict(TEXT_CONFIG),
        vision_config=dict(VISION_CONFIG),
        audio_config=dict(AUDIO_CONFIG),
        vocab_size=VOCAB_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        boi_token_id=BOI_TOKEN_ID,
        eoi_token_id=EOI_TOKEN_ID,
        boa_token_id=BOA_TOKEN_ID,
        eoa_token_index=EOA_TOKEN_INDEX,
    )
    mlx_model = gemma4.Model(mlx_args)

    # Real safetensors checkpoints store tensors in torch's native layout
    # (e.g. conv weights as (out, in/groups, *kernel)); `sanitize()` is
    # responsible for any layout conversion, so this only does the
    # tensor->mx.array conversion, nothing model-specific. `lm_head.weight`
    # is dropped: HF's in-memory state_dict includes the tied lm_head
    # weight, but `sanitize()` already (defensively) drops it too -- this
    # mirrors what a real, deduped safetensors checkpoint looks like.
    raw_weights = {
        n: mx.array(t.detach().numpy())
        for n, t in hf_model.state_dict().items()
        if n != "lm_head.weight"
    }
    sanitized = mlx_model.sanitize(raw_weights)

    # Confirm sanitize() produces *exactly* the mlx model's parameter tree
    # (this is the crux of "sanitize() must actually load these weights,
    # not just avoid crashing" -- a silent subset/superset would pass
    # `strict=False` loading without ever being noticed).
    mlx_param_keys = set(dict(tree_flatten(mlx_model.parameters())))
    assert mlx_param_keys - set(sanitized) == set(), "sanitize() is missing keys"
    assert set(sanitized) - mlx_param_keys == set(), "sanitize() has extra keys"

    mlx_model.load_weights(list(sanitized.items()), strict=True)
    return hf_model, mlx_model


def _compare_logits(hf_model, mlx_model, kwargs_torch, kwargs_mlx, atol=1e-5):
    with torch.no_grad():
        hf_logits = hf_model(**kwargs_torch).logits.numpy()

    mlx_logits = mlx_model(**kwargs_mlx)
    mx.eval(mlx_logits)
    mlx_logits_np = np.array(mlx_logits)

    assert hf_logits.shape == mlx_logits_np.shape
    max_abs = np.max(np.abs(hf_logits - mlx_logits_np))
    assert max_abs < atol, f"max abs logit diff {max_abs} exceeds {atol}"
    return max_abs


def test_text_only_parity():
    """(a) Text-only forward pass: no pixel_values/input_features at all --
    exercises the "unaffected by multimodal support" backward-compat path,
    plus the real per-layer-input (PLE) pipeline with `hidden_size_per_layer_
    input=8` (E4B has this active in real life), computed the ordinary way
    (reverse-embedding is never needed since there's no splicing)."""
    hf_model, mlx_model = _build_models(seed=1)
    rng = np.random.default_rng(0)
    input_ids_np = rng.integers(1, 200, size=(1, 9)).astype(np.int64)

    max_abs = _compare_logits(
        hf_model,
        mlx_model,
        dict(input_ids=torch.tensor(input_ids_np)),
        dict(inputs=mx.array(input_ids_np)),
    )
    print(f"text-only max abs diff: {max_abs}")


def test_text_image_parity():
    """(b) Text+image forward pass, no padding: a full 4x4 patch grid
    (pooling_kernel_size=2 -> 4 pooled soft tokens), spliced at
    `image_token_id` positions bracketed by boi/eoi (boi/eoi are ordinary
    text tokens here, not treated specially by the model -- only
    `image_token_id`/`audio_token_id` positions get spliced)."""
    hf_model, mlx_model = _build_models(seed=2)
    rng = np.random.default_rng(1)

    grid = 4
    n_patches = grid * grid
    patch_dim = 3 * VISION_CONFIG["patch_size"] ** 2
    pixel_values_np = rng.uniform(0, 1, size=(1, n_patches, patch_dim)).astype(
        np.float32
    )
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid), indexing="xy")
    pos_np = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.int64)[None]
    n_soft_tokens = n_patches // (VISION_CONFIG["pooling_kernel_size"] ** 2)

    prefix = rng.integers(1, 200, size=(3,)).astype(np.int64)
    suffix = rng.integers(1, 200, size=(3,)).astype(np.int64)
    input_ids_np = np.concatenate(
        [prefix, [BOI_TOKEN_ID], [IMAGE_TOKEN_ID] * n_soft_tokens, [EOI_TOKEN_ID], suffix]
    )[None].astype(np.int64)

    max_abs = _compare_logits(
        hf_model,
        mlx_model,
        dict(
            input_ids=torch.tensor(input_ids_np),
            pixel_values=torch.tensor(pixel_values_np),
            image_position_ids=torch.tensor(pos_np),
        ),
        dict(
            inputs=mx.array(input_ids_np),
            pixel_values=mx.array(pixel_values_np),
            pixel_position_ids=mx.array(pos_np),
        ),
    )
    print(f"text+image max abs diff: {max_abs}")


def test_text_image_parity_with_padding():
    """(b), padding variant: pads a 4x4 real patch grid up to a 6x6 grid
    with (-1,-1) padding patches, so the pooler's `valid_mask` compaction
    (dropping partially/fully padding pooled slots before splicing) is
    actually exercised rather than trivially a no-op."""
    hf_model, mlx_model = _build_models(seed=4)
    rng = np.random.default_rng(3)

    grid, padded_grid = 4, 6
    n_real = grid * grid
    padded_total = padded_grid * padded_grid
    patch_dim = 3 * VISION_CONFIG["patch_size"] ** 2

    pixel_values_np = np.zeros((1, padded_total, patch_dim), dtype=np.float32)
    pixel_values_np[:, :n_real] = rng.uniform(0, 1, size=(1, n_real, patch_dim)).astype(
        np.float32
    )
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid), indexing="xy")
    real_pos = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.int64)
    pos_np = -np.ones((1, padded_total, 2), dtype=np.int64)
    pos_np[0, :n_real] = real_pos

    with torch.no_grad():
        hf_vis = hf_model.model.get_image_features(
            torch.tensor(pixel_values_np), torch.tensor(pos_np), return_dict=True
        )
    n_valid = hf_vis.pooler_output[0].shape[0]
    assert 0 < n_valid <= padded_total // (VISION_CONFIG["pooling_kernel_size"] ** 2)

    input_ids_np = np.concatenate(
        [[10, 11], [BOI_TOKEN_ID], [IMAGE_TOKEN_ID] * n_valid, [EOI_TOKEN_ID], [12, 13]]
    )[None].astype(np.int64)

    max_abs = _compare_logits(
        hf_model,
        mlx_model,
        dict(
            input_ids=torch.tensor(input_ids_np),
            pixel_values=torch.tensor(pixel_values_np),
            image_position_ids=torch.tensor(pos_np),
        ),
        dict(
            inputs=mx.array(input_ids_np),
            pixel_values=mx.array(pixel_values_np),
            pixel_position_ids=mx.array(pos_np),
        ),
    )
    print(f"text+image(padded) max abs diff: {max_abs}")


def test_text_audio_parity():
    """(c) Text+audio forward pass: a short audio clip (not a multiple of
    `attention_chunk_size=4`, matching test_gemma4_audio_transformers_parity
    .py's convention of always covering a non-aligned length) spliced at
    `audio_token_id` positions bracketed by boa/eoa."""
    hf_model, mlx_model = _build_models(seed=3)
    rng = np.random.default_rng(2)

    seq_len_audio = 10
    feats_np = rng.normal(size=(1, seq_len_audio, MEL_BINS)).astype(np.float32)
    mask_np = np.ones((1, seq_len_audio), dtype=bool)

    def _conv_out_len(length):
        return (length + 1) // 2

    n_soft_tokens = _conv_out_len(_conv_out_len(seq_len_audio))

    prefix = rng.integers(1, 200, size=(3,)).astype(np.int64)
    suffix = rng.integers(1, 200, size=(3,)).astype(np.int64)
    input_ids_np = np.concatenate(
        [prefix, [BOA_TOKEN_ID], [AUDIO_TOKEN_ID] * n_soft_tokens, [EOA_TOKEN_INDEX], suffix]
    )[None].astype(np.int64)

    max_abs = _compare_logits(
        hf_model,
        mlx_model,
        dict(
            input_ids=torch.tensor(input_ids_np),
            input_features=torch.tensor(feats_np),
            input_features_mask=torch.tensor(mask_np),
        ),
        dict(
            inputs=mx.array(input_ids_np),
            input_features=mx.array(feats_np),
            input_features_mask=mx.array(mask_np),
        ),
        atol=2e-5,  # audio's own parity test uses 5e-5; the extra decoder
        # layers on top compound float32 rounding slightly further.
    )
    print(f"text+audio max abs diff: {max_abs}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
