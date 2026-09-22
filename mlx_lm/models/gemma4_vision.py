# Copyright © 2025 Apple Inc.

"""Gemma 4 vision tower (`google/gemma-4-E4B-it`), standalone.

This implements *only* the vision encoder (patch embedding -> transformer
encoder -> spatial pooling), matching the real `Gemma4VisionModel` from
`transformers.models.gemma4.modeling_gemma4` (as of the `google/gemma-4-E4B-it`
release, April 2026). It does not wire into a top-level multimodal model
(text/vision fusion, the multimodal embedder, soft-token scatter into the
text sequence) -- that belongs to a separate later task. `gemma4.py`'s
`Model.sanitize()` still deliberately drops all `vision_tower.*` keys.

What is **verified** (read directly from the real transformers source at
`transformers/models/gemma4/{modeling_gemma4,configuration_gemma4}.py`, and
from the real `google/gemma-4-E4B-it` checkpoint's `model.safetensors`
header / a handful of tensor values fetched via HTTP range requests,
*not* a full 16GB download):

- Config field names/values/defaults (`Gemma4VisionConfig`): hidden_size=768,
  intermediate_size=3072, num_hidden_layers=16, num_attention_heads=
  num_key_value_heads=12 (plain MHA, not GQA), head_dim=64, patch_size=16,
  hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6,
  position_embedding_size=10240, pooling_kernel_size=3,
  use_clipped_linears=True, standardize=False.
- `rope_parameters={"rope_theta": 100.0, "rope_type": "default"}` gets
  normalized to `rope_type="axial"` by `Gemma4VisionConfig.default_rope_type
  = "axial"` (see `modeling_rope_utils.py`'s generic "default" -> per-model
  default substitution) before reaching `Gemma4VisionRotaryEmbedding`, which
  hard-requires `rope_type == "axial"`. So this file always uses axial rope.
- Every `*_proj` in vision attention/MLP is a `Gemma4ClippableLinear`:
  clamp(input) -> `nn.Linear` (no bias) -> clamp(output). For the real
  `google/gemma-4-E4B-it` checkpoint these clamp bounds are REAL, FINITE,
  per-layer values (verified by reading a few `input_min`/`input_max`/
  `output_min`/`output_max` scalar tensors straight out of layers 0, 1, and
  15 of the real `model.safetensors` -- e.g. layer 0's `self_attn.q_proj`
  has `input_min=-6.375, input_max=6.3125, output_min=-11.3125,
  output_max=11.1875`; layer 15's differ). They are NOT `+-inf` no-ops, so
  they are implemented faithfully below, not skipped.
- Forward pass order (`Gemma4VisionEncoderLayer.forward`): sandwich norm --
  `input_layernorm` before attention, `post_attention_layernorm` on the
  attention output before the residual add; `pre_feedforward_layernorm`
  before the MLP, `post_feedforward_layernorm` on the MLP output before the
  residual add. Attention itself: q/k/v projections -> per-head RMSNorm
  (`q_norm`/`k_norm` have a learnable scale, `v_norm` does not, matching the
  checkpoint: there is no `v_norm.weight` tensor) -> axial RoPE on q and k
  only -> scaled-dot-product attention with `scale=1.0` (not `1/sqrt(d)` --
  read directly off `Gemma4VisionAttention.scaling = 1.0`) -> `o_proj`.
- Patch embedding (`Gemma4VisionPatchEmbedder`): patches arrive
  pre-flattened as `(batch, num_patches, 3 * patch_size**2)` (this file
  takes the same pre-patchified input -- turning raw images into patches is
  an image-processor concern, out of scope here). Pixel values are rescaled
  `2 * (x - 0.5)` (no mean/std normalization), projected with
  `input_proj` (`Linear(768, 768)`, no bias), then a *2D* learned position
  embedding is added: `position_embedding_table` has shape
  `(2, position_embedding_size, hidden_size)`, where index 0 is looked up by
  the patch's x-coordinate and index 1 by its y-coordinate, and the two
  looked-up vectors are summed. Negative (padding) coordinates are clamped
  to 0 for the lookup, then the result is zeroed for those padding patches.
- Axial RoPE (`Gemma4VisionRotaryEmbedding` + `apply_multidimensional_rope`
  with `ndim=2`): `head_dim=64` is split into two 32-wide halves. The first
  half is rotated using 16 frequencies (`theta=100`) times the patch's
  x-coordinate; the second half using the same 16 frequencies times the
  y-coordinate. Each half uses the standard "rotate_half" formula
  independently. Because `2 * 32 == head_dim`, the whole head is rotated
  (no leftover, un-rotated tail).
- Pooling (`Gemma4VisionPooler`): patches are zeroed at padding positions,
  then (if the encoder's patch count doesn't already match
  `output_length = num_patches // pooling_kernel_size**2`) average-pooled
  into `pooling_kernel_size x pooling_kernel_size` spatial blocks keyed by
  `pixel_position_ids`, then scaled by `sqrt(hidden_size)` in float32.
  `standardize=False` for this checkpoint, so the optional
  `std_bias`/`std_scale` affine step is skipped (would be a no-op learned
  bias/scale otherwise; only allocated when `config.standardize` is True).

What is a **deliberate scope decision, not a guess**: the real
`Gemma4VisionModel.forward` finishes by boolean-mask-selecting valid pooled
tokens across the whole batch (`hidden_states[pooler_mask]`), which flattens
variable numbers of soft tokens per image into one packed sequence for the
downstream multimodal embedder/fusion step. That packing is a property of
how the *fusion* layer wants its input, not of the vision tower's own math,
so `VisionModel.__call__` here returns `(pooled_hidden_states, valid_mask)`
with the batch dimension intact instead, and leaves the packing/statement to
whatever later task wires this into the full multimodal model.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    patch_size: int = 16
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_parameters: Optional[dict] = None
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10 * 1024
    use_clipped_linears: bool = False
    standardize: bool = False

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 100.0, "rope_type": "axial"}
        # Gemma4VisionConfig.default_rope_type == "axial": any "default"
        # rope_type coming from a raw HF config.json is normalized to axial
        # before it ever reaches the rotary embedding. This is the only rope
        # flavor the real vision rotary embedding implements.
        if self.rope_parameters.get("rope_type", "axial") == "default":
            self.rope_parameters["rope_type"] = "axial"


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale (used for `self_attn.v_norm`, which
    has no corresponding weight tensor in the checkpoint)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class ClippableLinear(nn.Module):
    """Mirrors `Gemma4ClippableLinear`: clamp(input) -> Linear -> clamp(output).

    The clamp bounds are real per-tensor scalars stored in the checkpoint
    (`*.input_min`, `*.input_max`, `*.output_min`, `*.output_max`); they are
    only allocated (and applied) when `use_clipped_linears` is set, matching
    the real module.
    """

    def __init__(self, in_features: int, out_features: int, use_clipped_linears: bool):
        super().__init__()
        self.use_clipped_linears = use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.input_min = mx.array(-mx.inf, dtype=mx.float32)
            self.input_max = mx.array(mx.inf, dtype=mx.float32)
            self.output_min = mx.array(-mx.inf, dtype=mx.float32)
            self.output_max = mx.array(mx.inf, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_clipped_linears:
            x = mx.clip(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = mx.clip(x, self.output_min, self.output_max)
        return x


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_axial_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Applies 2D axial RoPE to `x` (..., num_heads, head_dim).

    `cos`/`sin` have shape (..., head_dim) and are laid out as
    `[freq_x, freq_x, freq_y, freq_y]` (each block `head_dim // 4` wide, see
    `Gemma4VisionRotaryEmbedding.recomposition_frequencies`). This matches
    `apply_multidimensional_rope(..., ndim=2)`: the head dim is split into
    two equal halves, the first rotated using the x-frequencies, the second
    using the y-frequencies, each with the standard rotate-half formula.
    """
    half = x.shape[-1] // 2
    cos = mx.expand_dims(cos, axis=-2)
    sin = mx.expand_dims(sin, axis=-2)
    x1, x2 = x[..., :half], x[..., half:]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    y1 = x1 * c1 + _rotate_half(x1) * s1
    y2 = x2 * c2 + _rotate_half(x2) * s2
    return mx.concatenate([y1, y2], axis=-1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        # Verified: Gemma4VisionAttention.scaling = 1.0 (not 1/sqrt(head_dim)).
        self.scale = 1.0

        self.q_proj = ClippableLinear(
            dim, self.n_heads * self.head_dim, args.use_clipped_linears
        )
        self.k_proj = ClippableLinear(
            dim, self.n_kv_heads * self.head_dim, args.use_clipped_linears
        )
        self.v_proj = ClippableLinear(
            dim, self.n_kv_heads * self.head_dim, args.use_clipped_linears
        )
        self.o_proj = ClippableLinear(
            self.n_heads * self.head_dim, dim, args.use_clipped_linears
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        q = _apply_axial_rope(q, cos, sin)
        q = q.transpose(0, 2, 1, 3)

        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_axial_rope(k, cos, sin)
        k = k.transpose(0, 2, 1, 3)

        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_norm(v)
        v = v.transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention(
            q, k, v, cache=None, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_activation == "gelu_pytorch_tanh", (
            "Only the tanh-approximate GELU used by the real checkpoint is "
            f"implemented, got {args.hidden_activation!r}."
        )
        self.gate_proj = ClippableLinear(
            args.hidden_size, args.intermediate_size, args.use_clipped_linears
        )
        self.up_proj = ClippableLinear(
            args.hidden_size, args.intermediate_size, args.use_clipped_linears
        )
        self.down_proj = ClippableLinear(
            args.intermediate_size, args.hidden_size, args.use_clipped_linears
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin, mask)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h
        return x


class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [EncoderLayer(args) for _ in range(args.num_hidden_layers)]

        head_dim = args.head_dim
        theta = args.rope_parameters["rope_theta"]
        # Verified: Gemma4VisionRotaryEmbedding.compute_axial_rope_parameters:
        # spatial_dim = head_dim // 2; inv_freq over arange(0, spatial_dim, 2).
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta ** (mx.arange(0, spatial_dim, 2, dtype=mx.float32) / spatial_dim)
        )
        self._inv_freq = inv_freq

    def _rope_cos_sin(self, pixel_position_ids: mx.array):
        # pixel_position_ids: (B, N, 2) int, last dim is (x, y).
        pos = pixel_position_ids.astype(mx.float32)[..., None]  # (B, N, 2, 1)
        freqs = pos * self._inv_freq  # (B, N, 2, head_dim // 4)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        # Verified: recomposition_frequencies interleaves as [x, x, y, y].
        cos = mx.concatenate(
            [cos[..., 0, :], cos[..., 0, :], cos[..., 1, :], cos[..., 1, :]], axis=-1
        )
        sin = mx.concatenate(
            [sin[..., 0, :], sin[..., 0, :], sin[..., 1, :], sin[..., 1, :]], axis=-1
        )
        return cos, sin

    def __call__(
        self,
        x: mx.array,
        pixel_position_ids: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        cos, sin = self._rope_cos_sin(pixel_position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x


class PatchEmbedder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.patch_size = args.patch_size
        self.position_embedding_size = args.position_embedding_size
        self.input_proj = nn.Linear(
            3 * self.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = mx.zeros(
            (2, self.position_embedding_size, self.hidden_size)
        )

    def __call__(
        self,
        pixel_values: mx.array,
        pixel_position_ids: mx.array,
        padding_positions: mx.array,
    ) -> mx.array:
        # Verified: no mean/std normalization here, just a fixed rescale.
        pixel_values = 2.0 * (pixel_values.astype(mx.float32) - 0.5)
        hidden_states = self.input_proj(pixel_values.astype(self.input_proj.weight.dtype))

        clamped = mx.maximum(pixel_position_ids, 0)
        x_emb = mx.take(self.position_embedding_table[0], clamped[..., 0], axis=0)
        y_emb = mx.take(self.position_embedding_table[1], clamped[..., 1], axis=0)
        position_embeddings = x_emb + y_emb
        position_embeddings = mx.where(
            padding_positions[..., None], mx.zeros_like(position_embeddings), position_embeddings
        )
        return hidden_states + position_embeddings


class Pooler(nn.Module):
    """Spatial average-pooling + sqrt(hidden_size) scaling. No learnable
    parameters of its own (the optional `standardize` affine lives on the
    parent `VisionModel`, matching the checkpoint layout)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self, hidden_states: mx.array, pixel_position_ids: mx.array, length: int
    ):
        input_seq_len = hidden_states.shape[1]
        # Verified: matches `int((input_seq_len // length) ** 0.5)` exactly
        # (integer-divide first, then sqrt), not `(input_seq_len / length) ** 0.5`.
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k * k
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: "
                f"{k=}^2 times {length=} must equal {input_seq_len}."
            )

        clamped = mx.maximum(pixel_position_ids, 0)
        max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel_x = clamped[..., 0] // k
        kernel_y = clamped[..., 1] // k
        kernel_idx = kernel_x + (max_x // k) * kernel_y  # (B, N)

        one_hot = (
            kernel_idx[..., None] == mx.arange(length, dtype=kernel_idx.dtype)
        ).astype(mx.float32)
        weights = one_hot / k_squared  # (B, N, length)
        output = mx.matmul(
            mx.transpose(weights, (0, 2, 1)), hidden_states.astype(mx.float32)
        )
        valid_mask = mx.any(one_hot != 0, axis=1)  # (B, length) True == valid
        return output.astype(hidden_states.dtype), valid_mask

    def __call__(
        self,
        hidden_states: mx.array,
        pixel_position_ids: mx.array,
        padding_positions: mx.array,
        output_length: int,
    ):
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) "
                f"than there are patches ({hidden_states.shape[1]})."
            )

        hidden_states = mx.where(
            padding_positions[..., None], mx.zeros_like(hidden_states), hidden_states
        )

        if hidden_states.shape[1] != output_length:
            hidden_states, valid_mask = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        else:
            valid_mask = mx.logical_not(padding_positions)

        hidden_states = hidden_states.astype(mx.float32) * self.root_hidden_size
        return hidden_states, valid_mask


class VisionModel(nn.Module):
    """The Gemma 4 vision encoder: patchify -> encode -> pool.

    Does not include the multimodal embedder/projection into text space, and
    does not pack variable-length per-image soft tokens across a batch --
    see the module docstring for why.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_embedder = PatchEmbedder(args)
        self.encoder = Encoder(args)
        self.pooler = Pooler(args)
        if args.standardize:
            self.std_bias = mx.zeros((args.hidden_size,))
            self.std_scale = mx.ones((args.hidden_size,))

    def __call__(self, pixel_values: mx.array, pixel_position_ids: mx.array):
        """
        Args:
            pixel_values: (batch, num_patches, 3 * patch_size**2) pre-patchified
                pixel values (patchifying raw images is the image processor's job).
            pixel_position_ids: (batch, num_patches, 2) (x, y) patch coordinates;
                padding patches are (-1, -1).

        Returns:
            (pooled_hidden_states, valid_mask): pooled_hidden_states has shape
            (batch, num_patches // pooling_kernel_size**2, hidden_size) in
            float32 (matching the real pooler's float32 output before the
            caller casts/standardizes/packs it); valid_mask (batch, ...) is
            True where a pooled token is not purely padding.
        """
        padding_positions = mx.all(pixel_position_ids == -1, axis=-1)
        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        valid = mx.logical_not(padding_positions)
        if mx.any(padding_positions).item():
            # Additive key-padding mask, broadcastable to (B, H, L, S).
            mask = mx.where(valid[:, None, None, :], 0.0, -mx.inf).astype(
                inputs_embeds.dtype
            )
        else:
            mask = None

        hidden_states = self.encoder(inputs_embeds, pixel_position_ids, mask=mask)

        pooling_kernel_size = self.args.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size**2)
        hidden_states, valid_mask = self.pooler(
            hidden_states, pixel_position_ids, padding_positions, output_length
        )

        if self.args.standardize:
            hidden_states = (hidden_states - self.std_bias.astype(mx.float32)) * (
                self.std_scale.astype(mx.float32)
            )
        hidden_states = hidden_states.astype(inputs_embeds.dtype)

        return hidden_states, valid_mask

    def sanitize(self, weights: dict) -> dict:
        """Maps real HF `vision_tower.*` checkpoint keys onto this module's
        parameter tree. This is a 1:1 rename (strip the `model.` and
        `vision_tower.` prefixes only) -- every remaining path component
        (`patch_embedder.input_proj.weight`,
        `encoder.layers.{i}.self_attn.q_proj.linear.weight`,
        `encoder.layers.{i}.self_attn.q_proj.input_max`, ...) is copied
        verbatim from the real `model.safetensors` header, with no invented
        keys.
        """
        new_weights = {}
        for k, v in weights.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("vision_tower."):
                nk = nk[len("vision_tower.") :]
            new_weights[nk] = v
        return new_weights
