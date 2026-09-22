# Copyright © 2025 Apple Inc.

"""Gemma 4 audio tower (Universal Speech Model / Conformer-style encoder).

This mirrors ``transformers.models.gemma4.modeling_gemma4``'s
``Gemma4AudioModel`` and ``Gemma4MultimodalEmbedder`` (audio branch) so
checkpoints can be loaded without any weight renaming beyond what
``Gemma4Model.sanitize`` already does upstream (dropping the ``model.``
prefix). Class and method names intentionally track the HF reference
implementation closely to make the port auditable.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_audio"
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"

    # Sub-sample convolution projection.
    subsampling_conv_channels: List[int] = field(default_factory=lambda: [128, 32])

    # Conformer block parameters.
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9

    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536


class ClippableLinear(nn.Module):
    """A ``nn.Linear`` that optionally clamps its input/output to bounds
    read from the checkpoint (``input_min``/``input_max``/``output_min``/
    ``output_max``).

    These bounds are *not* fixed constants baked into the architecture:
    when ``use_clipped_linears`` is enabled they are per-tensor buffers
    calibrated during training/quantization-aware finetuning and shipped
    in the checkpoint. When absent (``use_clipped_linears=False``) no
    clamping is applied at all. Do not hardcode a clamp value here -- if
    ``use_clipped_linears`` is true but the checkpoint provides no finite
    bounds, the buffers default to +/-inf (i.e. a no-op clamp) exactly like
    the reference implementation.
    """

    def __init__(self, use_clipped_linears: bool, in_features: int, out_features: int):
        super().__init__()
        self.use_clipped_linears = use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.input_min = mx.array(-mx.inf)
            self.input_max = mx.array(mx.inf)
            self.output_min = mx.array(-mx.inf)
            self.output_max = mx.array(mx.inf)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_clipped_linears:
            x = mx.clip(x, self.input_min, self.input_max)

        x = self.linear(x)

        if self.use_clipped_linears:
            x = mx.clip(x, self.output_min, self.output_max)

        return x


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm with an optional (unlearned) scale, matching
    ``Gemma4RMSNorm``: ``mean_squared = mean(x**2) + eps`` then
    ``x * mean_squared**-0.5 [* weight]``, computed in float32."""

    def __init__(self, dims: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight if self.with_scale else None
        return mx.fast.rms_norm(x, weight, self.eps)


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding for the audio encoder.

    Produces position embeddings of shape ``[1, context_size, hidden_size]``
    with concatenated ``[sin..., cos...]`` layout. Purely a function of the
    config (not learned, not present in checkpoints), so it is recomputed
    on every call rather than cached as a module buffer.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size
            + config.attention_context_left
            - 1
            + config.attention_context_right
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * mx.exp(
            mx.arange(num_timescales) * -log_timescale_increment
        )
        inv_timescales = inv_timescales.reshape(1, 1, -1)

        position_ids = mx.arange(self.context_size // 2, -1, -1)
        position_ids = position_ids.reshape(1, -1, 1)
        scaled_time = position_ids.astype(inv_timescales.dtype) * inv_timescales
        pos_embed = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
        return pos_embed.astype(hidden_states.dtype)


def _local_attention_mask(
    output_mask: mx.array,
    chunk_size: int,
    max_past_horizon: int,
    max_future_horizon: int,
) -> mx.array:
    """Builds the boolean ``[B, 1, num_blocks, chunk_size, context_size]``
    mask consumed by :class:`Attention`, mirroring
    ``Gemma4AudioModel._convert_4d_mask_to_blocked_5d`` composed with the
    ``sliding_window_mask_function`` overlay used to build the base 4D
    bidirectional mask.
    """
    B, L = output_mask.shape

    q_idx = mx.arange(L).reshape(-1, 1)
    k_idx = mx.arange(L).reshape(1, -1)
    dist = q_idx - k_idx
    left = (dist >= 0) & (dist < max_past_horizon)
    right = (dist < 0) & ((-dist) < max_future_horizon)
    window = left | right  # (L, L)

    mask = window[None, :, :] & output_mask[:, None, :]  # (B, L, L)
    mask = mask[:, None]  # (B, 1, L, L)

    num_blocks = (L + chunk_size - 1) // chunk_size
    padded_L = num_blocks * chunk_size
    pad_amt = padded_L - L
    mask = mx.pad(
        mask, [(0, 0), (0, 0), (0, pad_amt), (0, pad_amt)], constant_values=False
    )

    context_size = chunk_size + max_past_horizon + max_future_horizon
    mask = mask.reshape(B, 1, num_blocks, chunk_size, padded_L)
    mask = mx.pad(
        mask,
        [(0, 0), (0, 0), (0, 0), (0, 0), (max_past_horizon, max_future_horizon)],
        constant_values=False,
    )

    block_starts = mx.arange(num_blocks) * chunk_size
    offsets = mx.arange(context_size)
    kv_indices = (block_starts.reshape(-1, 1) + offsets.reshape(1, -1)).reshape(
        1, 1, num_blocks, 1, context_size
    )
    kv_indices = mx.broadcast_to(
        kv_indices, (B, 1, num_blocks, chunk_size, context_size)
    )
    return mx.take_along_axis(mask, kv_indices, axis=-1)


class Attention(nn.Module):
    """Chunked local attention with relative position bias."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.attention_invalid_logits_value = config.attention_invalid_logits_value
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log1p(math.e) / math.log(2)

        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.q_proj = ClippableLinear(
            config.use_clipped_linears,
            config.hidden_size,
            self.num_heads * self.head_dim,
        )
        self.k_proj = ClippableLinear(
            config.use_clipped_linears,
            config.hidden_size,
            self.num_heads * self.head_dim,
        )
        self.v_proj = ClippableLinear(
            config.use_clipped_linears,
            config.hidden_size,
            self.num_heads * self.head_dim,
        )
        self.post = ClippableLinear(
            config.use_clipped_linears, config.hidden_size, config.hidden_size
        )

        self.relative_k_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.per_dim_scale = mx.zeros((self.head_dim,))

    def _convert_to_block(self, x: mx.array) -> mx.array:
        """Splits ``[B, L, H, D]`` into non-overlapping ``[B, num_blocks,
        chunk_size, H, D]`` blocks along the sequence dim."""
        B, L, H, D = x.shape
        num_blocks = (L + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - L
        x = mx.pad(x, [(0, 0), (0, pad), (0, 0), (0, 0)])
        return x.reshape(B, num_blocks, self.chunk_size, H, D)

    def _extract_block_context(self, x: mx.array) -> mx.array:
        """Extracts overlapping ``context_size`` windows per block, strided
        by ``chunk_size``, from a ``[B, L, H, D]`` tensor."""
        B, L, H, D = x.shape
        x = mx.pad(
            x,
            [
                (0, 0),
                (self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
                (0, 0),
                (0, 0),
            ],
        )
        num_blocks = (L + self.chunk_size - 1) // self.chunk_size
        block_starts = mx.arange(num_blocks) * self.chunk_size
        offsets = mx.arange(self.context_size)
        indices = block_starts.reshape(-1, 1) + offsets.reshape(1, -1)
        return mx.take(x, indices, axis=1)

    def _rel_shift(self, x: mx.array) -> mx.array:
        """Relative position shift for blocked attention (Transformer-XL
        style), see appendix B of https://huggingface.co/papers/1901.02860."""
        B, H, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = mx.pad(
            x, [(0, 0), (0, 0), (0, 0), (0, 0), (0, context_size + 1 - position_length)]
        )
        x = x.reshape(B, H, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.reshape(B, H, num_blocks, block_size, context_size)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        shape = (B, L, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).astype(mx.float32).reshape(shape)
        key_states = self.k_proj(hidden_states).astype(mx.float32).reshape(shape)
        value_states = self.v_proj(hidden_states).astype(mx.float32).reshape(shape)

        query_states = query_states * self.q_scale * nn.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.reshape(
            -1, self.num_heads, self.head_dim
        )
        relative_key_states = relative_key_states.astype(query_states.dtype)

        queries = query_states.transpose(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.transpose(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(B, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.transpose(1, 2, 0)
        matrix_bd = matrix_bd.reshape(
            B, self.num_heads, num_blocks, self.chunk_size, -1
        )
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.attention_logits_soft_cap
        attn_weights = mx.tanh(attn_weights)
        attn_weights = attn_weights * self.attention_logits_soft_cap

        if attention_mask is not None:
            attn_weights = mx.where(
                attention_mask, attn_weights, self.attention_invalid_logits_value
            )

        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(
            value_states.dtype
        )
        attn_output = attn_weights @ value_states.transpose(0, 3, 1, 2, 4)
        attn_output = attn_output.transpose(0, 2, 3, 1, 4).reshape(
            B, num_blocks * self.chunk_size, -1
        )
        attn_output = attn_output[:, :L]
        attn_output = self.post(attn_output.astype(hidden_states.dtype))

        return attn_output


class SubSampleConvProjectionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_eps: float):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, affine=True, bias=False)

    def __call__(self, hidden_states: mx.array, mask: Optional[mx.array] = None):
        # hidden_states is NHWC: (batch, time, freq, channels).
        if mask is not None:
            hidden_states = hidden_states * mask[:, :, None, None]

        hidden_states = self.conv(hidden_states.astype(self.conv.weight.dtype))
        hidden_states = nn.relu(self.norm(hidden_states))

        if mask is not None:
            mask = mask[:, ::2]

        return hidden_states, mask


class SubSampleConvProjection(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        channels = config.subsampling_conv_channels
        self.layer0 = SubSampleConvProjectionLayer(
            in_channels=1, out_channels=channels[0], norm_eps=config.rms_norm_eps
        )
        self.layer1 = SubSampleConvProjectionLayer(
            in_channels=channels[0],
            out_channels=channels[1],
            norm_eps=config.rms_norm_eps,
        )
        proj_input_dim = (channels[0] // 4) * channels[1]
        self.input_proj_linear = nn.Linear(
            proj_input_dim, config.hidden_size, bias=False
        )

    def __call__(
        self, input_features: mx.array, input_features_mask: Optional[mx.array] = None
    ):
        hidden_states = input_features[..., None]  # (B, L, mel_bins, 1) == NHWC, C=1
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)

        B, L, _, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(B, L, -1)
        return self.input_proj_linear(hidden_states), mask


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.ffw_layer_1 = ClippableLinear(
            config.use_clipped_linears, config.hidden_size, config.hidden_size * 4
        )
        self.ffw_layer_2 = ClippableLinear(
            config.use_clipped_linears, config.hidden_size * 4, config.hidden_size
        )

        self.pre_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = _activation(config.hidden_act)

        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gradient_clipping = min(
            self.gradient_clipping, mx.finfo(hidden_states.dtype).max
        )

        residual = hidden_states
        hidden_states = mx.clip(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = mx.clip(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states = hidden_states * self.post_layer_scale
        hidden_states = hidden_states + residual

        return hidden_states


class CausalConv1d(nn.Module):
    """Depthwise causal 1D conv over an NLC tensor (channels last, matching
    mlx's ``nn.Conv1d`` layout -- unlike the torch reference, no NCL
    transpose is needed anywhere in this file)."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )
        self.left_pad = kernel_size - 1

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.left_pad, 0), (0, 0)])
        return self.conv(x)


class LightConv1d(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.linear_start = ClippableLinear(
            config.use_clipped_linears, config.hidden_size, config.hidden_size * 2
        )
        self.linear_end = ClippableLinear(
            config.use_clipped_linears, config.hidden_size, config.hidden_size
        )
        self.depthwise_conv1d = CausalConv1d(
            config.hidden_size, config.conv_kernel_size
        )

        self.pre_layer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=True
        )
        self.conv_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=True
        )
        self.act_fn = _activation(config.hidden_act)

        self.gradient_clipping = config.gradient_clipping

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = nn.glu(hidden_states, axis=-1)

        hidden_states = self.depthwise_conv1d(hidden_states)

        gradient_clipping = min(
            self.gradient_clipping, mx.finfo(hidden_states.dtype).max
        )
        hidden_states = mx.clip(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.feed_forward1 = FeedForward(config)
        self.feed_forward2 = FeedForward(config)
        self.self_attn = Attention(config, layer_idx)
        self.lconv1d = LightConv1d(config)

        self.norm_pre_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_post_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_out = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_clipping = config.gradient_clipping

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array],
        position_embeddings: mx.array,
    ) -> mx.array:
        gradient_clipping = min(
            self.gradient_clipping, mx.finfo(self.norm_pre_attn.weight.dtype).max
        )

        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = mx.clip(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = mx.clip(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = mx.clip(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states


def _activation(name: str):
    return getattr(nn, name)


class AudioModel(nn.Module):
    """An audio encoder based on the Universal Speech Model architecture,
    mirroring ``transformers.models.gemma4.modeling_gemma4.Gemma4AudioModel``.

    Weight keys (relative to this module) match the HF checkpoint's
    ``audio_tower.*`` weights exactly.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.subsample_conv_projection = SubSampleConvProjection(config)
        self.rel_pos_enc = RelPositionalEncoding(config)
        self.layers = [
            EncoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.output_proj = nn.Linear(
            config.hidden_size, config.output_proj_dims, bias=True
        )

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):
        hidden_states, output_mask = self.subsample_conv_projection(
            input_features, attention_mask
        )
        position_embeddings = self.rel_pos_enc(hidden_states)

        # Even with no padding mask, the reference still applies the local
        # sliding-window mask (it is folded in via an `and_mask_function`
        # unconditionally). The coarse per-block context window extracted
        # by `Attention._extract_block_context` is wider than any single
        # token's true receptive field, so this fine-grained mask must
        # always be built -- skipping it when `output_mask is None` lets
        # boundary tokens attend to positions outside their real window.
        padding_mask = output_mask
        if padding_mask is None:
            padding_mask = mx.ones(hidden_states.shape[:2], dtype=mx.bool_)
        block_mask = _local_attention_mask(
            padding_mask,
            self.config.attention_chunk_size,
            self.config.attention_context_left - 1,
            self.config.attention_context_right,
        )

        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=block_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.output_proj(hidden_states)
        return hidden_states, output_mask

    def sanitize(self, weights: dict) -> dict:
        """Maps real HF `audio_tower.*` checkpoint keys onto this module's
        parameter tree. Almost every key is a verbatim rename (only the
        `model.`/`audio_tower.` prefixes are stripped) -- the two exceptions,
        both verified against the real `Gemma4AudioModel` state dict, are:

        - `Gemma4AudioCausalConv1d` subclasses `nn.Conv1d` directly in torch
          (no wrapping module), so its weight has no extra path component;
          this file wraps it as `CausalConv1d.conv` (mlx's `nn.Conv1d` has no
          built-in causal padding), so `...depthwise_conv1d.weight` becomes
          `...depthwise_conv1d.conv.weight`.
        - Conv weight axis order: torch `Conv2d`/`Conv1d` store weights as
          `(out, in/groups, *kernel)`; mlx's `Conv2d`/`Conv1d` expect
          `(out, *kernel, in/groups)`.
        """
        new_weights = {}
        for k, v in weights.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("audio_tower."):
                nk = nk[len("audio_tower.") :]

            if nk.endswith("depthwise_conv1d.weight"):
                nk = nk.replace(
                    "depthwise_conv1d.weight", "depthwise_conv1d.conv.weight"
                )
                v = v.transpose(0, 2, 1)
            elif nk.endswith(("layer0.conv.weight", "layer1.conv.weight")):
                v = v.transpose(0, 2, 3, 1)

            new_weights[nk] = v
        return new_weights


class MultimodalEmbedder(nn.Module):
    """Embeds audio soft tokens into language model space, mirroring
    ``Gemma4MultimodalEmbedder`` (audio branch). Weight keys (relative to
    this module) match the HF checkpoint's ``embed_audio.*`` weights."""

    def __init__(self, config: ModelArgs, text_hidden_size: int):
        super().__init__()
        multimodal_hidden_size = config.output_proj_dims
        self.embedding_projection = nn.Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = RMSNorm(
            multimodal_hidden_size, eps=config.rms_norm_eps, with_scale=False
        )

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)
