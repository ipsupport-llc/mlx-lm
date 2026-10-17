# Vision tower for Muse Glimmer (muse_glimmer) image inputs.
#
# Ported from transformers 5.x modeling_muse_glimmer.py (Qwen2.5-VL-style
# windowed attention + dynamic-resolution packing) and the shared
# transformers.vision_utils helpers it calls (get_vision_cu_seqlens,
# get_vision_position_ids, get_vision_window_index,
# get_vision_interpolation_indices_and_weights). Those helpers are plain
# index/weight math over grid_thw + config scalars (no model weights), so
# they are reimplemented here directly in MLX/NumPy rather than imported.

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str = "muse_glimmer"
    hidden_size: int = 1536
    intermediate_size: int = 4224
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    hidden_act: str = "gelu_pytorch_tanh"
    patch_size: int = 16
    patch_temporal: int = 1
    merge_size: int = 2
    pos_emb_height: int = 64
    pos_emb_width: int = 64
    layer_norm_eps: float = 1e-5
    layer_types: list = field(default_factory=list)
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, params: dict) -> "VisionConfig":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in params.items() if k in names})


def get_cu_seqlens(grid_thw: np.ndarray) -> mx.array:
    """Cumulative segment boundaries, one segment per frame (grid_thw rows,
    each repeated `t` times)."""
    lengths = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return mx.array(np.pad(np.cumsum(lengths), (1, 0)).astype(np.int32))


def get_position_ids(grid_thw: np.ndarray) -> np.ndarray:
    """(total_tokens, 2) raster-order (h, w) position ids, un-merged
    (`spatial_merge_size=1` -- Glimmer defers the merge to pixel_shuffle)."""
    out = []
    for t, h, w in grid_thw:
        hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pos = np.stack([hpos.flatten(), wpos.flatten()], axis=-1)
        out.append(np.tile(pos, (t, 1)))
    return np.concatenate(out, axis=0)


def get_window_index(grid_thw: np.ndarray, window_size: int, patch_size: int) -> tuple[mx.array, mx.array]:
    """Qwen2.5-VL-style window reorder: `window_index` gathers tokens into
    window-contiguous order, `cu_window_seqlens` are the resulting window
    boundaries (`spatial_merge_size=1` here too, so `vit_merger_window_size`
    is just `window_size // patch_size`)."""
    win = window_size // patch_size
    window_index, cu_window_seqlens, next_id = [], [0], 0
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        index = np.arange(t * h * w).reshape(t, h, w)
        pad_h, pad_w = (-h) % win, (-w) % win
        nh, nw = (h + pad_h) // win, (w + pad_w) // win
        padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        padded = padded.reshape(t, nh, win, nw, win).transpose(0, 1, 3, 2, 4).reshape(t, nh * nw, win, win)
        seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
        flat = padded.reshape(-1)
        window_index.append(flat[flat != -100] + next_id)
        cu_window_seqlens.extend((np.cumsum(seqlens) + cu_window_seqlens[-1]).tolist())
        next_id += t * h * w
    window_index = np.concatenate(window_index)
    cu_window_seqlens = np.array(sorted(set(cu_window_seqlens)), dtype=np.int32)
    return mx.array(window_index), mx.array(cu_window_seqlens)


def _axis_taps_weights(index: np.ndarray, size: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear taps/weights for one axis, `align_corners=False,
    padding="zeros"` (`F.grid_sample`'s convention, which is what Glimmer's
    position-embedding interpolation uses)."""
    src = (index.astype(np.float64) + 0.5) * side / size - 0.5
    floor = np.floor(src)
    taps = np.stack([floor, floor + 1], axis=-1)
    distance = np.abs(src[:, None] - taps)
    weights = np.clip(1 - distance, 0, None)
    in_range = (taps >= 0) & (taps <= side - 1)
    weights = weights * in_range
    taps = np.clip(taps, 0, side - 1).astype(np.int64)
    return taps, weights


def get_interpolation_indices_weights(grid_thw: np.ndarray, side: int) -> tuple[mx.array, mx.array]:
    """Per-patch gather indices/weights resampling the learned `side x side`
    position-embedding table to each image's `(h, w)` grid (raster order,
    `spatial_merge_size=1`): `F.grid_sample(mode="bilinear",
    align_corners=False, padding_mode="zeros")`, as 4 taps + weights."""
    indices_all, weights_all = [], []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        rows = np.repeat(np.arange(h), w)
        cols = np.tile(np.arange(w), h)
        h_taps, h_w = _axis_taps_weights(rows, np.full(h * w, h), side)
        w_taps, w_w = _axis_taps_weights(cols, np.full(h * w, w), side)
        idx = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
        wt = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        if t > 1:
            idx, wt = np.tile(idx, (t, 1)), np.tile(wt, (t, 1))
        indices_all.append(idx)
        weights_all.append(wt)
    return mx.array(np.concatenate(indices_all)), mx.array(np.concatenate(weights_all))


def get_pixel_shuffle_index(grid_thw: np.ndarray, merge_size: int) -> mx.array:
    """Gather index that groups each `merge_size x merge_size` spatial block
    into `merge_size**2` consecutive rows, for `pixel_shuffle` to fold into
    the channel dim."""
    indices, offset = [], 0
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        perm = np.arange(h * w).reshape(h // merge_size, merge_size, w // merge_size, merge_size)
        perm = perm.transpose(0, 2, 1, 3).reshape(-1)
        if t > 1:
            perm = (perm[None] + np.arange(t)[:, None] * h * w).reshape(-1)
        indices.append(perm + offset)
        offset += t * h * w
    return mx.array(np.concatenate(indices))


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, (0, 2))
    sin = mx.expand_dims(sin, (0, 2))
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class VisionRotaryEmbedding(nn.Module):
    """Axial 2D rope: the same `head_dim//4` frequency table looked up at
    each token's (h, w) position, concatenated and duplicated to cover the
    full head dim (Glimmer's `recomposition_frequencies`)."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.spatial_dim = head_dim // 2
        self.theta = theta

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # Computed fresh each call, not stored: an mx.array module attribute
        # is treated as a loadable parameter, and this isn't one.
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.spatial_dim, 2, dtype=mx.float32) / self.spatial_dim))
        freqs = position_ids[..., None].astype(mx.float32) * inv_freq
        freq_h, freq_w = freqs[:, 0], freqs[:, 1]
        freq_hw = mx.concatenate([freq_h, freq_w], axis=-1)
        angles = mx.concatenate([freq_hw, freq_hw], axis=-1)
        return mx.cos(angles), mx.sin(angles)


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        in_dim = config.patch_temporal * 3 * config.patch_size**2
        self.patch_embedding = nn.Linear(in_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )
        self.num_grid_per_side = config.pos_emb_height

    def __call__(self, pixel_values: mx.array, grid_thw: np.ndarray) -> mx.array:
        embeddings = self.patch_embedding(pixel_values)
        indices, weights = get_interpolation_indices_weights(grid_thw, self.num_grid_per_side)
        pos = self.position_embedding_table(indices) * weights[:, :, None]
        return embeddings + pos.sum(axis=1).astype(embeddings.dtype)


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array, cu_seqlens: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        seq_len = x.shape[0]
        shape = (1, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).reshape(shape)
        k = self.k_proj(x).reshape(shape)
        v = self.v_proj(x).reshape(shape)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))

        bounds = cu_seqlens[1:-1].tolist()
        splits = [mx.split(t, bounds, axis=2) for t in (q, k, v)]
        out = [
            mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=self.scale)
            for qi, ki, vi in zip(*splits)
        ]
        out = mx.concatenate(out, axis=2).transpose(0, 2, 1, 3).reshape(seq_len, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, cu_seqlens: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.merge_size = config.merge_size
        self.window_size = config.pos_emb_height * config.patch_size
        self.patch_embedder = PatchEmbed(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = VisionRotaryEmbedding(head_dim, config.rope_theta)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, pixel_values: mx.array, grid_thw: np.ndarray) -> mx.array:
        cu_seqlens = get_cu_seqlens(grid_thw)
        window_index, cu_window_seqlens = get_window_index(grid_thw, self.window_size, self.config.patch_size)

        x = self.patch_embedder(pixel_values, grid_thw)
        x = self.ln_pre(x)
        x = x[window_index]

        position_ids = get_position_ids(grid_thw)[:, ::-1].copy() + 1  # Glimmer flips (h, w) -> (w, h), offset 1
        position_ids = mx.array(position_ids)[window_index]
        cos, sin = self.rotary_emb(position_ids)

        # [local, local, local, global] repeating by layer, as the text
        # decoder's layer_types: "full_attention" (cu_seqlens) else windowed.
        layer_types = self.config.layer_types or (["window_attention"] * 3 + ["full_attention"]) * (
            len(self.layers) // 4 + 1
        )
        for layer, layer_type in zip(self.layers, layer_types):
            seqlens = cu_seqlens if layer_type == "full_attention" else cu_window_seqlens
            x = layer(x, seqlens, cos, sin)

        reverse = mx.argsort(window_index)
        x = x[reverse]
        x = self.ln_post(x)

        shuffle_index = get_pixel_shuffle_index(grid_thw, self.merge_size)
        x = x[shuffle_index]
        dim = x.shape[-1]
        factor = self.merge_size * self.merge_size
        x = x.reshape(-1, factor, dim).transpose(0, 2, 1).reshape(-1, dim * factor)
        return x

    def sanitize(self, weights: dict) -> dict:
        return {k: v for k, v in weights.items() if "position_ids" not in k}


class VisionAdapter(nn.Module):
    def __init__(self, hidden_size: int, projector_hidden_size: int, act: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(projector_hidden_size, projector_hidden_size, bias=False)
        self.act = nn.gelu if act == "gelu" else nn.gelu_approx

    def __call__(self, x: mx.array) -> mx.array:
        return self.act(self.fc2(self.act(self.fc1(x))))
