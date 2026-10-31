# Copyright © 2026 IPSupport LLC

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "lfm3"
    vocab_size: int = 128_000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    norm_eps: float = 1e-5
    conv_bias: bool = False
    conv_L_cache: int = 7
    layer_types: Optional[List[str]] = None
    head_dim: Optional[int] = None
    sliding_window: int = 256
    use_sandwich_norm: bool = True
    rope_parameters: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = True
    tie_embedding: Optional[bool] = None

    def __post_init__(self):
        # Defaults as in the Hugging Face configuration.
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 10_000.0},
            }
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
                for i in range(self.num_hidden_layers)
            ]
        if self.tie_embedding is not None:
            self.tie_word_embeddings = self.tie_embedding
        for t in set(self.layer_types) - {"conv"}:
            params = self.rope_parameters.get(t)
            if params is None:
                raise ValueError(f"rope_parameters has no entry for the {t} layers")
            if params.get("rope_type", "default") != "default":
                raise ValueError(f"RoPE type {params['rope_type']} is not supported for LFM3")


def _rope_freqs(args: ModelArgs, layer_type: str) -> mx.array:
    """RoPE periods for mx.fast.rope over the whole head. A partial rotary
    factor keeps the full head's frequency progression for the first pairs
    and leaves the others unrotated (an infinite period)."""
    params = args.rope_parameters[layer_type]
    base = params["rope_theta"]
    dim = args.head_dim
    angles = int(params.get("partial_rotary_factor", 1.0) * dim) // 2
    periods = base ** (mx.arange(0, 2 * angles, 2, dtype=mx.float32) / dim)
    return mx.concatenate([periods, mx.full((dim // 2 - angles,), math.inf)])


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        # A sigmoid gate per head on the attention output.
        self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)
        self._freqs = _rope_freqs(args, layer_type)

    def _rope(self, x, offset):
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1))
        keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1))
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self._rope(queries, offset)
        keys = self._rope(keys, offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        gate = mx.sigmoid(self.gate_proj(x)).transpose(0, 2, 1)[..., None]
        output = (output * gate).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class ShortConv(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.L_cache = args.conv_L_cache
        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=args.conv_bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=args.conv_bias)

    def __call__(self, x, mask=None, cache=None):
        B, C, x = mx.split(self.in_proj(x), 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)
        if cache is not None:
            state = cache[0]
            if state is None:
                state = mx.zeros((Bx.shape[0], self.L_cache - 1, self.hidden_size), dtype=Bx.dtype)
            Bx = mx.concatenate([state, Bx], axis=1)
            n_keep = self.L_cache - 1
            t = x.shape[1]
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, t)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(Bx, positions, axis=1)
            else:
                cache[0] = Bx[:, -n_keep:, :]
            cache.advance(t)
        else:
            Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])
        return self.out_proj(C * self.conv(Bx))


class MLP(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.is_attention_layer = layer_type != "conv"
        if self.is_attention_layer:
            self.self_attn = Attention(args, layer_type)
        else:
            self.conv = ShortConv(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.use_sandwich_norm = args.use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = self.operator_norm(x)
        if self.is_attention_layer:
            r = self.self_attn(h, mask=mask, cache=cache)
        else:
            r = self.conv(h, mask=mask, cache=cache)
        if self.use_sandwich_norm:
            r = self.post_operator_norm(r)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich_norm:
            r = self.post_ffn_norm(r)
        return h + r


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, t) for t in args.layer_types]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        types = args.layer_types
        # The first layer of each kind: its cache shapes that kind's mask.
        self.first = {t: types.index(t) for t in set(types)}

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        masks = {}
        if "full_attention" in self.first:
            masks["full_attention"] = create_attention_mask(h, cache[self.first["full_attention"]])
        if "sliding_attention" in self.first:
            masks["sliding_attention"] = create_attention_mask(
                h, cache[self.first["sliding_attention"]], window_size=self.args.sliding_window
            )
        if "conv" in self.first:
            masks["conv"] = create_ssm_mask(h, cache[self.first["conv"]])
        for layer, c in zip(self.layers, cache):
            h = layer(h, masks.get(layer.layer_type), cache=c)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        out = {}
        for name, param in weights.items():
            # Checkpoints with an (empty) vision config nest the text model.
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            if name.startswith("model.vision") or name.startswith("vision"):
                continue
            if "conv.weight" in name and param.ndim == 3 and param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)
            if self.args.tie_word_embeddings and name.startswith("lm_head."):
                continue
            out[name] = param
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "conv":
                caches.append(ArraysCache(size=1))
            elif layer.layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches
