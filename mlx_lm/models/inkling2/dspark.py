"""The DSpark draft of Inkling 2 (``mtp.safetensors``).

A block of an anchor (the last sampled token) and ``block_size - 1`` mask
embeddings runs through a small dense transformer. Its keys and values are
the block's own plus a context: per draft layer, a learned mix of the
target's hidden states at ``target_layer_ids`` for every committed
position. A recurrent Markov head then corrects the logits of the mask
positions one after the other, from the token drafted before. As vLLM's
``vllm/models/inkling2/nvidia/dspark.py``.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import KVCache

__all__ = ["DSpark", "DSparkArgs"]


@dataclass
class DSparkArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: list[int]
    sliding_window_size: int
    block_size: int
    markov_rank: int
    target_layer_ids: list[int]
    markov_head_type: str = "rnn_hidden"


def _mix(features: mx.array, alpha: mx.array, norm: nn.RMSNorm) -> mx.array:
    """RMSNorm of the alpha-weighted sum of the target features (..., taps, hidden)."""
    mixed = (features.astype(mx.float32) * alpha[:, None]).sum(axis=-2)
    return mx.fast.rms_norm(mixed, norm.weight.astype(mx.float32), norm.eps)


class _Context:
    """Context keys and values of one draft layer, committed positions only:
    the last ``window`` of them for a local layer."""

    def __init__(self, window: int):
        self.window = window
        self.cache = None if window else KVCache()
        self.keys = self.values = None

    def append(self, keys: mx.array, values: mx.array):
        if self.cache is not None:
            self.keys, self.values = self.cache.update_and_fetch(keys, values)
            return
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys = keys[..., -self.window :, :]
        self.values = values[..., -self.window :, :]

    @property
    def state(self):
        return [a for a in (self.keys, self.values) if a is not None]


class Attention(nn.Module):
    def __init__(self, args: DSparkArgs):
        super().__init__()
        dims, self.head_dim = args.hidden_size, args.head_dim
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        q_dims = self.num_heads * self.head_dim
        kv_dims = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dims, q_dims, bias=False)
        self.k_proj = nn.Linear(dims, kv_dims, bias=False)
        self.v_proj = nn.Linear(dims, kv_dims, bias=False)
        self.o_proj = nn.Linear(q_dims, dims, bias=False)
        self.k_proj_ctx = nn.Linear(dims, kv_dims, bias=False)
        self.v_proj_ctx = nn.Linear(dims, kv_dims, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def _heads(self, x: mx.array, n: int) -> mx.array:
        B, L, _ = x.shape
        return x.reshape(B, L, n, self.head_dim)

    def context(self, x: mx.array, position: int) -> tuple[mx.array, mx.array]:
        keys = self.k_norm_ctx(self._heads(self.k_proj_ctx(x), self.num_kv_heads))
        keys = self.rope(keys.transpose(0, 2, 1, 3), offset=position)
        values = self._heads(self.v_proj_ctx(x), self.num_kv_heads)
        return keys, values.transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, position: int, context: _Context) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_norm(self._heads(self.q_proj(x), self.num_heads))
        keys = self.k_norm(self._heads(self.k_proj(x), self.num_kv_heads))
        values = self._heads(self.v_proj(x), self.num_kv_heads)
        queries = self.rope(queries.transpose(0, 2, 1, 3), offset=position)
        keys = self.rope(keys.transpose(0, 2, 1, 3), offset=position)
        values = values.transpose(0, 2, 1, 3)
        if context.keys is not None:
            keys = mx.concatenate([context.keys, keys], axis=2)
            values = mx.concatenate([context.values, values], axis=2)
        # The block attends to all of itself and its context: no mask.
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim**-0.5
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, args: DSparkArgs):
        super().__init__()
        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.dflare_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x: mx.array, position: int, context: _Context) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), position, context)
        return x + self.mlp(self.post_attention_layernorm(x))


class MarkovHead(nn.Module):
    def __init__(self, args: DSparkArgs, vocab_size: int):
        super().__init__()
        rank, dims = args.markov_rank, args.hidden_size
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.rnn_proj = nn.Linear(2 * rank + 2 * dims, 3 * rank)
        self.hidden_out = nn.Linear(rank, dims)
        self.dflare_norm = nn.RMSNorm(dims, eps=args.rms_norm_eps)

    def step(self, state, prev, hidden, f_rnn):
        """One recurrence step: the new state and the correction of the
        hidden state (the LM head turns it into the logits' bias)."""
        R = self.markov_rank
        joint = mx.concatenate(
            [state, self.markov_w1(prev).astype(mx.float32), hidden, f_rnn], axis=-1
        )
        raw = self.rnn_proj(joint.astype(self.rnn_proj.weight.dtype))
        raw = raw.astype(mx.float32)
        gate = mx.sigmoid(raw[..., :R])
        state = gate * state + (1 - gate) * mx.tanh(raw[..., R : 2 * R])
        latent = mx.tanh(raw[..., 2 * R :])
        return state, self.hidden_out(latent.astype(self.hidden_out.weight.dtype))


class DSpark(nn.Module):
    def __init__(self, args: DSparkArgs, vocab_size: int):
        super().__init__()
        self.args = args
        self.layers = [Layer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.markov_head = MarkovHead(args, vocab_size)
        taps = len(args.target_layer_ids)
        self.fuse_weight = mx.zeros((args.num_hidden_layers + 1, taps))
        self.mask_embed = mx.zeros((args.hidden_size,))

    @property
    def block_size(self) -> int:
        return self.args.block_size

    def make_cache(self) -> list[_Context]:
        window = self.args.sliding_window_size
        return [
            _Context(window if i in self.args.local_layer_ids else 0)
            for i in range(len(self.layers))
        ]

    def _alpha(self) -> mx.array:
        return mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1)

    def add_context(self, features: mx.array, position: int, cache: list[_Context]):
        """Commit the target features (B, L, taps, hidden) of positions
        position .. position + L - 1."""
        alpha = self._alpha()
        dtype = self.mask_embed.dtype
        for i, (layer, context) in enumerate(zip(self.layers, cache)):
            x = _mix(features, alpha[i], layer.dflare_norm).astype(dtype)
            context.append(*layer.self_attn.context(x, position))

    def __call__(self, anchor: mx.array, position: int, cache: list[_Context]):
        """Final hidden states of the block [anchor, masks...] at positions
        position .. position + block_size - 1; anchor: (B, 1, hidden)."""
        B = anchor.shape[0]
        masks = mx.broadcast_to(
            self.mask_embed.astype(anchor.dtype),
            (B, self.block_size - 1, anchor.shape[-1]),
        )
        x = mx.concatenate([anchor, masks], axis=1)
        for layer, context in zip(self.layers, cache):
            x = layer(x, position, context)
        return self.norm(x)

    def draft(self, anchor_token, anchor, position, last_features, cache, head, k):
        """k greedy draft tokens after anchor_token. last_features: the target
        features (B, taps, hidden) of the position before the anchor; head
        maps hidden states to final logits."""
        hidden = self(anchor, position, cache)
        f_rnn = _mix(last_features, self._alpha()[-1], self.markov_head.dflare_norm)
        B = anchor.shape[0]
        state = mx.zeros((B, self.markov_head.markov_rank), mx.float32)
        prev, drafts = anchor_token.reshape(B), []
        for i in range(k):
            h = hidden[:, i + 1]
            state, correction = self.markov_head.step(
                state, prev, h.astype(mx.float32), f_rnn
            )
            prev = mx.argmax(head(h + correction.astype(h.dtype)), axis=-1)
            drafts.append(prev)
        return mx.stack(drafts, axis=-1)
