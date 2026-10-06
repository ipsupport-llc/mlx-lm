# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .pipeline import PipelineMixin
from .qwen3_5_vision import VisionConfig, VisionModel
from .qwen3_next import Qwen3NextAttention as Attention
from .qwen3_next import Qwen3NextMLP as MLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


class MRoPEState:
    """3D (t, h, w) positions of the current image prompt. None means plain RoPE."""

    def __init__(self):
        self.positions = None  # mx.array [3, N] int32
        self.delta = 0  # position minus index after the prompt

    def window(self, offset: int, length: int):
        if self.positions is None:
            return None
        n = self.positions.shape[1]
        if offset + length <= n:
            return self.positions[:, offset : offset + length]
        start = max(offset, n)
        tail = mx.arange(start, offset + length, dtype=mx.int32) + self.delta
        tail = mx.broadcast_to(tail[None], (3, tail.shape[0]))
        if offset >= n:
            return tail
        return mx.concatenate([self.positions[:, offset:], tail], axis=1)


class MRoPE(nn.Module):
    """Interleaved multimodal RoPE (Qwen3-VL, Qwen3.5) on the first `dims` channels."""

    def __init__(
        self, dims: int, base: float, mrope_section, state: MRoPEState, text_rope
    ):
        super().__init__()
        self.dims = dims
        self.base = base
        self.media = state
        # Text positions use the original RoPE, so cached text is identical.
        self.text_rope = text_rope
        half = dims // 2
        self._inv_freq = 1.0 / (
            base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )
        selector = [0] * half
        for axis, start in ((1, 1), (2, 2)):
            for idx in range(start, min(mrope_section[axis] * 3, half), 3):
                selector[idx] = axis
        self._selector = mx.array(selector, dtype=mx.int32)

    def __call__(self, x, offset=0):
        L = x.shape[-2]
        pos = self.media.window(offset, L) if isinstance(offset, int) else None
        if pos is None:
            return self.text_rope(x, offset=offset)
        start = pos[0, 0]
        linear = mx.arange(L, dtype=pos.dtype) + start
        if mx.all(pos == linear[None]).item():
            return self.text_rope(x, offset=int(start.item()))
        angles = (
            mx.take(pos, self._selector, axis=0).T.astype(mx.float32) * self._inv_freq
        )  # [L, half]
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        half = self.dims // 2
        x1, x2, rest = x[..., :half], x[..., half : self.dims], x[..., self.dims :]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1)


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return self.norm(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: Optional[dict] = None
    image_token_id: Optional[int] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    # Loaded on first use with lazy_towers.
    lazy_modules = ("vision_tower",)

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))
        # VL checkpoints: Qwen3-VL vision tower and mRoPE in full attention.
        self.vision_tower = None
        vision_config = getattr(args, "vision_config", None)
        if vision_config is not None and vision_config.get("deepstack_visual_indexes"):
            # Deepstack is not implemented. Qwen3.5 does not use it.
            print(
                "[WARNING] qwen3_5: deepstack vision isn't supported; loading text-only."
            )
            vision_config = None
        if (
            vision_config is not None
            and getattr(args, "image_token_id", None) is not None
        ):
            self.vision_tower = VisionModel(
                VisionConfig.from_dict({**vision_config, "model_type": "qwen3_5"})
            )
            self.mrope = MRoPEState()
            text = self.language_model.args
            dims = int(text.head_dim * text.partial_rotary_factor)
            section = (text.rope_parameters or {}).get("mrope_section", [11, 11, 10])
            for layer in self.language_model.model.layers:
                if not layer.is_linear:
                    layer.self_attn.rope = MRoPE(
                        dims, text.rope_theta, section, self.mrope, layer.self_attn.rope
                    )

    def set_media_positions(self, media):
        """Set (positions [3, N], delta) for one image request, or None to clear."""
        if self.vision_tower is None:
            return
        if media is None:
            self.mrope.positions, self.mrope.delta = None, 0
        else:
            self.mrope.positions, self.mrope.delta = media

    def embed_with_images(
        self, input_ids: mx.array, pixel_values: mx.array, grid_thw: mx.array
    ) -> mx.array:
        """Token embeddings with the image tokens replaced by vision features."""
        embeds = self.language_model.model.embed_tokens(input_ids)
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        features, _ = self.vision_tower(pixel_values.astype(dtype), grid_thw)
        mask = (input_ids == self.args.image_token_id)[0]
        positions = mx.array(
            [i for i, m in enumerate(mask.tolist()) if m], dtype=mx.int32
        )
        if positions.shape[0] != features.shape[0]:
            raise ValueError(
                f"{positions.shape[0]} image tokens for {features.shape[0]} image features"
            )
        embeds = embeds.at[0, positions].add(
            features.astype(embeds.dtype) - embeds[0, positions]
        )
        return embeds

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        vision = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                # Keep only if the model has a vision tower.
                if self.vision_tower is not None:
                    vision[key.replace("model.visual", "vision_tower", 1)] = value
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        sanitized = self.language_model.sanitize(sanitized)
        if vision:
            stripped = {k[len("vision_tower.") :]: v for k, v in vision.items()}
            sanitized.update(
                {
                    "vision_tower." + k: v
                    for k, v in self.vision_tower.sanitize(stripped).items()
                }
            )
        return sanitized

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        # Keep the vision tower in bf16. Quantization breaks its position embedding.
        text = self.language_model.quant_predicate
        if self.vision_tower is None:
            return text

        def predicate(path, module):
            if path.startswith("vision_tower"):
                return False
            return text(path, module) if text is not None else True

        return predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
