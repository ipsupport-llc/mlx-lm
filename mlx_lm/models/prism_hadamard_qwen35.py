# Copyright © 2026 Apple Inc.

"""Native support for prism-ml's Hadamard-rotated, 2-bit ternary "Bonsai 2"
qwen3_5 checkpoints (model_type "prism_hadamard_qwen35").

Every Linear-style weight (and the embedding table) is stored already
quantized (bits=2, group_size=128, affine) in a rotated basis: a blockwise
Hadamard transform with an explicit per-position sign flip was applied to
each tensor before quantization. The checkpoint's own reference runtime
applies the *same* transform to activations at inference time so the result
is mathematically identical to the unrotated model (up to quantization
error). This file re-implements that activation-side transform natively so
the checkpoint loads through mlx_lm's standard model-loading path with no
custom loader, no separate conversion step, and no precision loss -- the
original 2-bit weights are used completely unchanged.

Reference: prism-ml's own runtime.py (`fwht` / `Packed.__call__`), which
this module mirrors exactly.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn

from . import qwen3_5
from .base import BaseModelArgs


def fwht(x: mx.array, block: int, signs: mx.array, inverse: bool = False) -> mx.array:
    shape, dtype = x.shape, x.dtype
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class RotatedQuantizedLinear(nn.QuantizedLinear):
    """QuantizedLinear that un-rotates its input activation (blockwise
    Hadamard + sign flip) before the quantized matmul, undoing the rotation
    baked into the checkpoint's stored weight."""

    def __init__(self, *args, block: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block
        in_dims = (self.weight.shape[1] * 32) // self.bits
        self.signs = mx.ones((in_dims,))

    def __call__(self, x):
        if self.block:
            x = fwht(x, self.block, self.signs)
        return super().__call__(x)

    @classmethod
    def from_linear(cls, linear_layer, group_size=None, bits=None, mode="affine", block=0):
        output_dims, input_dims = linear_layer.weight.shape
        ql = cls(input_dims, output_dims, False, group_size, bits, mode=mode, block=block)
        ql.weight, ql.scales, *biases = mx.quantize(
            linear_layer.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None
        return ql


class RotatedQuantizedEmbedding(nn.QuantizedEmbedding):
    """QuantizedEmbedding whose dequantized rows are the *rotated* basis;
    applies the inverse blockwise Hadamard transform to recover the actual
    hidden state, matching prism-ml's Packed(embedding=True) path."""

    def __init__(self, *args, block: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block
        self.signs = mx.ones((self.dims,))

    def __call__(self, x):
        out = super().__call__(x)
        return fwht(out, self.block, self.signs, inverse=True) if self.block else out

    def as_linear(self, x):
        if self.block:
            x = fwht(x, self.block, self.signs)
        return super().as_linear(x)

    @classmethod
    def from_embedding(cls, embedding_layer, group_size=None, bits=None, mode="affine", block=0):
        num_embeddings, dims = embedding_layer.weight.shape
        ql = cls(num_embeddings, dims, group_size, bits, mode=mode, block=block)
        ql.weight, ql.scales, *biases = mx.quantize(
            embedding_layer.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None
        return ql


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict = field(default_factory=dict)
    modules: List[Dict[str, Any]] = field(default_factory=list)
    # RMSNorm weights stored zero-centered (the runtime adds 1) -- JANG
    # repacks, see utils._adapt_prism_hadamard_repack. prism's own MLX
    # checkpoints store them with the 1 already added.
    zero_centered_norms: bool = False

    @classmethod
    def from_dict(cls, params):
        return cls(
            model_type=params["model_type"],
            text_config=params.get("text_config", params),
            modules=params.get("modules", []),
            zero_centered_norms=params.get("zero_centered_norms", False),
        )


# The norms qwen3_5 stores with a +1 offset (its sanitize applies it to
# HF-layout checkpoints); linear_attn.norm has none.
_SHIFTED_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _resolve(root, path):
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = qwen3_5.TextModelArgs.from_dict(args.text_config)
        self.language_model = qwen3_5.TextModel(text_args)

        for spec in args.modules:
            parent, attr = _resolve(self.language_model, spec["path"])
            layer = getattr(parent, attr)
            block = spec.get("block", 0)
            if spec.get("embedding", False):
                new_layer = RotatedQuantizedEmbedding.from_embedding(
                    layer, group_size=128, bits=2, mode="affine", block=block
                )
            else:
                new_layer = RotatedQuantizedLinear.from_linear(
                    layer, group_size=128, bits=2, mode="affine", block=block
                )
            setattr(parent, attr, new_layer)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            if self.args.zero_centered_norms and key.endswith(_SHIFTED_NORMS):
                value = value + 1.0
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
