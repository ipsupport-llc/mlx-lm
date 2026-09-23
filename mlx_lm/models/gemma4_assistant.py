# Copyright © 2026 Apple Inc.

"""Gemma 4 Multi-Token-Prediction drafter ("assistant", e.g.
google/gemma-4-26B-A4B-it-assistant).

Not a standalone language model: a small Gemma 4 text stack (4 layers for the
26B-A4B) that owns no KV cache at all -- every layer is KV-shared and attends
to the *main* model's keys/values (the last full-attention and the last
sliding-attention layer of the main model). Each draft step takes

    concat(main_embedding(last_token) * embed_scale, main_hidden)

(main_hidden: the main model's final, normed hidden state that predicted
`last_token`, or on later draft steps this drafter's own projected output),
projects it down with `pre_projection`, runs the decoder layers with the
query at the fixed position of `last_token` and no mask (HF uses a
bidirectional mask; with one query over already-past keys that is "attend to
everything", plus the last `sliding_window` keys for sliding layers), and
returns the tied-embedding logits and `post_projection(h)` -- the hidden
state fed back in for the next draft step.

Mirrors transformers' `Gemma4AssistantForCausalLM` +
`SinglePositionMultiTokenCandidateGenerator`. The generation loop that feeds
it lives in `generate.gemma4_mtp_generate_step`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from . import gemma4_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_assistant"
    text_config: Dict[str, Any] = field(default_factory=dict)
    backbone_hidden_size: int = 2816
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True


class DrafterTextModel(nn.Module):
    """The drafter's decoder stack. Kept separate from
    `gemma4_text.Gemma4TextModel` because that one always scales its input
    by `embed_scale` and wires KV sharing to earlier layers of the *same*
    model, while here the input is already a projection and every layer
    borrows the main model's KV."""

    def __init__(self, config: gemma4_text.ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            gemma4_text.DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        if args.use_ordered_embeddings:
            # The E2B/E4B drafters score only the top centroid clusters of the
            # vocabulary (Gemma4AssistantMaskedEmbedder); not implemented yet.
            raise NotImplementedError(
                "gemma4_assistant with use_ordered_embeddings=True (E2B/E4B "
                "drafters) is not supported yet."
            )
        self.text_args = gemma4_text.ModelArgs.from_dict(args.text_config)
        if self.text_args.num_kv_shared_layers != self.text_args.num_hidden_layers:
            raise ValueError("Expected every drafter layer to be KV-shared.")
        H, B = self.text_args.hidden_size, args.backbone_hidden_size
        self.model = DrafterTextModel(self.text_args)
        self.pre_projection = nn.Linear(2 * B, H, bias=False)
        self.post_projection = nn.Linear(H, B, bias=False)
        self.final_logit_softcapping = self.text_args.final_logit_softcapping

    @property
    def sliding_window(self) -> int:
        return self.text_args.sliding_window

    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv: Dict[str, Tuple[mx.array, mx.array]],
        position: int,
    ) -> Tuple[mx.array, mx.array]:
        """One draft step.

        Args:
            inputs_embeds: ``[B, 1, 2 * backbone_hidden_size]``.
            shared_kv: ``{"full_attention": (K, V), "sliding_attention": (K,
              V)}`` from the main model, already RoPE'd, ``[B, heads, L,
              head_dim]``, containing only valid (accepted) positions; the
              sliding pair already cut to the last ``sliding_window`` keys.
            position: absolute position of the token whose embedding is in
              ``inputs_embeds`` (RoPE offset for the queries).

        Returns:
            (logits ``[B, 1, vocab]``, next hidden ``[B, 1,
            backbone_hidden_size]``).
        """
        h = self.pre_projection(inputs_embeds)
        for layer in self.model.layers:
            h, _, _ = layer(
                h, None, None, shared_kv=shared_kv[layer.layer_type], offset=position
            )
        h = self.model.norm(h)
        logits = self.model.embed_tokens.as_linear(h)
        if self.final_logit_softcapping is not None:
            logits = gemma4_text.logit_softcap(self.final_logit_softcapping, logits)
        return logits, self.post_projection(h)

    def sanitize(self, weights):
        return {
            k: v
            for k, v in weights.items()
            if not k.startswith("masked_embedding")
            and "rotary_emb" not in k
            and k != "lm_head.weight"
        }

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # No cache of its own: all KV comes from the main model.
        return []
