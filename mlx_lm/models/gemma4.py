# Copyright © 2025 Apple Inc.

"""Top-level multimodal Gemma 4 model.

Composes the text decoder (`gemma4_text.py`), the vision tower
(`gemma4_vision.py`), and the audio tower (`gemma4_audio.py`) the same way
`transformers.models.gemma4.modeling_gemma4.Gemma4Model` /
`Gemma4ForConditionalGeneration` do: run each tower, project its soft tokens
into text embedding space with a `Gemma4MultimodalEmbedder` (`embed_vision`/
`embed_audio`), splice the projected soft tokens into the text token
embedding sequence at the placeholder positions (`input_ids ==
config.image_token_id` / `config.audio_token_id`), and run the fused sequence
through the text decoder.

What is verified (read directly off `transformers/models/gemma4/{modeling_gemma4,
modular_gemma4,configuration_gemma4}.py` and a real `google/gemma-4-E4B-it`
`config.json`):

- `Gemma4Model.forward`: builds `llm_input_ids` by replacing every
  image/audio placeholder id with `text_config.pad_token_id` *before* the
  main embedding lookup (so the lookup never indexes a placeholder id -- it's
  in-range but semantically a dummy that gets overwritten below), embeds
  those with `get_input_embeddings()` (the *scaled* word embedding -- see
  below), computes the per-layer-input "token identity" component from
  `llm_input_ids` (also before splicing, so per-layer inputs never need a
  reverse embedding lookup at image/audio positions), then
  `masked_scatter`s the projected vision/audio features into the scaled text
  embedding sequence at the placeholder positions, and finally runs the text
  decoder on the fused, already-scaled sequence with the precomputed
  per-layer "token identity" tensor passed through (the decoder still adds
  its own context-dependent per-layer projection on top, using the *fused*
  sequence).
- `Gemma4MultimodalEmbedder` (`embed_vision`/`embed_audio`): a single shared
  architecture (`embedding_pre_projection_norm` -- a `Gemma4RMSNorm` with
  `with_scale=False`, i.e. no learnable weight tensor -- followed by
  `embedding_projection`, a bias-free `Linear`) parameterized by a different
  input width per branch: `config.audio_config.output_proj_dims` (1536 for
  `google/gemma-4-E4B-it`) for audio, `config.vision_config.hidden_size`
  (768) for vision (vision has no `output_proj_dims` field, so
  `Gemma4MultimodalEmbedder.__init__`'s `getattr(config, "output_proj_dims",
  config.hidden_size)` falls back to `hidden_size`). Neither tower's own
  module (`gemma4_vision.VisionModel`, `gemma4_audio.AudioModel`) includes
  this projection itself -- it lives one level up, on `Gemma4Model`, which is
  why `gemma4_vision.py`'s `VisionModel.__call__` docstring calls out that it
  deliberately returns pre-projection pooled hidden states.
- The critical, easy-to-get-wrong subtlety this file has to reproduce
  without touching `gemma4_text.py`: HF's `embed_tokens` for the text
  model is a `Gemma3TextScaledWordEmbedding` -- it multiplies by
  `sqrt(hidden_size)` *inside the embedding lookup itself*, once, and the
  image/audio features are spliced into that *already-scaled* sequence with
  no further scaling. `gemma4_text.py`'s `embed_tokens`, by contrast, is a
  plain unscaled `nn.Embedding`, and `Gemma4TextModel.__call__`
  unconditionally multiplies *whatever* `input_embeddings` it receives
  (freshly embedded or passed in) by `embed_scale`. If this file spliced raw
  projected multimodal features into a raw (unscaled) text embedding
  sequence and let that uniform multiply apply to the whole thing, the
  multimodal features would end up incorrectly scaled by
  `sqrt(hidden_size)` (~50x for E4B) relative to HF. This file instead
  divides the projected multimodal features by that same `embed_scale`
  before splicing, so that `gemma4_text.py`'s internal multiply exactly
  cancels it back out -- reproducing HF's behavior through the existing,
  unmodified `gemma4_text.py` public API rather than by changing its
  contract.
- Per-layer inputs (PLE, `hidden_size_per_layer_input>0` -- true for
  `google/gemma-4-E4B-it`, `256`): this file calls
  `Gemma4TextModel._get_per_layer_inputs` directly (a "private"-by-convention
  but not actually private method) with `llm_input_ids`, mirroring HF's
  `get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)` call, which -- for
  an explicit non-`None` `input_ids` argument -- never even looks at the
  embeddings argument (only the `input_ids is None` reverse-lookup branch
  does), so this file does not need to pass or construct
  `llm_inputs_embeds` at all. The resulting tensor is passed through as
  `per_layer_inputs` to `gemma4_text.Model.__call__`, which (per its
  existing, unmodified logic) treats a non-`None` `per_layer_inputs` as
  already containing the token-identity component and only adds the
  context-dependent projection computed from the final (fused, scaled)
  sequence -- exactly matching
  `Gemma4TextModel.project_per_layer_inputs(inputs_embeds=<fused, scaled>,
  per_layer_inputs=<precomputed>)`.
- Token ids (`image_token_id=258880`, `audio_token_id=258881`,
  `boi_token_id=255999`, `eoi_token_id=258882`, `boa_token_id=256000`,
  `eoa_token_index=258883`) and the three `*_config` sub-dicts
  (`text_config`/`vision_config`/`audio_config`) are read directly off a real
  `google/gemma-4-E4B-it` `config.json`.

What is a deliberate scope decision, not a guess: video input
(`pixel_values_videos`/`video_token_id`) is not implemented -- the task and
the available checkpoints (image + audio) don't exercise it, and it reuses
the vision tower via a near-identical code path in HF that would be
straightforward to add later following the same pattern as images.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import gemma4_audio, gemma4_text, gemma4_vision
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4"
    text_config: dict = None
    vision_config: Optional[dict] = None
    audio_config: Optional[dict] = None
    vocab_size: int = 262144
    boi_token_id: Optional[int] = 255_999
    eoi_token_id: Optional[int] = 258_882
    image_token_id: Optional[int] = 258_880
    video_token_id: Optional[int] = 258_884
    boa_token_id: Optional[int] = 256_000
    eoa_token_index: Optional[int] = 258_883
    audio_token_id: Optional[int] = 258_881

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        self.text_config["vocab_size"] = self.vocab_size
        self.text_config["num_attention_heads"] = self.text_config.get(
            "num_attention_heads", 8
        )
        self.text_config["num_key_value_heads"] = self.text_config.get(
            "num_key_value_heads", 1
        )


class MultimodalEmbedder(nn.Module):
    """Projects a multimodal tower's soft tokens into text embedding space.

    Mirrors `transformers.models.gemma4.modeling_gemma4.Gemma4MultimodalEmbedder`
    (one shared architecture for both the vision and audio branches --
    parameterized here directly by `multimodal_hidden_size` since
    `gemma4_vision.ModelArgs` has no `output_proj_dims` field the way
    `gemma4_audio.ModelArgs` does; `gemma4_audio.MultimodalEmbedder` already
    implements this same architecture and is used as-is for `embed_audio`
    below).
    """

    def __init__(
        self, multimodal_hidden_size: int, text_hidden_size: int, eps: float = 1e-6
    ):
        super().__init__()
        self.embedding_projection = nn.Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = gemma4_audio.RMSNorm(
            multimodal_hidden_size, eps=eps, with_scale=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


def _masked_scatter(embeddings: mx.array, mask: mx.array, features: mx.array) -> mx.array:
    """Equivalent to `torch.Tensor.masked_scatter(mask, features)`: fills
    the `True` positions of `mask` (row-major order) with successive rows of
    `features` (`features.shape[0]` must equal `mask.sum()`).

    `embeddings`: (B, L, D). `mask`: (B, L) bool. `features`: (N, D).
    """
    B, L, D = embeddings.shape
    flat_embeddings = embeddings.reshape(B * L, D)
    flat_mask = mask.reshape(B * L)
    # `positions[i]` is the row of `features` that should land at flattened
    # position `i` if `flat_mask[i]` is True (irrelevant, and clamped
    # in-bounds, otherwise).
    positions = mx.cumsum(flat_mask.astype(mx.int32)) - 1
    positions = mx.clip(positions, 0, max(features.shape[0] - 1, 0))
    gathered = mx.take(features, positions, axis=0)
    flat_out = mx.where(flat_mask[:, None], gathered.astype(flat_embeddings.dtype), flat_embeddings)
    return flat_out.reshape(B, L, D)


def _compact_valid_rows(x: mx.array, valid: mx.array) -> mx.array:
    """Selects the rows of `x` (..., D) where the boolean mask `valid`
    (matching shape minus the last dim) is True, flattening all leading dims
    -- i.e. `x.reshape(-1, D)[valid.reshape(-1)]` in numpy terms. Mirrors
    HF packing variable numbers of valid soft tokens per image/audio clip
    into one flat sequence (`hidden_states[pooler_mask]` for vision,
    `audio_features[audio_mask_from_encoder]` for audio) before splicing.
    """
    flat_x = x.reshape(-1, x.shape[-1])
    flat_valid = valid.reshape(-1)
    # Boolean compaction has a data-dependent output length, so this (like
    # any masked_select/nonzero) needs a host-side index list.
    idx = np.nonzero(np.array(flat_valid))[0]
    return mx.take(flat_x, mx.array(idx), axis=0)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )
        text_hidden_size = self.language_model.args.hidden_size
        # gemma4_unified (remapped here in utils.MODEL_REMAPPING) is an
        # encoder-free multimodal variant: its checkpoints carry a
        # `vision_embedder` and no vision/audio towers, a layout this file
        # doesn't implement. Its vision_config/audio_config describe those
        # missing towers, so building them made every load fail with
        # "Missing N parameters: audio_tower..., vision_tower...". Loaded
        # text-only; sanitize() drops the unified media weights.
        text_only = args.model_type == "gemma4_unified"

        self.vision_tower = None
        self.embed_vision = None
        if args.vision_config is not None and not text_only:
            vision_args = gemma4_vision.ModelArgs.from_dict(args.vision_config)
            self.vision_tower = gemma4_vision.VisionModel(vision_args)
            self.embed_vision = MultimodalEmbedder(
                vision_args.hidden_size,
                text_hidden_size,
                eps=vision_args.rms_norm_eps,
            )

        self.audio_tower = None
        self.embed_audio = None
        if args.audio_config is not None and not text_only:
            audio_args = gemma4_audio.ModelArgs.from_dict(args.audio_config)
            self.audio_tower = gemma4_audio.AudioModel(audio_args)
            self.embed_audio = gemma4_audio.MultimodalEmbedder(
                audio_args, text_hidden_size=text_hidden_size
            )

    def _fuse_multimodal_inputs(
        self,
        inputs: mx.array,
        pixel_values: Optional[mx.array],
        pixel_position_ids: Optional[mx.array],
        input_features: Optional[mx.array],
        input_features_mask: Optional[mx.array],
    ):
        args = self.args
        text_model = self.language_model.model
        embed_scale = text_model.embed_scale

        image_mask = (
            inputs == args.image_token_id
            if args.image_token_id is not None
            else mx.zeros_like(inputs).astype(mx.bool_)
        )
        audio_mask = (
            inputs == args.audio_token_id
            if args.audio_token_id is not None
            else mx.zeros_like(inputs).astype(mx.bool_)
        )
        multimodal_mask = image_mask | audio_mask

        pad_token_id = text_model.config.pad_token_id
        llm_input_ids = mx.where(multimodal_mask, pad_token_id, inputs)

        # Raw (unscaled) text embeddings -- `embed_scale` is applied once,
        # uniformly, inside `gemma4_text.Model.__call__`; see module
        # docstring for why the multimodal features below are pre-divided
        # by it instead of being spliced in post-scale.
        fused = text_model.embed_tokens(llm_input_ids)

        per_layer_inputs = None
        if text_model.hidden_size_per_layer_input:
            per_layer_inputs = text_model._get_per_layer_inputs(llm_input_ids)

        if pixel_values is not None:
            if self.vision_tower is None or self.embed_vision is None:
                raise ValueError(
                    "pixel_values was given but this model has no vision_config "
                    "(vision_tower/embed_vision were not initialized)."
                )
            pooled, valid_mask = self.vision_tower(pixel_values, pixel_position_ids)
            image_features = _compact_valid_rows(pooled, valid_mask)
            image_features = self.embed_vision(image_features)

            n_image_tokens = image_mask.sum().item()
            if image_features.shape[0] != n_image_tokens:
                raise ValueError(
                    "Image features and image placeholder tokens do not match: "
                    f"{n_image_tokens} placeholder tokens vs. "
                    f"{image_features.shape[0]} pooled image features."
                )
            fused = _masked_scatter(
                fused, image_mask, image_features / embed_scale
            )

        if input_features is not None:
            if self.audio_tower is None or self.embed_audio is None:
                raise ValueError(
                    "input_features was given but this model has no audio_config "
                    "(audio_tower/embed_audio were not initialized)."
                )
            audio_hidden, audio_valid_mask = self.audio_tower(
                input_features, input_features_mask
            )
            if audio_valid_mask is None:
                audio_valid_mask = mx.ones(audio_hidden.shape[:2], dtype=mx.bool_)
            audio_features = _compact_valid_rows(audio_hidden, audio_valid_mask)
            audio_features = self.embed_audio(audio_features)

            n_audio_tokens = audio_mask.sum().item()
            if audio_features.shape[0] != n_audio_tokens:
                raise ValueError(
                    "Audio features and audio placeholder tokens do not match: "
                    f"{n_audio_tokens} placeholder tokens vs. "
                    f"{audio_features.shape[0]} pooled audio features."
                )
            fused = _masked_scatter(
                fused, audio_mask, audio_features / embed_scale
            )

        return fused, per_layer_inputs

    # -- Bidirectional attention within an image (26B-A4B, 12B) ----------

    def _text_model(self):
        return self.language_model.model

    def uses_vision_bidirectional_attention(self) -> bool:
        return self._text_model().config.use_bidirectional_attention == "vision"

    def set_vision_spans(self, spans):
        """[start, end) prompt positions of each image's soft tokens, for
        the prefill that follows (None to clear): their tokens then attend
        to each other in both directions, as HF does. The caller clears it
        once the prompt is processed; decoding is causal either way."""
        self._text_model().vision_spans = (
            list(spans) if spans and self.uses_vision_bidirectional_attention() else None
        )

    @staticmethod
    def image_spans(ids, image_token_id):
        """[start, end) runs of `image_token_id` in a list of token ids."""
        spans, start = [], None
        for i, t in enumerate(list(ids) + [None]):
            if t == image_token_id and start is None:
                start = i
            elif t != image_token_id and start is not None:
                spans.append((start, i))
                start = None
        return spans

    def prefill_chunk_size(self, start: int, n: int) -> int:
        """How much of a prefill chunk of `n` tokens from prompt position
        `start` to take so it doesn't end inside an image (its first tokens
        would miss the rest): up to the image, or through it."""
        for s, e in self._text_model().vision_spans or ():
            if s < start + n < e:
                return s - start if s > start else e - start
        return n

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_position_ids: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
    ):
        if pixel_values is None and input_features is None:
            # Unaffected text-only (or pre-fused-embeddings) path: identical
            # to this file's behavior before multimodal support was added.
            return self.language_model(
                inputs,
                cache=cache,
                input_embeddings=input_embeddings,
                per_layer_inputs=per_layer_inputs,
            )

        if input_embeddings is not None or per_layer_inputs is not None:
            raise ValueError(
                "input_embeddings/per_layer_inputs cannot be passed together "
                "with pixel_values/input_features -- they are computed "
                "internally from `inputs` during multimodal fusion."
            )

        fused_embeddings, fused_per_layer_inputs = self._fuse_multimodal_inputs(
            inputs,
            pixel_values,
            pixel_position_ids,
            input_features,
            input_features_mask,
        )
        # A direct call with the images: their spans come from `inputs`
        # (offset by what the cache already holds), for this call only.
        text_model = self._text_model()
        saved = text_model.vision_spans
        if pixel_values is not None and saved is None and inputs.shape[0] == 1:
            base = next((c.offset for c in (cache or []) if c is not None), 0)
            self.set_vision_spans(
                [(base + s, base + e) for s, e in self.image_spans(inputs[0].tolist(), self.args.image_token_id)]
            )
        try:
            return self.language_model(
                inputs,
                cache=cache,
                input_embeddings=fused_embeddings,
                per_layer_inputs=fused_per_layer_inputs,
            )
        finally:
            text_model.vision_spans = saved

    def sanitize(self, weights):
        text_weights = {}
        vision_weights = {}
        audio_weights = {}
        embed_vision_weights = {}
        embed_audio_weights = {}

        for k, v in weights.items():
            starts_w_model = k.startswith("model.")
            k = k.removeprefix("model.")

            if k.startswith("vision_tower."):
                vision_weights[k] = v
                continue
            if k.startswith("audio_tower."):
                audio_weights[k] = v
                continue
            if k.startswith("embed_vision."):
                embed_vision_weights[k.removeprefix("embed_vision.")] = v
                continue
            if k.startswith("embed_audio."):
                embed_audio_weights[k.removeprefix("embed_audio.")] = v
                continue
            if k.startswith(
                ("multi_modal_projector", "vision_embedder")
            ):
                # "vision_embedder": gemma4_unified encoder-free vision
                # variant, a different checkpoint layout this file doesn't
                # support. "multi_modal_projector": not present on the real
                # google/gemma-4-E4B-it checkpoint layout (embed_vision's
                # own embedding_projection plays that role instead); dropped
                # defensively in case some other checkpoint has it.
                continue

            if k == "lm_head.weight" and self.language_model.tie_word_embeddings:
                # HF's in-memory state_dict includes a tied `lm_head.weight`
                # even when `tie_word_embeddings=True`; real safetensors
                # checkpoints normally dedupe it away, but drop it
                # defensively either way since `gemma4_text.Model` has no
                # `lm_head` parameter to receive it when tied.
                continue

            if not starts_w_model:
                text_weights[k] = v
                continue

            if k.startswith("language_model"):
                k = k.replace("language_model.", "language_model.model.")

            text_weights[k] = v

        sanitized = dict(self.language_model.sanitize(text_weights))

        if self.vision_tower is not None:
            for k, v in self.vision_tower.sanitize(vision_weights).items():
                sanitized[f"vision_tower.{k}"] = v
        if self.audio_tower is not None:
            for k, v in self.audio_tower.sanitize(audio_weights).items():
                sanitized[f"audio_tower.{k}"] = v
        if self.embed_vision is not None:
            for k, v in embed_vision_weights.items():
                sanitized[f"embed_vision.{k}"] = v
        if self.embed_audio is not None:
            for k, v in embed_audio_weights.items():
                sanitized[f"embed_audio.{k}"] = v

        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    def make_cache(self):
        return self.language_model.make_cache()
