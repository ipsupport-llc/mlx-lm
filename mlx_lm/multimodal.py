# Copyright © 2026 Apple Inc.

"""Image inputs for `mlx_lm.server` chat completions.

The OpenAI chat API sends images as `{"type": "image_url", "image_url":
{"url": "data:image/png;base64,..."}}` content parts. Before this module the
server rejected any non-text part outright ("Only 'text' content type is
supported"), so any client that attached an image got a failed request.

Flow for a request carrying images (vision-capable models only):

1. `extract_images` (HTTP thread) pulls the raw image bytes out of the
   messages, in document order, and rewrites each image part to the
   `{"type": "image"}` form chat templates expect.
2. The chat template renders one image placeholder token per image.
3. `ImageInputs.build` (generation thread) preprocesses the images, expands
   each placeholder into the model's real soft-token span, and runs the
   model's own multimodal fusion once to get the full-prompt input
   embeddings. The server then hands those embeddings to the normal
   `stream_generate` path (it already supports `input_embeddings`, chunked in
   lock-step with the tokens), so chunked prefill can't split an image away
   from its placeholders.

Prompt caching: the server's prompt cache is keyed by token ids, and two
different images produce identical placeholder ids. `build` therefore also
returns a cache key where each image's positions hold a pseudo id derived
from a hash of that image's bytes -- the same image in a later turn reuses
the cache, a different image only matches the text before it.

Only Gemma 4 is wired up so far (its image preprocessing is HF's
`Gemma4ImageProcessorPil`, which needs Pillow but not torch).
"""

import base64
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx

# Pseudo token ids for image positions in prompt-cache keys only. Far above
# any real vocabulary so they can never collide with a real token.
_IMAGE_KEY_BASE = 1 << 40


def _image_bytes_from_url(url: str) -> bytes:
    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
        except ValueError:
            raise ValueError("Malformed image data URI.")
        # validate=True: the default silently drops non-alphabet characters,
        # turning garbage into b"" that only fails later, deep in the
        # generation thread.
        return base64.b64decode("".join(b64.split()), validate=True)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    raise ValueError(f"Unsupported image URL scheme: {url[:32]}...")


def extract_images(messages: List[Any]) -> List[bytes]:
    """Collect image bytes from OpenAI-style `image_url` content parts (in
    order) and rewrite those parts in place to `{"type": "image"}`."""
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if part.get("type") != "image_url":
                continue
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if not isinstance(url, str):
                raise ValueError("image_url part is missing its url.")
            try:
                images.append(_image_bytes_from_url(url))
            except ValueError:
                raise
            except Exception as e:  # network errors, bad base64, ...
                raise ValueError(f"Could not load image_url: {e}") from e
            content[i] = {"type": "image"}
    return images


class ImageInputs:
    """Gemma 4 image preprocessing + prompt expansion + fusion."""

    def __init__(self, model, processor_config: dict):
        from transformers.models.gemma4.image_processing_pil_gemma4 import (
            Gemma4ImageProcessorPil,
        )

        cfg = processor_config.get("image_processor", processor_config)
        kwargs = {
            k: cfg[k]
            for k in ("patch_size", "max_soft_tokens", "pooling_kernel_size")
            if k in cfg
        }
        self.processor = Gemma4ImageProcessorPil(**kwargs)
        self.model = model
        args = model.args
        self.image_token_id = args.image_token_id
        self.boi_token_id = args.boi_token_id
        self.eoi_token_id = args.eoi_token_id

    def build(
        self, prompt: List[int], images: List[bytes]
    ) -> Tuple[List[int], mx.array, List[int]]:
        """Returns (expanded token ids, input embeddings [L, H], cache key)."""
        from PIL import Image

        n_placeholders = sum(1 for t in prompt if t == self.image_token_id)
        if n_placeholders != len(images):
            raise ValueError(
                f"Chat template produced {n_placeholders} image placeholder(s) "
                f"for {len(images)} image(s)."
            )

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        feats = self.processor(images=pil_images, return_tensors="np")
        n_soft = [int(n) for n in feats["num_soft_tokens_per_image"]]
        keys = [
            _IMAGE_KEY_BASE + int.from_bytes(hashlib.sha256(b).digest()[:5], "big")
            for b in images
        ]

        expanded, cache_key = [], []
        img = 0
        for t in prompt:
            if t == self.image_token_id:
                span = [self.image_token_id] * n_soft[img]
                expanded += [self.boi_token_id] + span + [self.eoi_token_id]
                cache_key += (
                    [self.boi_token_id] + [keys[img]] * n_soft[img] + [self.eoi_token_id]
                )
                img += 1
            else:
                expanded.append(t)
                cache_key.append(t)

        fused, _ = self.model._fuse_multimodal_inputs(
            mx.array(expanded)[None],
            mx.array(feats["pixel_values"]),
            mx.array(feats["image_position_ids"]),
            None,
            None,
        )
        embeddings = fused[0]
        mx.eval(embeddings)

        # The ids generation runs on must have image positions replaced by
        # pad, exactly like the fusion does internally: models with per-layer
        # embeddings (Gemma 4 E2B/E4B, hidden_size_per_layer_input > 0) derive
        # them from these ids, and HF computes them from the pad-substituted
        # ids. Passing the raw image-token ids here instead measurably
        # changed the output (last-token logits off by ~26) and the model
        # answered as if no image had been given.
        pad = self.model.language_model.model.config.pad_token_id
        gen_ids = [pad if t == self.image_token_id else t for t in expanded]
        return gen_ids, embeddings, cache_key


def vision_spans(cache_key: List[int]) -> List[Tuple[int, int]]:
    """[start, end) of each image's soft tokens, from a cache key built by
    ImageInputs.build (its image positions hold pseudo ids >= _IMAGE_KEY_BASE;
    images are always separated by their begin / end tokens)."""
    spans, start = [], None
    for i, t in enumerate(list(cache_key) + [0]):
        if t >= _IMAGE_KEY_BASE and start is None:
            start = i
        elif t < _IMAGE_KEY_BASE and start is not None:
            spans.append((start, i))
            start = None
    return spans


def load_image_inputs(model, model_path) -> Optional[ImageInputs]:
    """ImageInputs for a vision-capable Gemma 4 model, else None."""
    if getattr(model, "model_type", None) != "gemma4":
        return None
    if getattr(model, "vision_tower", None) is None:
        return None
    from .utils import _download

    cfg_path = Path(_download(str(model_path))) / "processor_config.json"
    if not cfg_path.exists():
        return None
    return ImageInputs(model, json.loads(cfg_path.read_text()))
