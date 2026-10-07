# Copyright © 2026 Apple Inc.

import mlx.core as mx

from .qwen3_5 import Model as Qwen3_5Model
from .qwen3_5 import ModelArgs  # noqa: F401  (the loader reads module.ModelArgs)


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        # The fused experts become switch_mlp's gate / up / down; the rest
        # (prefixes, the vision tower, MTP, norms) is Qwen3.5's.
        weights = dict(weights)
        for l in range(self.language_model.args.num_hidden_layers):
            for base in ("model.language_model", "language_model.model", "model"):
                prefix = f"{base}.layers.{l}.mlp"
                gate_up = weights.pop(f"{prefix}.experts.gate_up_proj", None)
                if gate_up is None:
                    continue
                mid = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )
        # The MTP head's experts come one tensor per expert.
        n = self.language_model.args.num_experts
        for l in range(self.language_model.args.mtp_num_hidden_layers):
            for base in ("mtp", "language_model.mtp"):
                prefix = f"{base}.layers.{l}.mlp"
                if f"{prefix}.experts.0.gate_proj.weight" not in weights:
                    continue
                for m in ("gate_proj", "up_proj", "down_proj"):
                    weights[f"{prefix}.switch_mlp.{m}.weight"] = mx.stack(
                        [weights.pop(f"{prefix}.experts.{e}.{m}.weight") for e in range(n)]
                    )
        return super().sanitize(weights)
