# Copyright © 2026 Apple Inc.

from .qwen3_5 import Model as Qwen3_5Model
from .qwen3_5 import ModelArgs  # noqa: F401  (the loader reads module.ModelArgs)


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        # The fused experts become switch_mlp's gate / up / down; the rest
        # (prefixes, the vision tower, MTP, norms) is Qwen3.5's.
        weights = dict(weights)
        for l in range(self.language_model.args.num_hidden_layers):
            for base in ("model.language_model", "language_model.model"):
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
        return super().sanitize(weights)
