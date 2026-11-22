# Copyright © 2026 Apple Inc.

"""Image processors of multimodal models, one module per model."""

from mlx_lm.models.inkling2 import processor as inkling2
from mlx_lm.utils import model_package

from . import muse_spark

# The processor of each model package.
PROCESSORS = {
    "inkling2": inkling2.Processor,
    "muse_spark": muse_spark.Processor,
}


def load_processor(config: dict):
    """The processor of a model, or None."""
    if not config.get("vision_config"):
        return None
    name = config.get("processor_type", model_package(config))
    processor = PROCESSORS.get(name)
    return processor.from_config(config) if processor else None
