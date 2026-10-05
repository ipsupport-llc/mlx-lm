"""Lookup-only embedding tables read by row from their safetensors file.

The table stays a read-only memory map: only the used rows are in memory,
and the OS can drop them under memory pressure. The rows are gathered on
the CPU, so each call waits for the token ids.
"""

import json
import struct
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_DTYPES = {
    "U32": np.uint32,
    "U8": np.uint8,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # read as bits, viewed as bfloat16 in MLX
}


def safetensors_index(path: str) -> Dict[str, Tuple[int, str, Tuple[int, ...]]]:
    """name -> (byte offset in the file, dtype, shape) for each tensor."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    index = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, _ = info["data_offsets"]
        index[name] = (8 + n + start, info["dtype"], tuple(info["shape"]))
    return index


class _Rows:
    """One tensor of the file as a memory map, not a model parameter."""

    def __init__(self, path: str, offset: int, dtype: str, shape: Tuple[int, ...]):
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype}")
        self.dtype = dtype
        self.shape = shape
        self.map = np.memmap(
            path, dtype=_DTYPES[dtype], mode="r", offset=offset, shape=shape
        )

    def take(self, ids: np.ndarray) -> mx.array:
        rows = mx.array(self.map[ids])
        return rows.view(mx.bfloat16) if self.dtype == "BF16" else rows


class MappedEmbedding(nn.Module):
    def __init__(
        self,
        weight: _Rows,
        scales: Optional[_Rows] = None,
        biases: Optional[_Rows] = None,
        group_size: Optional[int] = None,
        bits: Optional[int] = None,
        mode: str = "affine",
    ):
        super().__init__()
        self._weight = weight
        self._scales = scales
        self._biases = biases
        self._group_size = group_size
        self._bits = bits
        self._mode = mode

    def __call__(self, x: mx.array) -> mx.array:
        ids = np.asarray(x).astype(np.int64, copy=False)
        flat = ids.reshape(-1)
        w = self._weight.take(flat)
        if self._scales is not None:
            w = mx.dequantize(
                w,
                self._scales.take(flat),
                self._biases.take(flat) if self._biases is not None else None,
                group_size=self._group_size,
                bits=self._bits,
                mode=self._mode,
            )
        return w.reshape(*ids.shape, -1)


def map_lookup_tables(model: nn.Module, weights: dict, sources: dict) -> list:
    """Replace the tables named in `lookup_tables` with MappedEmbedding and
    remove their tensors from `weights`. A table that sanitize changed is
    not in the file as loaded, so it stays loaded. Returns the paths."""
    names = set()
    for _, m in model.named_modules():
        names.update(getattr(m, "lookup_tables", ()) or ())
    if not names:
        return []
    indexes = {}
    mapped = []
    for path, module in model.named_modules():
        if path.rsplit(".", 1)[-1] not in names:
            continue
        if not isinstance(module, (nn.Embedding, nn.QuantizedEmbedding)):
            continue
        parts = {}
        for part in ("weight", "scales", "biases"):
            key = f"{path}.{part}"
            if key in weights:
                src = sources.get(id(weights[key]))
                if src is None:
                    break
                parts[part] = src
        else:
            if "weight" not in parts:
                continue
            quantized = isinstance(module, nn.QuantizedEmbedding)
            if quantized != ("scales" in parts):
                continue
            rows = {}
            for part, (file, name) in parts.items():
                if file not in indexes:
                    indexes[file] = safetensors_index(file)
                offset, dtype, shape = indexes[file][name]
                rows[part] = _Rows(file, offset, dtype, shape)
            replacement = MappedEmbedding(
                rows["weight"],
                rows.get("scales"),
                rows.get("biases"),
                group_size=module.group_size if quantized else None,
                bits=module.bits if quantized else None,
                mode=getattr(module, "mode", "affine") if quantized else "affine",
            )
            parent = model
            for part in path.split(".")[:-1]:
                parent = (
                    parent[int(part)]
                    if isinstance(parent, list)
                    else getattr(parent, part)
                )
            setattr(parent, path.rsplit(".", 1)[-1], replacement)
            for part in parts:
                weights.pop(f"{path}.{part}")
            mapped.append(path)
    return mapped
