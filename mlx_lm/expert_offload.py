# Copyright © 2026 IPSupport LLC

"""Routed experts left on disk: run a MoE model that doesn't fit memory.

Each ``SwitchGLU`` keeps an LRU cache of its experts, each with its own
arrays. A step waits for the router, reads the experts it needs and doesn't
have with ``pread`` straight from the checkpoint (stacked experts are one
contiguous slice per tensor), and computes:

* a decoding step (a few tokens): each needed expert with
  ``quantized_matmul`` on its own arrays;
* a prompt: ``gather_qmm`` on a stack of just the experts it uses.

The rest of the model loads as usual. ``share`` is the part of each layer's
experts kept in memory.
"""

import fcntl
import glob
import json
import logging
import os
import struct
import sys
import threading
import weakref
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from .models import switch_layers

_F_NOCACHE = 48
_DTYPES = {
    "U32": (np.uint32, mx.uint32),
    "U8": (np.uint8, mx.uint8),
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, mx.float16),
    "F32": (np.float32, mx.float32),
}
_PARTS = [(p, k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]
# Calls with at most this many tokens compute expert by expert.
_DECODE_TOKENS = 4


def _tensor_index(model_path):
    """Tensor name -> (file, absolute offset, bytes, dtype, shape)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        for name, t in header.items():
            a, b = t["data_offsets"]
            out[name] = (f, 8 + n + a, b - a, t["dtype"], t["shape"])
    return out


def _file_prefix(module_path, index):
    """The checkpoint's name for a module: the same "layers.N...." tail and
    the same names before it in any order ("language_model.model" is
    "model.language_model" on disk). None if not exactly one matches."""
    i = module_path.find("layers.")
    if i < 0:
        return None
    tail = module_path[i:]
    head = Counter(filter(None, module_path[:i].split(".")))
    found = []
    for k in index:
        if not k.endswith(".gate_proj.weight"):
            continue
        prefix = k[: -len(".gate_proj.weight")]
        j = prefix.find("layers.")
        if j >= 0 and prefix[j:] == tail and Counter(filter(None, prefix[:j].split("."))) == head:
            found.append(prefix)
    return found[0] if len(found) == 1 else None


def _close(pool, fds):
    pool.shutdown(wait=True)
    for fd in fds.values():
        os.close(fd)


class _Reader:
    """pread from a pool, a file descriptor per file, past the page cache
    on macOS. Closed with the model."""

    def __init__(self, threads=8):
        self.pool = ThreadPoolExecutor(threads)
        self.fds = {}
        self.lock = threading.Lock()
        weakref.finalize(self, _close, self.pool, self.fds)

    def fd(self, f):
        with self.lock:
            if f not in self.fds:
                fd = os.open(f, os.O_RDONLY)
                if sys.platform == "darwin":
                    fcntl.fcntl(fd, _F_NOCACHE, 1)
                self.fds[f] = fd
            return self.fds[f]

    def read(self, f, off, n):
        return os.pread(self.fd(f), n, off)


class _ExpertStore:
    """One SwitchGLU's experts on disk, up to `capacity` in memory."""

    def __init__(self, glu, slices, capacity, reader):
        self.slices = slices
        self.capacity = capacity
        self.reader = reader
        self.cache = OrderedDict()
        self.q = {
            p: (getattr(glu, p).group_size, getattr(glu, p).bits, getattr(glu, p).mode)
            for p in ("gate_proj", "up_proj", "down_proj")
        }
        self.reads = 0
        self.lock = threading.Lock()

    def _read(self, e):
        return e, {
            key: self.reader.read(f, off + e * stride, stride)
            for key, (f, off, stride, dt, shape) in self.slices.items()
        }

    def _arrays(self, raw):
        out = {}
        for key, data in raw.items():
            f, off, stride, dt, shape = self.slices[key]
            a = mx.array(np.frombuffer(data, dtype=_DTYPES[dt][0]).reshape(shape))
            out[key] = a.view(mx.bfloat16) if dt == "BF16" else a
        return out

    def experts(self, used, keep):
        missing = [e for e in used if e not in self.cache]
        got = {}
        if missing:
            for e, raw in self.reader.pool.map(self._read, missing):
                got[e] = self._arrays(raw)
            self.reads += len(missing)
        out = {e: self.cache[e] if e in self.cache else got[e] for e in used}
        for e in used:
            if e in self.cache:
                self.cache.move_to_end(e)
            elif keep:
                self.cache[e] = got[e]
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return out

    def qmm(self, x, arrays, p):
        gs, bits, mode = self.q[p]
        return mx.quantized_matmul(
            x,
            arrays[(p, "weight")],
            arrays[(p, "scales")],
            arrays.get((p, "biases")),
            transpose=True,
            group_size=gs,
            bits=bits,
            mode=mode,
        )


class OffloadedSwitchGLU(switch_layers.SwitchGLU):
    """A SwitchGLU whose experts are read from disk as they're needed."""

    def __call__(self, x, indices):
        with self._expert_store.lock:
            return self._call(x, indices)

    def _call(self, x, indices):
        store = self._expert_store
        idx = np.array(indices)  # what the router chose: needed to read
        flat = idx.reshape(-1, idx.shape[-1])
        used = sorted(set(flat.reshape(-1).tolist()))
        if flat.shape[0] <= _DECODE_TOKENS:
            arrays = store.experts(used, keep=True)
            xs = x.reshape(-1, x.shape[-1])
            done = {}
            rows = []
            for t in range(flat.shape[0]):
                xt = xs[t : t + 1]
                row = []
                for e in flat[t].tolist():
                    if (t, e) not in done:
                        a = arrays[e]
                        h = self.activation(store.qmm(xt, a, "up_proj"), store.qmm(xt, a, "gate_proj"))
                        done[(t, e)] = store.qmm(h, a, "down_proj")
                    row.append(done[(t, e)])
                rows.append(mx.concatenate(row, axis=0))
            return mx.stack(rows).reshape(*idx.shape, x.shape[-1])
        # A prompt (or a batch): a stack of just the experts it uses.
        arrays = store.experts(used, keep=len(used) <= store.capacity)
        saved = {}
        for p, k in _PARTS:
            if (p, k) in store.slices:
                mod = getattr(self, p)
                saved[(p, k)] = getattr(mod, k)
                setattr(mod, k, mx.stack([arrays[e][(p, k)] for e in used]))
        pos = {e: i for i, e in enumerate(used)}
        remap = mx.array(np.vectorize(pos.get, otypes=[np.uint32])(idx))
        try:
            y = super().__call__(x, remap)
            mx.eval(y)
        finally:
            for (p, k), v in saved.items():
                setattr(getattr(self, p), k, v)
        return y


def offload_experts(model, model_path, share):
    """Leave `model`'s routed experts on disk, `share` of each layer's kept
    in memory. Before its parameters are evaluated: the experts' stacked
    arrays are never read. Returns how many layers it applies to."""
    index = _tensor_index(model_path)
    reader = _Reader()
    done = 0
    for name, m in model.named_modules():
        if not isinstance(m, switch_layers.SwitchGLU) or isinstance(m, OffloadedSwitchGLU):
            continue
        if not all(isinstance(getattr(m, p), switch_layers.QuantizedSwitchLinear) for p in ("gate_proj", "up_proj", "down_proj")):
            continue
        if any(getattr(getattr(m, p), "global_scale", None) is not None for p in ("gate_proj", "up_proj", "down_proj")):
            continue  # per-expert nvfp4 scales: not handled here
        if any("bias" in getattr(m, p) for p in ("gate_proj", "up_proj", "down_proj")):
            continue  # experts with a linear bias: kept in memory
        prefix = _file_prefix(name, index)
        if prefix is None:
            continue
        slices = {}
        experts = None
        for p, k in _PARTS:
            key = f"{prefix}.{p}.{k}"
            if key not in index:
                continue
            f, off, nbytes, dt, shape = index[key]
            if dt not in _DTYPES or shape[0] == 0:
                slices = None
                break
            experts = shape[0]
            slices[(p, k)] = (f, off, nbytes // shape[0], dt, shape[1:])
        if not slices:
            continue
        capacity = max(1, int(round(experts * share)))
        store = _ExpertStore(m, slices, capacity, reader)
        for (p, k), (f, off, stride, dt, shape) in slices.items():
            setattr(getattr(m, p), k, mx.zeros((1,), dtype=_DTYPES[dt][1]))
        m.__class__ = OffloadedSwitchGLU
        object.__setattr__(m, "_expert_store", store)
        done += 1
    if done:
        logging.info(f"Experts on disk: {done} MoE layers, {share:.0%} of each layer's experts kept in memory")
    else:
        logging.warning("Experts on disk: no MoE layer found that can be read from the checkpoint")
    return done
