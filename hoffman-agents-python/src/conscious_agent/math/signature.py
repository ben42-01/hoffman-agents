from __future__ import annotations

import numpy as np

from .rng import fnv1a32


def build_transition_signature(
    prev_id: int | None,
    curr_id: int,
    embedding_dim: int,
    math_version: str = "v3",
) -> np.ndarray:
    """One-hot-pair embedding of a transition prev -> curr, L2-normalised.

    v3 hashes the transition with FNV-1a, which is deterministic and identical
    to the Node implementation. 'legacy' keeps the 2.x builtin ``hash()``, which
    is salted per process (set PYTHONHASHSEED to reproduce a 2.x run).
    """
    sig = np.zeros(embedding_dim, dtype=np.float64)
    if prev_id is not None:
        key = f"{prev_id}->{curr_id}"
        h = hash(key) if math_version == "legacy" else fnv1a32(key)
        sig[h % embedding_dim] = 1.0
    sig[curr_id % embedding_dim] = 1.0
    norm = np.linalg.norm(sig)
    if norm > 0:
        sig = sig / norm
    return sig
