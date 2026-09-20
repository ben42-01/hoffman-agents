"""2.x meta-trie math, preserved verbatim for math_version='legacy'.

Known defects (fixed in v3): the newest meta-state is made absorbing,
self-transitions are never recorded, plain power iteration does not converge
on periodic chains, and the lock flag is hashed into the id.
"""
from __future__ import annotations

import hashlib

import numpy as np


def compute_meta_state_id(state_ids, mean_prediction_error, ergodic_state="idle", is_locked=False) -> int:
    error_bucket = (
        0 if mean_prediction_error < 0.05 else
        1 if mean_prediction_error < 0.15 else
        2 if mean_prediction_error < 0.35 else
        3 if mean_prediction_error < 0.65 else 4
    )
    coarse_states = tuple(sid % 8 for sid in state_ids[-2:])
    data = str((coarse_states, error_bucket, ergodic_state, is_locked))
    hash_bytes = hashlib.sha256(data.encode()).digest()
    return int.from_bytes(hash_bytes[:8], "big") & 0x0FFFFFFF


def stationary_distribution(mt) -> dict[int, float]:
    all_ids = list(mt._registry.keys())
    if not all_ids:
        return {}

    active = set()
    for sid in all_ids:
        node = mt._trie.lookup([sid])
        if node is not None and node.children:
            active.add(sid)
    if mt._last_meta_state is not None:
        active.add(mt._last_meta_state)

    if not active:
        return {}

    all_ids = sorted(active)
    idx = {sid: i for i, sid in enumerate(all_ids)}
    n = len(all_ids)
    P = np.zeros((n, n))

    for state_id in all_ids:
        node = mt._trie.lookup([state_id])
        if node is not None and node.children:
            total = sum(c.visit_count for c in node.children.values())
            if total > 0:
                for child_state, child_node in node.children.items():
                    if child_state in idx:
                        P[idx[state_id], idx[child_state]] = child_node.visit_count / total

    for i in range(n):
        if P[i].sum() == 0:
            P[i, i] = 1.0

    pi = np.ones(n) / n
    for _ in range(1000):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-8:
            break
        pi = pi_new

    return {all_ids[i]: float(pi[i]) for i in range(n)}
