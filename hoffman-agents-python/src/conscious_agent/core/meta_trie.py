from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .experience_trie import ExperienceTrie
from .trace_buffer import TraceBuffer


@dataclass
class MetaStateSnapshot:
    state_ids: tuple[int, ...]
    mean_prediction_error: float
    timestamp: int


class MetaTrie:
    def __init__(
        self,
        snapshot_window: int = 10,
        max_depth: int = 10,
    ) -> None:
        self._trie = ExperienceTrie(max_depth=max_depth)
        self._snapshot_window = snapshot_window
        self._registry: dict[int, MetaStateSnapshot] = {}
        self._last_meta_state: int | None = None
        self._token_registry: dict[int, dict[str, int]] = {}

    @staticmethod
    def _compute_meta_state_id(
        state_ids: tuple[int, ...],
        mean_prediction_error: float,
        ergodic_state: str = "idle",
        is_locked: bool = False,
    ) -> int:
        error_bucket = (
            0 if mean_prediction_error < 0.05 else
            1 if mean_prediction_error < 0.15 else
            2 if mean_prediction_error < 0.35 else
            3 if mean_prediction_error < 0.65 else 4
        )
        coarse_states = tuple(sid % 8 for sid in state_ids[-2:])
        data = str((coarse_states, error_bucket, ergodic_state, is_locked))
        hash_bytes = hashlib.sha256(data.encode()).digest()
        return int.from_bytes(hash_bytes[:8], "big")

    def observe_self(
        self,
        trace_buffer: TraceBuffer,
        timestamp: int = 0,
        ergodic_state: str = "idle",
        is_locked: bool = False,
    ) -> int:
        recent = trace_buffer.get_recent(self._snapshot_window)
        if not recent:
            return 0

        state_ids = tuple(e.to_state for e in recent)
        mean_error = trace_buffer.prediction_error_mean(window=self._snapshot_window)
        meta_id = self._compute_meta_state_id(state_ids, mean_error, ergodic_state, is_locked)

        if meta_id not in self._registry:
            self._registry[meta_id] = MetaStateSnapshot(
                state_ids=state_ids,
                mean_prediction_error=mean_error,
                timestamp=timestamp,
            )

        if self._last_meta_state is not None and meta_id != self._last_meta_state:
            self._trie.insert([self._last_meta_state, meta_id])

        self._last_meta_state = meta_id
        return meta_id

    def get_meta_state_snapshot(self, meta_state_id: int) -> MetaStateSnapshot | None:
        return self._registry.get(meta_state_id)

    def stationary_distribution(self) -> dict[int, float]:
        all_ids = list(self._registry.keys())
        if not all_ids:
            return {}

        active = set()
        for sid in all_ids:
            node = self._trie.lookup([sid])
            if node is not None and node.children:
                active.add(sid)
        if self._last_meta_state is not None:
            active.add(self._last_meta_state)

        if not active:
            return {}

        all_ids = sorted(active)
        idx = {sid: i for i, sid in enumerate(all_ids)}
        n = len(all_ids)
        P = np.zeros((n, n))

        for state_id in all_ids:
            node = self._trie.lookup([state_id])
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

    def dominant_meta_state(self) -> int | None:
        dist = self.stationary_distribution()
        if not dist:
            return None
        return max(dist, key=dist.get)

    @property
    def trie(self) -> ExperienceTrie:
        return self._trie

    @property
    def last_meta_state(self) -> int | None:
        return self._last_meta_state

    @property
    def registry_size(self) -> int:
        return len(self._registry)

    def record_token(self, meta_state_id: int, token: str) -> None:
        if meta_state_id not in self._token_registry:
            self._token_registry[meta_state_id] = {}
        self._token_registry[meta_state_id][token] = (
            self._token_registry[meta_state_id].get(token, 0) + 1
        )

    def predict_token(self, meta_state_id: int, min_observations: int = 3) -> str | None:
        counts = self._token_registry.get(meta_state_id)
        if not counts:
            return None
        best_token = max(counts, key=counts.get)
        if counts[best_token] >= min_observations:
            return best_token
        return None

    def clear(self) -> None:
        self._trie.clear()
        self._registry.clear()
        self._last_meta_state = None
        self._token_registry.clear()
