from __future__ import annotations

from dataclasses import dataclass, field

from .meta_trie import MetaTrie


@dataclass
class SelfTokenState:
    token: str = "I"
    referent_meta_state_id: int | None = None
    stationary_prob: float = 0.0
    locked: bool = False
    lock_generation: int | None = None
    lock_threshold: float = 0.25
    consecutive_above_threshold: int = 0
    lock_consecutive_required: int = 3
    stability_history: list[float] = field(default_factory=list)
    protection_radius: int = 2

    def update(self, meta_trie: MetaTrie, generation: int) -> None:
        dist = meta_trie.stationary_distribution()
        if not dist:
            return

        dominant = max(dist, key=dist.get)
        prob = min(dist[dominant], 1.0)

        self.stationary_prob = prob
        self.stability_history.append(prob)
        if len(self.stability_history) > 20:
            self.stability_history.pop(0)

        if not self.locked:
            if prob > self.lock_threshold:
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.lock_consecutive_required:
                    self._lock(dominant, generation)
            else:
                self.consecutive_above_threshold = 0
        else:
            self.referent_meta_state_id = dominant

    def _lock(self, meta_state_id: int, generation: int) -> None:
        self.locked = True
        self.referent_meta_state_id = meta_state_id
        self.lock_generation = generation

    def is_stable(self) -> bool:
        return self.locked

    def stationary_variance(self) -> float:
        if len(self.stability_history) < 2:
            return 0.0
        import numpy as np
        std = np.std(self.stability_history)
        return 1.0 - min(std, 1.0)

    is_locked = is_stable
    stability_score = stationary_variance

    def protected_nodes(self, meta_trie: MetaTrie) -> set[int]:
        if self.referent_meta_state_id is None:
            return set()

        protected: set[int] = {self.referent_meta_state_id}

        def _collect_radius(state_id: int, depth: int) -> None:
            if depth > self.protection_radius:
                return
            node = meta_trie.trie.lookup([state_id])
            if node is None:
                return
            for child_state in node.children:
                protected.add(child_state)
                _collect_radius(child_state, depth + 1)

        _collect_radius(self.referent_meta_state_id, 0)
        return protected
