from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..math import markov
from ..math.rng import fnv1a32
from ..legacy import meta_trie as legacy
from .experience_trie import ExperienceTrie
from .trace_buffer import TraceBuffer

ID_MASK = 0x0FFFFFFF


@dataclass
class MetaStateSnapshot:
    state_ids: tuple[int, ...]
    mean_prediction_error: float
    timestamp: int


def error_bucket(e: float) -> int:
    return 0 if e < 0.05 else 1 if e < 0.15 else 2 if e < 0.35 else 3 if e < 0.65 else 4


class MetaTrie:
    """The agent's model of its own experiential dynamics.

    A Markov chain over meta-states estimated from the sequence of
    self-observations. Transitions are depth-2 paths [from, to] in ``_trie``;
    the child visit count is the transition count N(from -> to).
    """

    def __init__(
        self,
        snapshot_window: int = 10,
        max_depth: int = 10,
        math_version: str = "v3",
        history_limit: int = 256,
    ) -> None:
        self._trie = ExperienceTrie(max_depth=max_depth)
        self._snapshot_window = snapshot_window
        self._registry: dict[int, MetaStateSnapshot] = {}
        self._last_meta_state: int | None = None
        self._token_registry: dict[int, dict[str, int]] = {}
        self.math_version = math_version
        # Recent meta-state trajectory (v3): selects the attractor currently occupied.
        self._history: list[int] = []
        self._history_limit = history_limit
        # Combination provenance (v3): inherited id -> {constituent_id, local_id};
        # _provenance_tree keeps each constituent's own state for fuse().
        self._provenance: dict[int, dict] = {}
        self._provenance_tree: dict[str, dict] = {}

    def _compute_meta_state_id(
        self,
        state_ids: tuple[int, ...],
        mean_prediction_error: float,
        ergodic_state: str = "idle",
        is_locked: bool = False,
    ) -> int:
        if self.math_version == "legacy":
            return legacy.compute_meta_state_id(state_ids, mean_prediction_error, ergodic_state, is_locked)
        # The lock flag is deliberately not part of the id: hashing it in split
        # the chain into disconnected pre-lock and post-lock components.
        data = json.dumps(
            [[sid % 8 for sid in state_ids[-2:]], error_bucket(mean_prediction_error), ergodic_state or "idle"],
            separators=(",", ":"),
        )
        mid = fnv1a32(data) & ID_MASK
        salt = 1
        while mid in self._provenance:
            mid = fnv1a32(f"{data}#{salt}") & ID_MASK
            salt += 1
        return mid

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

        if self.math_version == "legacy":
            if self._last_meta_state is not None and meta_id != self._last_meta_state:
                self._trie.insert([self._last_meta_state, meta_id])
        else:
            # Self-transitions are real dwell time in an attractor and must count.
            if self._last_meta_state is not None:
                self._trie.insert([self._last_meta_state, meta_id])
            self._history.append(meta_id)
            if len(self._history) > self._history_limit:
                self._history.pop(0)

        self._last_meta_state = meta_id
        return meta_id

    def get_meta_state_snapshot(self, meta_state_id: int) -> MetaStateSnapshot | None:
        return self._registry.get(meta_state_id)

    def transition_counts(self, include_inherited: bool = False) -> tuple[list[int], np.ndarray]:
        """Transition counts between meta-states (ids sorted ascending)."""
        def keep(i: int) -> bool:
            return include_inherited or i not in self._provenance

        edges = []
        ids: set[int] = set()
        for frm, node in self._trie.root.children.items():
            if not keep(frm):
                continue
            for to, child in node.children.items():
                if not keep(to) or child.visit_count <= 0:
                    continue
                edges.append((frm, to, child.visit_count))
                ids.add(frm)
                ids.add(to)
        sorted_ids = sorted(ids)
        idx = {sid: i for i, sid in enumerate(sorted_ids)}
        counts = np.zeros((len(sorted_ids), len(sorted_ids)))
        for frm, to, c in edges:
            counts[idx[frm], idx[to]] += c
        return sorted_ids, counts

    def ergodic_diagnostics(self, alpha: float = 0.0, occupancy_window: int = 20,
                            include_inherited: bool = False) -> dict:
        """Ergodic analysis of the empirical meta-state chain.

        1. States with no observed outgoing transition (typically the newest
           meta-state) are pruned, repeatedly, instead of being made absorbing.
        2. The closed (recurrent) class the agent most recently occupied is chosen.
        3. On that irreducible class: kernel, pi, period and mixing estimates.
        """
        ids, counts = self.transition_counts(include_inherited)
        total_transitions = float(counts.sum())

        kept_idx = markov.prune_unobserved_rows(counts)
        if not kept_idx:
            return {
                "states": [], "pi": {}, "pi_array": np.zeros(0), "converged": False,
                "period": None, "aperiodic": False, "ergodic": False, "n_classes": 0, "n_closed_classes": 0,
                "class_size": 0, "n_transitions": 0.0, "total_transitions": total_transitions,
                "pruned_states": len(ids), "lambda2": None, "relaxation_time": None, "mixing_time": None,
                "entropy": 0.0, "kl_from_uniform": 0.0, "normalized_kl": 0.0,
                "dominant": None, "dominant_prob": 0.0, "dominance": 0.0, "occupancy": 0.0,
            }

        kept_ids = [ids[i] for i in kept_idx]
        C = markov.sub_matrix(counts, kept_idx)

        classes = markov.communicating_classes(C)
        closed = markov.closed_classes(C)
        class_of = {kept_ids[i]: k for k, cls in enumerate(closed) for i in cls}
        chosen = None
        for mid in reversed(self._history):
            if mid in class_of:
                chosen = class_of[mid]
                break
        if chosen is None:
            best = -1.0
            for k, cls in enumerate(closed):
                mass = float(C[np.ix_(cls, cls)].sum())
                if mass > best:
                    best, chosen = mass, k
        cls = closed[chosen]
        states = [kept_ids[i] for i in cls]

        class_counts = markov.sub_matrix(C, cls)
        n_transitions = float(class_counts.sum())
        P, _ = markov.normalize_rows(class_counts, alpha)
        st = markov.stationary(P)
        pi = st["pi"]
        per = markov.period(P, 0)
        mixing = markov.mixing_time_estimate(P, pi=pi)
        n = len(states)

        dominant_idx = markov.argmax_stable(pi)
        kl = markov.kl_from_uniform(pi)
        in_class = set(states)
        recent = self._history[-occupancy_window:]
        occupancy = sum(1 for m in recent if m in in_class) / len(recent) if recent else 0.0

        return {
            "states": states,
            "pi": {s: float(pi[i]) for i, s in enumerate(states)},
            "pi_array": pi,
            "converged": st["converged"],
            "period": per,
            "aperiodic": per == 1,
            "ergodic": per == 1,
            "n_classes": len(classes),
            "n_closed_classes": len(closed),
            "class_size": n,
            "n_transitions": n_transitions,
            "total_transitions": total_transitions,
            "pruned_states": len(ids) - len(kept_ids),
            **mixing,
            "entropy": markov.entropy(pi),
            "kl_from_uniform": kl,
            "normalized_kl": 1.0 if n == 1 else kl / np.log(n),
            "dominant": states[dominant_idx],
            "dominant_prob": float(pi[dominant_idx]),
            "dominance": 1.0 if n == 1 else float(pi[dominant_idx]) - 1.0 / n,
            "occupancy": occupancy,
        }

    def stationary_distribution(self) -> dict[int, float]:
        if self.math_version == "legacy":
            return legacy.stationary_distribution(self)
        return self.ergodic_diagnostics()["pi"]

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

    @property
    def history(self) -> list[int]:
        return list(self._history)

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
        self._history = []
        self._provenance.clear()
        self._provenance_tree = {}
