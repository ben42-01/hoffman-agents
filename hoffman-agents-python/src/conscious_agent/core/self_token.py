from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from ..legacy import self_token as legacy


@dataclass
class SelfTokenState:
    """The "I" attractor.

    v3 lock rule. "I" locks onto meta-state m* when, for
    ``lock_consecutive_required`` consecutive self-observations, all of these
    hold on the closed recurrent class of the meta-state chain:

      1. evidence   - >= ``min_transitions`` transitions in the class and
                      >= ``min_transitions_per_state`` per class state
      2. ergodicity - the class kernel is aperiodic and pi has converged
      3. dominance  - pi(m*) - 1/n >= ``lock_margin`` (or normalised
                      KL(pi || uniform) >= ``kl_threshold``)
      4. occupancy  - >= ``min_occupancy`` of recent observations in the class
      5. stability  - the dominant state equals the previous observation's

    Once locked, "I" unlocks after ``unlock_consecutive_required`` observations
    with dominance < ``unlock_margin`` or occupancy < min_occupancy / 2, but only
    while the evidence criterion holds: locking changes the output mode, which
    creates new meta-states, and a lock is not revised without data.
    """

    token: str = "I"
    referent_meta_state_id: int | None = None
    stationary_prob: float = 0.0
    locked: bool = False
    lock_generation: int | None = None
    lock_threshold: float = 0.25  # 2.x rule, used by math_version='legacy' only
    consecutive_above_threshold: int = 0
    lock_consecutive_required: int = 3
    stability_history: list[float] = field(default_factory=list)
    protection_radius: int = 2
    math_version: str = "v3"
    min_transitions: int = 20
    min_transitions_per_state: float = 2
    lock_margin: float = 0.15
    unlock_margin: float = 0.05
    kl_threshold: float | None = None
    min_occupancy: float = 0.6
    unlock_consecutive_required: int = 3
    consecutive_below_threshold: int = 0
    lock_history: list[dict] = field(default_factory=list)
    last_diagnostics: dict | None = field(default=None, repr=False, compare=False)
    _last_dominant: int | None = field(default=None, repr=False)
    _pending_event: dict | None = field(default=None, repr=False, compare=False)

    LOCK_OPTION_KEYS = (
        "lock_threshold", "lock_consecutive_required", "min_transitions", "min_transitions_per_state",
        "lock_margin", "unlock_margin", "kl_threshold", "min_occupancy", "unlock_consecutive_required",
        "protection_radius",
    )

    def update(self, meta_trie, generation: int) -> None:
        if self.math_version == "legacy":
            return legacy.update(self, meta_trie, generation)

        d = meta_trie.ergodic_diagnostics()
        self.last_diagnostics = d
        previous_dominant = self._last_dominant
        self._last_dominant = d["dominant"]

        if d["class_size"] == 0:
            self.consecutive_above_threshold = 0
            return

        self.stationary_prob = d["dominant_prob"]
        self.stability_history.append(d["dominant_prob"])
        if len(self.stability_history) > 20:
            self.stability_history.pop(0)

        if not self.locked:
            if self.lock_criteria(d, previous_dominant)["met"]:
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.lock_consecutive_required:
                    self._lock(d["dominant"], generation, d)
            else:
                self.consecutive_above_threshold = 0
            return

        holds = (not self._has_evidence(d)) or (
            d["dominance"] >= self.unlock_margin and d["occupancy"] >= self.min_occupancy / 2
        )
        if holds:
            self.consecutive_below_threshold = 0
            if self._has_evidence(d):
                self.referent_meta_state_id = d["dominant"]
        else:
            self.consecutive_below_threshold += 1
            if self.consecutive_below_threshold >= self.unlock_consecutive_required:
                self._unlock(generation, d)

    def _has_evidence(self, d: dict) -> bool:
        return (d["n_transitions"] >= self.min_transitions
                and d["n_transitions"] >= self.min_transitions_per_state * d["class_size"])

    def lock_criteria(self, d: dict, previous_dominant: Any = "__unset__") -> dict:
        if previous_dominant == "__unset__":
            previous_dominant = self._last_dominant
        concentrated = d["dominance"] >= self.lock_margin or (
            self.kl_threshold is not None and d["normalized_kl"] >= self.kl_threshold
        )
        checks = {
            "evidence": self._has_evidence(d),
            "ergodic": bool(d["converged"] and d["aperiodic"]),
            "dominance": bool(concentrated),
            "occupancy": d["occupancy"] >= self.min_occupancy,
            "stable": previous_dominant is not None and previous_dominant == d["dominant"],
        }
        return {"met": all(checks.values()), "checks": checks}

    def _lock(self, meta_state_id: int, generation: int, d: dict | None = None) -> None:
        self.locked = True
        self.referent_meta_state_id = meta_state_id
        self.lock_generation = generation
        self.consecutive_below_threshold = 0
        event = {"event": "lock", "generation": generation, "referent": meta_state_id,
                 "stationary_prob": d["dominant_prob"] if d else self.stationary_prob}
        self.lock_history.append(event)
        self._pending_event = event

    def _unlock(self, generation: int, d: dict | None = None) -> None:
        event = {"event": "unlock", "generation": generation, "referent": self.referent_meta_state_id,
                 "stationary_prob": d["dominant_prob"] if d else self.stationary_prob}
        self.locked = False
        self.referent_meta_state_id = None
        self.consecutive_above_threshold = 0
        self.consecutive_below_threshold = 0
        self.lock_history.append(event)
        self._pending_event = event

    def consume_event(self) -> dict | None:
        """Return and clear the most recent lock/unlock event, if any."""
        e, self._pending_event = self._pending_event, None
        return e

    def is_stable(self) -> bool:
        return self.locked

    is_locked = is_stable

    def stability_score(self) -> float:
        """1 - std(recent dominant stationary probabilities)."""
        if len(self.stability_history) < 2:
            return 0.0
        return 1.0 - min(float(np.std(self.stability_history)), 1.0)

    # Deprecated alias: despite the name this returns a stability score.
    stationary_variance = stability_score

    def protected_nodes(self, meta_trie) -> set[int]:
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

    def to_dict(self) -> dict:
        skip = {"last_diagnostics", "_last_dominant", "_pending_event"}
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        out["stability_history"] = list(self.stability_history)
        out["lock_history"] = [dict(e) for e in self.lock_history]
        out["last_dominant"] = self._last_dominant
        return out

    @staticmethod
    def from_dict(data: dict) -> SelfTokenState:
        names = {f.name for f in fields(SelfTokenState)} - {"last_diagnostics", "_last_dominant", "_pending_event"}
        kwargs = {k: v for k, v in data.items() if k in names}
        kwargs.setdefault("math_version", "legacy")
        st = SelfTokenState(**kwargs)
        st.stability_history = list(data.get("stability_history", []))
        st.lock_history = [dict(e) for e in data.get("lock_history", [])]
        st._last_dominant = data.get("last_dominant")
        return st

    @classmethod
    def lock_options(cls, src: Any) -> dict:
        get = src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d))
        return {k: get(k) for k in cls.LOCK_OPTION_KEYS if get(k) is not None}
