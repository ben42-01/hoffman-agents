from __future__ import annotations

import random
from dataclasses import dataclass, field

from .world_state import WorldState

NATIVE_TOKENS = [
    "cross", "boundary", "arrive", "return", "enter", "leave",
    "different", "same", "familiar", "again", "expect", "notice",
]


@dataclass
class SimpleWorld:
    n_states: int = 10
    seed: int = 42
    tokens: list[str] | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._token_pool = self.tokens if self.tokens is not None else NATIVE_TOKENS
        self._states: list[list[str]] = []
        self._current_idx = 0
        self._build_states()
        self._transition_matrix = self._build_transition_matrix()

    def _build_states(self) -> None:
        self._states = []
        for i in range(self.n_states):
            length = self._rng.randint(2, 5)
            tokens = self._rng.choices(self._token_pool, k=length)
            self._states.append(tokens)

    def _build_transition_matrix(self) -> list[list[float]]:
        mat = []
        for i in range(self.n_states):
            row = [self._rng.random() for _ in range(self.n_states)]
            total = sum(row)
            row = [v / total for v in row]
            mat.append(row)
        return mat

    def step(self) -> WorldState:
        row = self._transition_matrix[self._current_idx]
        self._current_idx = self._rng.choices(range(self.n_states), weights=row)[0]
        return WorldState.from_sequence("world", list(self._states[self._current_idx]))

    def reset(self) -> None:
        self._current_idx = self._rng.randint(0, self.n_states - 1)

    @property
    def current_state(self) -> list[str]:
        return list(self._states[self._current_idx])
