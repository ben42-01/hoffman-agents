from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from ..math import markov
from ..math import trace as _trace


def product_labels(a: Sequence, b: Sequence) -> list[str]:
    return [f"{x}|{y}" for x in a for y in b]


class StochasticMatrix:
    """A stochastic matrix K(r, c) from row labels to column labels."""

    def __init__(self, rows: Sequence, cols: Sequence, matrix, tol: float = 1e-9) -> None:
        self.rows = list(rows)
        self.cols = list(cols)
        self.matrix = np.array(matrix, dtype=np.float64).reshape(len(self.rows), len(self.cols))
        self._row_index = {s: i for i, s in enumerate(self.rows)}
        self._col_index = {s: i for i, s in enumerate(self.cols)}
        self.validate(tol)

    def validate(self, tol: float = 1e-9) -> bool:
        for i, row in enumerate(self.matrix):
            if np.any(np.isnan(row)) or np.any(row < -tol):
                raise ValueError(f"row {self.rows[i]} has negative or NaN entries")
            s = float(row.sum())
            if abs(s - 1.0) > tol:
                raise ValueError(f"row {self.rows[i]} sums to {s}, expected 1")
        return True

    def row_index(self, label) -> int:
        if label not in self._row_index:
            raise KeyError(f"unknown row state {label}")
        return self._row_index[label]

    def prob(self, frm, to) -> float:
        j = self._col_index.get(to)
        return 0.0 if j is None else float(self.matrix[self.row_index(frm), j])

    def distribution(self, frm) -> dict:
        row = self.matrix[self.row_index(frm)]
        return {c: float(row[j]) for j, c in enumerate(self.cols)}

    def sample(self, frm, rng=random._inst):
        """Inverse-CDF sampling; consumes exactly one rng draw (same as Node)."""
        row = self.matrix[self.row_index(frm)]
        r = rng.random()
        cumulative = 0.0
        for j, p in enumerate(row):
            cumulative += p
            if r < cumulative:
                return self.cols[j]
        for j in range(len(row) - 1, -1, -1):
            if row[j] > 0:
                return self.cols[j]
        return self.cols[-1]

    def compose(self, other: StochasticMatrix) -> StochasticMatrix:
        if self.cols != other.rows:
            raise ValueError("compose: column states of the first kernel must equal row states of the second")
        return StochasticMatrix(self.rows, other.cols, self.matrix @ other.matrix)

    def tensor(self, other: StochasticMatrix) -> StochasticMatrix:
        return StochasticMatrix(product_labels(self.rows, other.rows), product_labels(self.cols, other.cols),
                                markov.kron(self.matrix, other.matrix))

    def mix(self, other: StochasticMatrix, w: float):
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"mix weight must be in [0, 1], got {w}")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("mix: kernels must share row and column states")
        matrix = w * self.matrix + (1.0 - w) * other.matrix
        if isinstance(self, MarkovKernel):
            return MarkovKernel(self.rows, matrix)
        return StochasticMatrix(self.rows, self.cols, matrix)

    def to_dict(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "matrix": self.matrix.tolist()}

    @staticmethod
    def from_dict(data: dict) -> StochasticMatrix:
        return StochasticMatrix(data["rows"], data["cols"], data["matrix"])

    @staticmethod
    def from_counts(rows: Sequence, cols: Sequence, counts, alpha: float = 0.0) -> StochasticMatrix:
        """Rows with no observations become uniform."""
        P, empty = markov.normalize_rows(counts, alpha)
        for i in empty:
            P[i, :] = 1.0 / len(cols)
        return StochasticMatrix(rows, cols, P)


class MarkovKernel(StochasticMatrix):
    """A Markov kernel on a single finite state space."""

    def __init__(self, states: Sequence, matrix, tol: float = 1e-9) -> None:
        super().__init__(states, states, matrix, tol)

    @property
    def states(self) -> list:
        return self.rows

    def tensor(self, other: MarkovKernel) -> MarkovKernel:
        return MarkovKernel(product_labels(self.states, other.states), markov.kron(self.matrix, other.matrix))

    def compose(self, other: MarkovKernel) -> MarkovKernel:
        return MarkovKernel(self.states, super().compose(other).matrix)

    def trace(self, window) -> MarkovKernel:
        """Trace chain on a subset of states: what an observer who only sees
        ``window`` (state labels) experiences."""
        idx = [self.row_index(label) for label in window]
        return MarkovKernel(list(window), _trace.trace_chain(self.matrix, idx))

    def has_trace(self, other: MarkovKernel, tol: float = 1e-9) -> bool:
        """True if ``other`` (a kernel on a subset of these states) is this kernel's trace."""
        return _trace.is_trace_of(other.matrix, self.matrix, [self.row_index(l) for l in other.states], tol)

    def power(self, k: int) -> MarkovKernel:
        return MarkovKernel(self.states, np.linalg.matrix_power(self.matrix, k))

    def stationary(self) -> dict:
        r = markov.stationary(self.matrix)
        return {"distribution": {s: float(r["pi"][i]) for i, s in enumerate(self.states)},
                "pi": r["pi"], "converged": r["converged"]}

    def diagnostics(self) -> dict:
        r = markov.stationary(self.matrix)
        pi = r["pi"]
        classes = markov.communicating_classes(self.matrix)
        irreducible = len(classes) == 1
        per = markov.period(self.matrix, 0)
        mixing = (markov.mixing_time_estimate(self.matrix, pi=pi) if irreducible
                  else {"lambda2": 1.0, "relaxation_time": float("inf"), "mixing_time": float("inf")})
        return {
            "states": self.states,
            "stationary": {s: float(pi[i]) for i, s in enumerate(self.states)},
            "converged": r["converged"],
            "irreducible": irreducible,
            "period": per if irreducible else None,
            "aperiodic": irreducible and per == 1,
            "ergodic": irreducible and per == 1,
            "classes": [[self.states[i] for i in c] for c in classes],
            "closed_classes": [[self.states[i] for i in c] for c in markov.closed_classes(self.matrix)],
            "entropy": markov.entropy(pi),
            **mixing,
        }

    def to_dict(self) -> dict:
        return {"states": self.states, "matrix": self.matrix.tolist()}

    @staticmethod
    def from_dict(data: dict) -> MarkovKernel:
        return MarkovKernel(data["states"], data["matrix"])

    @staticmethod
    def identity(states: Sequence) -> MarkovKernel:
        return MarkovKernel(states, np.eye(len(states)))

    @staticmethod
    def from_counts(states: Sequence, counts, alpha: float = 0.0) -> MarkovKernel:
        P, empty = markov.normalize_rows(counts, alpha)
        for i in empty:
            P[i, i] = 1.0
        return MarkovKernel(states, P)
