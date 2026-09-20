from __future__ import annotations

import random

import numpy as np

from ..math import markov
from .markov_kernel import MarkovKernel, StochasticMatrix, product_labels


class FormalConsciousAgent:
    """Finite reference implementation of a conscious agent (Hoffman & Prakash, 2014).

    C = (X, G, P, D, A, N) with
      P : W x X -> Delta(X)   perception   x' ~ P[w](x, .)
      D : X     -> Delta(G)   decision     g  ~ D(x', .)
      A : G x W -> Delta(W)   action       w' ~ A[g](w, .)
      N                       completed perceive-decide-act cycles

    One cycle maps (x, w) to (x', w'); the induced chain on X x W is
      Q((x, w), (x', w')) = P[w](x, x') * sum_g D(x', g) A[g](w, w').
    """

    def __init__(self, X, G, W, P, D, A, x0=None, N: int = 0, rng=random._inst) -> None:
        self.X = list(X)
        self.G = list(G)
        self.W = list(W)
        self.P = _as_kernel_map(P, self.W, self.X, "P")
        self.D = D if isinstance(D, StochasticMatrix) else StochasticMatrix(self.X, self.G, D)
        self.A = _as_kernel_map(A, self.G, self.W, "A")
        _check_labels(self.D.rows, self.X, "D rows")
        _check_labels(self.D.cols, self.G, "D cols")
        self.x = x0 if x0 is not None else self.X[0]
        self.N = N
        self._rng = rng

    def step(self, w) -> dict:
        x = self.P[w].sample(self.x, self._rng)
        g = self.D.sample(x, self._rng)
        nxt = self.A[g].sample(w, self._rng)
        self.x = x
        self.N += 1
        return {"x": x, "g": g, "w": nxt}

    def run(self, w0, n: int) -> list[dict]:
        trajectory, w = [], w0
        for _ in range(n):
            out = self.step(w)
            trajectory.append(out)
            w = out["w"]
        return trajectory

    def joint_kernel(self) -> MarkovKernel:
        nx, nw = len(self.X), len(self.W)
        Q = np.zeros((nx * nw, nx * nw))
        # sum_g D(x', g) A[g](w, w')  as an (nx, nw, nw) tensor
        DA = np.einsum("jg,gkl->jkl", self.D.matrix, np.stack([self.A[g].matrix for g in self.G]))
        for k, w in enumerate(self.W):
            Pw = self.P[w].matrix
            for i in range(nx):
                for j in range(nx):
                    if Pw[i, j] == 0:
                        continue
                    Q[i * nw + k, j * nw:(j + 1) * nw] += Pw[i, j] * DA[j, k]
        return MarkovKernel(product_labels(self.X, self.W), Q)

    def diagnostics(self) -> dict:
        d = self.joint_kernel().diagnostics()
        pi = [d["stationary"][f"{x}|{w}"] for x in self.X for w in self.W]
        mx = markov.marginalize(pi, len(self.X), len(self.W), 0)
        mw = markov.marginalize(pi, len(self.X), len(self.W), 1)
        return {
            **d,
            "experience_marginal": {x: float(mx[i]) for i, x in enumerate(self.X)},
            "world_marginal": {w: float(mw[i]) for i, w in enumerate(self.W)},
        }

    @staticmethod
    def combine(c1: FormalConsciousAgent, c2: FormalConsciousAgent, rng=None) -> FormalConsciousAgent:
        """X = X1 x X2, G = G1 x G2, P[w] = P1[w] (x) P2[w], D = D1 (x) D2,
        A[(g1, g2)] = A1[g1] A2[g2] (actions applied in sequence, agent 1 first)."""
        _check_labels(c1.W, c2.W, "combine: world states")
        P = {w: c1.P[w].tensor(c2.P[w]) for w in c1.W}
        D = c1.D.tensor(c2.D)
        A = {f"{g1}|{g2}": c1.A[g1].compose(c2.A[g2]) for g1 in c1.G for g2 in c2.G}
        return FormalConsciousAgent(product_labels(c1.X, c2.X), product_labels(c1.G, c2.G), c1.W, P, D, A,
                                    x0=f"{c1.x}|{c2.x}", rng=rng or c1._rng)

    def to_dict(self) -> dict:
        return {
            "X": self.X, "G": self.G, "W": self.W,
            "P": {k: v.to_dict() for k, v in self.P.items()},
            "D": self.D.to_dict(),
            "A": {k: v.to_dict() for k, v in self.A.items()},
            "x": self.x, "N": self.N,
        }

    @staticmethod
    def from_dict(data: dict, rng=random._inst) -> FormalConsciousAgent:
        return FormalConsciousAgent(
            data["X"], data["G"], data["W"],
            {k: MarkovKernel.from_dict(v) for k, v in data["P"].items()},
            StochasticMatrix.from_dict(data["D"]),
            {k: MarkovKernel.from_dict(v) for k, v in data["A"].items()},
            x0=data["x"], N=data["N"], rng=rng,
        )


def _as_kernel_map(spec, keys, states, name) -> dict:
    out = {}
    for key in keys:
        if key not in spec:
            raise ValueError(f"{name}: missing kernel for {key}")
        k = spec[key]
        out[key] = k if isinstance(k, MarkovKernel) else MarkovKernel(states, k)
        _check_labels(out[key].states, states, f"{name}[{key}] states")
    return out


def _check_labels(actual, expected, what) -> None:
    if [str(a) for a in actual] != [str(e) for e in expected]:
        raise ValueError(f"{what} mismatch: {actual} vs {expected}")
