"""Trace chains: what an observer restricted to part of a network experiences.

For a Markov kernel P on states V and an observer window S (complement C), the
observer registers the chain only while it is in S. What it registers is again
a Markov chain, the trace (induced, censored) chain

    P_S = P_SS + P_SC (I - P_CC)^(-1) P_CS

("stay in S", or "leave, wander hidden in C, come back"). Hoffman, Prakash &
Chattopadhyay's Trace Chain Theorem (2025) states this effective dynamics exists
and is unique; "A is a trace of B" is the order behind their trace logic.
Standard facts: traces are transitive, the trace's stationary distribution is
pi restricted to S, and (Kac) the mean number of network steps between observer
events is 1 / pi(S).

Port of hoffman-agents-node/src/math/trace.js.
"""
from __future__ import annotations

import numpy as np

from . import markov


def _check_window(n: int, S) -> list[int]:
    S = list(S)
    if not S:
        raise ValueError("trace window must be a non-empty list of state indices")
    seen = set()
    for s in S:
        if not isinstance(s, (int, np.integer)) or s < 0 or s >= n:
            raise ValueError(f"window index {s} out of range 0..{n - 1}")
        if s in seen:
            raise ValueError(f"window index {s} repeated")
        seen.add(s)
    return [i for i in range(n) if i not in seen]


def _solve(A: np.ndarray, B: np.ndarray) -> np.ndarray | None:
    if A.size == 0:
        return np.zeros((0, B.shape[1]))
    if abs(np.linalg.det(A)) < 1e-12 or np.linalg.cond(A) > 1e12:
        return None
    return np.linalg.solve(A, B)


def trace_chain(P, S) -> np.ndarray:
    """Trace chain of P on the window S (list of indices; result follows S's order)."""
    P = markov.as_matrix(P)
    S = [int(s) for s in S]
    C = _check_window(len(P), S)
    PSS = P[np.ix_(S, S)]
    if not C:
        return PSS.copy()
    Z = _solve(np.eye(len(C)) - P[np.ix_(C, C)], P[np.ix_(C, S)])
    if Z is None:
        raise ValueError("no trace: from some hidden state the chain never returns to the window")
    return PSS + P[np.ix_(S, C)] @ Z


def is_trace_of(PA, PB, S, tol: float = 1e-9) -> bool:
    T = trace_chain(PB, S)
    PA = markov.as_matrix(PA)
    return T.shape == PA.shape and bool(np.all(np.abs(T - PA) <= tol))


def clock_rate(pi, S) -> float:
    """Fraction of network steps at which the observer registers an event (= pi(S))."""
    return float(sum(pi[i] for i in S))


def mean_return_time(P, S, pi=None) -> float:
    """Mean network steps between observer events, from hitting times (Kac: 1/pi(S))."""
    P = markov.as_matrix(P)
    S = [int(s) for s in S]
    C = _check_window(len(P), S)
    if pi is None:
        pi = markov.stationary(P)["pi"]
    pi_s = clock_rate(pi, S)
    h = np.zeros(0)
    if C:
        sol = _solve(np.eye(len(C)) - P[np.ix_(C, C)], np.ones((len(C), 1)))
        if sol is None:
            raise ValueError("no return: the chain can avoid the window forever")
        h = sol[:, 0]
    return float(sum(pi[s] / pi_s * (1 + (P[s, C] @ h if C else 0.0)) for s in S))


def conditional_entropy_profile(P, pi, n_max: int) -> list[float]:
    """H(X_n | X_0) for X_0 ~ pi, n = 0..n_max, in nats (non-decreasing for stationary chains)."""
    P = markov.as_matrix(P)
    M = np.eye(len(P))
    out = []
    for _ in range(n_max + 1):
        out.append(float(sum(pi[i] * markov.entropy(M[i]) for i in range(len(P)))))
        M = M @ P
    return out


# ────────────── trace logic (Hoffman & Prakash, "Traces of Consciousness") ──────────────

def qualia_kernel(D, A, P) -> np.ndarray:
    """Qualia kernel Q = D A P (D: X->G, A: G->W, P: W->X): experience-to-experience dynamics."""
    return markov.as_matrix(D) @ markov.as_matrix(A) @ markov.as_matrix(P)


def lebesgue_leq(nu: dict, mu: dict, tol: float = 1e-9) -> bool:
    """Lebesgue order on finite labelled measures (dict label -> prob): nu <=_L mu iff
    supp(nu) lies in supp(mu) and mu restricted to supp(nu) is proportional to nu
    (Bennett, Hoffman & Murthy's Lebesgue logic; conditioning mu on supp(nu) gives nu)."""
    mass = 0.0
    for label, p in nu.items():
        if p <= tol:
            continue
        if label not in mu:
            return False
        mass += mu[label]
    if not mass > 0:
        return False
    return all(abs(mu.get(label, 0.0) / mass - p) <= tol for label, p in nu.items())


def stationary_measure(states, P) -> dict:
    """The stationary measure of a labelled kernel as dict label -> prob."""
    pi = markov.stationary(P)["pi"]
    return {s: float(pi[i]) for i, s in enumerate(states)}
