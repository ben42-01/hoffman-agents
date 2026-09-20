"""Finite-state Markov chain utilities.

A kernel P is row-stochastic: P[i, j] >= 0 and sum_j P[i, j] = 1. Distributions
are row vectors, so one step of the chain is mu' = mu P.

Port of hoffman-agents-node/src/math/markov.js (same algorithms and
tie-breaking; floating-point results agree to ~1e-12).
"""
from __future__ import annotations

from math import gcd, log, exp, inf

import numpy as np

from .rng import mulberry32

EPS = 1e-12


def as_matrix(P) -> np.ndarray:
    return np.asarray(P, dtype=np.float64)


def is_stochastic(P, tol: float = 1e-9) -> bool:
    P = as_matrix(P)
    if P.size == 0:
        return True
    return bool(np.all(P >= -tol) and np.all(np.abs(P.sum(axis=1) - 1.0) <= tol))


def normalize_rows(counts, alpha: float = 0.0) -> tuple[np.ndarray, list[int]]:
    """Counts -> kernel with symmetric Dirichlet pseudo-count ``alpha``.

    Rows that remain all-zero stay zero and are listed in ``empty_rows`` so the
    caller decides what an unobserved row means.
    """
    C = as_matrix(counts)
    if C.size == 0:
        return C.reshape(C.shape), []
    totals = C.sum(axis=1) + alpha * C.shape[1]
    P = np.zeros_like(C)
    empty = []
    for i, total in enumerate(totals):
        if total <= 0:
            empty.append(i)
            continue
        P[i] = (C[i] + alpha) / total
    return P, empty


def prune_unobserved_rows(counts) -> list[int]:
    """Indices surviving iterative removal of rows with no outgoing counts.

    An unobserved row must not be treated as a self-loop: that would make it a
    spurious absorbing attractor.
    """
    C = as_matrix(counts)
    n = C.shape[0]
    alive = [True] * n
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not alive[i]:
                continue
            out = sum(C[i, j] for j in range(n) if alive[j])
            if out == 0:
                alive[i] = False
                changed = True
    return [i for i in range(n) if alive[i]]


def sub_matrix(M, idx: list[int]) -> np.ndarray:
    M = as_matrix(M)
    if not idx:
        return np.zeros((0, 0))
    return M[np.ix_(idx, idx)]


def solve_stationary(P) -> np.ndarray | None:
    """Solve pi (P - I) = 0, sum(pi) = 1. None if pi is not unique."""
    P = as_matrix(P)
    n = P.shape[0]
    if n == 0:
        return np.zeros(0)
    A = (P - np.eye(n)).T.copy()
    A[n - 1, :] = 1.0
    b = np.zeros(n)
    b[n - 1] = 1.0
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    pi = np.maximum(np.linalg.solve(A, b), 0.0)
    s = pi.sum()
    if not s > 0:
        return None
    return pi / s


def stationary(P, tol: float = 1e-12, max_iter: int = 10000) -> dict:
    """Stationary distribution of an irreducible kernel.

    Power iteration on the lazy chain (P + I) / 2, which has the same pi but
    is aperiodic, so it converges for periodic P. Falls back to a linear solve.
    """
    P = as_matrix(P)
    n = P.shape[0]
    if n == 0:
        return {"pi": np.zeros(0), "converged": True, "iterations": 0, "method": "empty"}
    if n == 1:
        return {"pi": np.ones(1), "converged": True, "iterations": 0, "method": "trivial"}

    pi = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        nxt = 0.5 * (pi + pi @ P)
        nxt = nxt / nxt.sum()
        diff = float(np.abs(nxt - pi).sum())
        pi = nxt
        if diff < tol:
            return {"pi": pi, "converged": True, "iterations": it, "method": "power"}

    solved = solve_stationary(P)
    if solved is not None:
        return {"pi": solved, "converged": True, "iterations": max_iter, "method": "solve"}
    return {"pi": pi, "converged": False, "iterations": max_iter, "method": "power"}


def communicating_classes(P) -> list[list[int]]:
    """Strongly connected components (iterative Tarjan) of i -> j when P[i, j] > 0."""
    P = as_matrix(P)
    n = P.shape[0]
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    classes: list[list[int]] = []
    counter = 0

    for root in range(n):
        if index[root] != -1:
            continue
        work = [[root, 0]]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            frame = work[-1]
            v = frame[0]
            descended = False
            while frame[1] < n:
                w = frame[1]
                frame[1] += 1
                if not P[v, w] > 0:
                    continue
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append([w, 0])
                    descended = True
                    break
                elif on_stack[w]:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                classes.append(sorted(comp))
    classes.sort(key=lambda c: c[0])
    return classes


def closed_classes(P) -> list[list[int]]:
    P = as_matrix(P)
    n = P.shape[0]
    out = []
    for cls in communicating_classes(P):
        members = set(cls)
        if all(not (P[i, j] > 0) or j in members for i in cls for j in range(n)):
            out.append(cls)
    return out


def is_irreducible(P) -> bool:
    P = as_matrix(P)
    return P.shape[0] > 0 and len(communicating_classes(P)) == 1


def period(P, start: int = 0) -> int:
    """gcd of cycle lengths through the class of ``start`` (0: no cycle)."""
    P = as_matrix(P)
    n = P.shape[0]
    if n == 0:
        return 0
    cls = next(c for c in communicating_classes(P) if start in c)
    members = set(cls)
    level = {start: 0}
    queue = [start]
    g = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in range(n):
            if not P[u, v] > 0 or v not in members:
                continue
            if v not in level:
                level[v] = level[u] + 1
                queue.append(v)
            else:
                g = gcd(g, abs(level[u] + 1 - level[v]))
    return g


def is_ergodic(P) -> bool:
    return is_irreducible(P) and period(P) == 1


def second_eigenvalue_modulus(P, pi=None, iterations: int = 400, window: int = 50) -> float:
    """|lambda_2| by power iteration on x -> xP - sum(xP) pi (Perron part removed)."""
    P = as_matrix(P)
    n = P.shape[0]
    if n <= 1:
        return 0.0
    if pi is None:
        pi = stationary(P)["pi"]
    pi = np.asarray(pi, dtype=np.float64)
    rand = mulberry32(0x5EED)
    x = np.array([rand.random() - 0.5 for _ in range(n)])

    def deflate(v: np.ndarray) -> np.ndarray:
        return v - v.sum() * pi

    x = deflate(x)
    nx = float(np.linalg.norm(x))
    if nx < EPS:
        return 0.0
    x = x / nx
    logs = []
    for _ in range(iterations):
        x = deflate(x @ P)
        nx = float(np.linalg.norm(x))
        if nx < 1e-300:
            return 0.0
        logs.append(log(nx))
        x = x / nx
    tail = logs[-window:]
    return min(1.0, exp(sum(tail) / len(tail)))


def mixing_time_estimate(P, eps: float = 0.25, pi=None) -> dict:
    """t_rel = 1 / (1 - |lambda_2|), t_mix(eps) <= t_rel ln(1 / (eps pi_min))."""
    P = as_matrix(P)
    n = P.shape[0]
    if n <= 1:
        return {"lambda2": 0.0, "relaxation_time": 1.0, "mixing_time": 0.0}
    if pi is None:
        pi = stationary(P)["pi"]
    lambda2 = second_eigenvalue_modulus(P, pi)
    gap = 1.0 - lambda2
    if gap < 1e-12:
        return {"lambda2": lambda2, "relaxation_time": inf, "mixing_time": inf}
    pi_min = min(p for p in pi if p > 0)
    t_rel = 1.0 / gap
    return {"lambda2": lambda2, "relaxation_time": t_rel, "mixing_time": t_rel * log(1.0 / (eps * pi_min))}


def entropy(p) -> float:
    return float(-sum(v * log(v) for v in p if v > 0))


def kl_from_uniform(p) -> float:
    p = list(p)
    return 0.0 if not p else max(0.0, log(len(p)) - entropy(p))


def total_variation(p, q) -> float:
    p, q = list(p), list(q)
    n = max(len(p), len(q))
    p += [0.0] * (n - len(p))
    q += [0.0] * (n - len(q))
    return sum(abs(a - b) for a, b in zip(p, q)) / 2


def kron(A, B) -> np.ndarray:
    """Kronecker product; row i*m + k <-> (i, k)."""
    return np.kron(as_matrix(A), as_matrix(B))


def marginalize(joint, n: int, m: int, axis: int = 0) -> np.ndarray:
    J = np.asarray(joint, dtype=np.float64).reshape(n, m)
    return J.sum(axis=1 - axis)


def irreversibility(P, pi=None) -> float:
    """Normalised net probability flux of a stationary chain.

    sum_{i<j} |pi_i P_ij - pi_j P_ji| / sum_{i<j} (pi_i P_ij + pi_j P_ji):
    0 = reversible (detailed balance, no arrow of time), 1 = every transition
    one-directional. Self-loops carry no flux and are ignored.
    """
    P = as_matrix(P)
    if pi is None:
        pi = stationary(P)["pi"]
    F = np.asarray(pi)[:, None] * P
    upper = np.triu_indices(len(P), k=1)
    a, b = F[upper], F.T[upper]
    total = float((a + b).sum())
    return float(np.abs(a - b).sum()) / total if total > 0 else 0.0


def dobrushin(P) -> float:
    """Dobrushin contraction coefficient max_{i,j} TV(P_i., P_j.).

    TV(mu P, nu P) <= dobrushin(P) * TV(mu, nu): it bounds how much information
    about the starting state survives one step.
    """
    P = as_matrix(P)
    worst = 0.0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            worst = max(worst, 0.5 * float(np.abs(P[i] - P[j]).sum()))
    return worst


def mat_pow(P, k: int) -> np.ndarray:
    return np.linalg.matrix_power(as_matrix(P), k)


def return_probabilities(P, t_max: int = 32, max_starts: int = 64, seed: int = 1, lazy: bool = True,
                         start_distribution: str = "uniform", pi=None) -> dict:
    """Mean return probability R(t) = E_i[(L^t)_ii] of L = (I + P)/2 (or P if not lazy).

    start_distribution 'uniform' averages over states (all, or a deterministic
    sample of max_starts); 'stationary' averages over i ~ pi, which is the right
    choice for chains with drift, where returns from transient states measure
    the drift rather than the geometry.
    """
    P = as_matrix(P)
    n = len(P)
    rng = mulberry32(seed)
    starts = list(range(n))
    if start_distribution == "stationary":
        p = np.asarray(pi if pi is not None else stationary(P)["pi"], dtype=np.float64)
        cdf = np.cumsum(p)
        total = float(cdf[-1])
        picks = []
        for _ in range(max_starts):
            u = rng.random() * total
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if cdf[mid] > u:
                    hi = mid
                else:
                    lo = mid + 1
            picks.append(lo)
        starts = sorted(picks)
    elif start_distribution != "uniform":
        raise ValueError(f"unknown start_distribution '{start_distribution}'")
    elif n > max_starts:
        for i in range(n - 1, 0, -1):
            j = int(rng.random() * (i + 1))
            starts[i], starts[j] = starts[j], starts[i]
        starts = sorted(starts[:max_starts])
    L = 0.5 * (np.eye(n) + P) if lazy else P
    V = np.zeros((len(starts), n))
    V[np.arange(len(starts)), starts] = 1.0
    R = np.zeros(t_max + 1)
    R[0] = 1.0
    for t in range(1, t_max + 1):
        V = V @ L
        R[t] = float(V[np.arange(len(starts)), starts].mean())
    return {"R": R, "starts": len(starts)}


def spectral_dimension(P, t_min: int = 2, t_max: int = 16, max_starts: int = 64, seed: int = 1,
                       lazy: bool = True, start_distribution: str = "uniform", pi=None) -> dict:
    """Spectral dimension d_s from R(t) ~ t^(-d_s/2).

    d_s = 1, 2, 3 on lattices of that dimension; no plateau on expanders (no
    geometry); additive under independent products. Returns the least-squares
    fit over [t_min, t_max] and dyadic local values d_s(t) = -2 ln(R(2t)/R(t))/ln 2.
    ``saturation`` = R(t_max) * n (1 = fully mixed; the window must end before that).
    """
    r = return_probabilities(P, t_max=2 * t_max, max_starts=max_starts, seed=seed, lazy=lazy,
                             start_distribution=start_distribution, pi=pi)
    R = r["R"]
    ts = [t for t in range(t_min, t_max + 1) if R[t] > 0]
    if len(ts) >= 2:
        x = np.log(ts)
        y = np.log([R[t] for t in ts])
        m = len(ts)
        slope = (m * float((x * y).sum()) - float(x.sum()) * float(y.sum())) / (m * float((x * x).sum()) - float(x.sum()) ** 2)
    else:
        slope = float("nan")
    local = []
    t = t_min
    while t <= t_max:
        if R[t] > 0 and R[2 * t] > 0:
            local.append({"t": t, "d": -2 * log(R[2 * t] / R[t]) / log(2)})
        t *= 2
    return {"dimension": -2 * slope, "local": local, "R": R.tolist(), "window": (t_min, t_max),
            "starts": r["starts"], "saturation": float(R[t_max]) * len(P)}


def entropy_rate(P, pi=None) -> float:
    """Entropy rate h = sum_i pi_i H(P_i) in nats: new information per step."""
    P = as_matrix(P)
    if pi is None:
        pi = stationary(P)["pi"]
    return float(sum(pi[i] * entropy(P[i]) for i in range(len(P))))


def hitting_times(P) -> np.ndarray:
    """H[i, j] = E_i[T_j] (steps to first reach j from i; H[j, j] = 0), irreducible chains."""
    P = as_matrix(P)
    n = len(P)
    H = np.zeros((n, n))
    for j in range(n):
        others = [i for i in range(n) if i != j]
        if not others:
            continue
        A = np.eye(len(others)) - P[np.ix_(others, others)]
        if abs(np.linalg.det(A)) < 1e-14:
            raise ValueError("hitting_times: chain is not irreducible")
        H[others, j] = np.linalg.solve(A, np.ones(len(others)))
    return H


def commute_times(P) -> np.ndarray:
    """K[i, j] = E_i[T_j] + E_j[T_i]; a metric on the states of an irreducible chain."""
    H = hitting_times(P)
    return H + H.T


def argmax_stable(values, tol: float = 1e-12) -> int:
    """Index of the maximum, taking the lowest index among near-ties.

    Used wherever the choice feeds back into dynamics, so Node and Python pick
    the same state despite last-bit floating-point differences.
    """
    values = list(values)
    best = max(values)
    return next(i for i, v in enumerate(values) if v >= best - tol)
