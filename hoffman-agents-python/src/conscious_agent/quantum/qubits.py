"""Quantum agents: complex linear algebra for small qubit systems, used by
examples/17_quantum_agents.

A quantum conscious agent replaces Hoffman's Markov kernels by their quantum
generalisation: experiences are density matrices, perception and action are
quantum channels (completely positive trace-preserving maps, given by Kraus
operators), and decisions are measurements. A Markov kernel is the special case
of a channel that destroys all coherence.

Port of hoffman-agents-node/src/quantum/qubits.js (numpy complex matrices).
"""
from __future__ import annotations

import math

import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def trace_re(A) -> float:
    """Real part of the trace (the trace of a Hermitian product is real)."""
    return float(np.real(np.trace(A)))


def density_from_state(psi) -> np.ndarray:
    """Pure state |psi><psi| from a (complex) amplitude vector."""
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


def pauli_rotation(H, t: float) -> np.ndarray:
    """exp(-i t H) for H with H^2 = I (a product of Pauli matrices): cos t I - i sin t H."""
    return math.cos(t) * np.eye(len(H), dtype=complex) - 1j * math.sin(t) * H


def conjugate(rho, U) -> np.ndarray:
    return U @ rho @ U.conj().T


def apply_channel(rho, kraus) -> np.ndarray:
    """Channel rho -> sum_k K rho K^dagger."""
    return sum((K @ rho @ K.conj().T for K in kraus), np.zeros_like(rho, dtype=complex))


def on_qubit(ops, which: int, n_qubits: int) -> list[np.ndarray]:
    """Kraus operators lifted to act on qubit `which` of `n_qubits`."""
    out = []
    for K in ops:
        M = None
        for q in range(n_qubits):
            f = K if q == which else PAULI["I"]
            M = f if M is None else np.kron(M, f)
        out.append(M)
    return out


def depolarizing(p: float) -> list[np.ndarray]:
    """With probability p the qubit is replaced by noise."""
    return [math.sqrt(1 - 3 * p / 4) * PAULI["I"], math.sqrt(p / 4) * PAULI["X"],
            math.sqrt(p / 4) * PAULI["Y"], math.sqrt(p / 4) * PAULI["Z"]]


def dephasing(p: float) -> list[np.ndarray]:
    """With probability p the phase is flipped (coherence decays)."""
    return [math.sqrt(1 - p) * PAULI["I"], math.sqrt(p) * PAULI["Z"]]


def markov_channel(P) -> list[np.ndarray]:
    """Kraus operators K_ij = sqrt(P_ij) |j><i| of a Markov kernel P. On diagonal
    (classical) states it acts exactly as P; it destroys all coherence."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    out = []
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                K = np.zeros((n, n), dtype=complex)
                K[j, i] = math.sqrt(P[i, j])
                out.append(K)
    return out


def kraus_deviation(kraus) -> float:
    """Largest deviation of sum_k K^dagger K from the identity (0 for a channel)."""
    S = sum(K.conj().T @ K for K in kraus)
    return float(np.max(np.abs(S - np.eye(len(S)))))


def spin(d) -> np.ndarray:
    """Spin observable along the unit vector d = [x, y, z]."""
    return d[0] * PAULI["X"] + d[1] * PAULI["Y"] + d[2] * PAULI["Z"]


def correlation_matrix(rho) -> np.ndarray:
    """T_ij = Tr(rho sigma_i ⊗ sigma_j); spins along a, b correlate as a^T T b."""
    S = [PAULI["X"], PAULI["Y"], PAULI["Z"]]
    return np.array([[trace_re(rho @ np.kron(a, b)) for b in S] for a in S])


def symmetric_eigen(M) -> dict:
    """Eigenvalues descending; vectors[k] is the unit eigenvector of values[k]."""
    w, V = np.linalg.eigh(np.asarray(M, dtype=float))
    order = np.argsort(-w, kind="stable")
    return {"values": [float(w[k]) for k in order], "vectors": [V[:, k].copy() for k in order]}


def _tt(rho):
    T = correlation_matrix(rho)
    return T, T.T @ T


def max_chsh(rho) -> float:
    """Horodecki criterion (1995): max CHSH over spin measurements = 2 sqrt(m1 + m2),
    m1 >= m2 the two largest eigenvalues of T^T T. Never above 2 sqrt 2."""
    _, TtT = _tt(rho)
    v = symmetric_eigen(TtT)["values"]
    return 2 * math.sqrt(max(0.0, v[0] + v[1]))


def optimal_settings(rho) -> dict:
    """Measurement directions attaining max_chsh (Horodecki's construction)."""
    T, TtT = _tt(rho)
    e = symmetric_eigen(TtT)
    c1, c2 = e["vectors"][0], e["vectors"][1]
    m1, m2 = max(0.0, e["values"][0]), max(0.0, e["values"][1])
    eta = math.atan2(math.sqrt(m2), math.sqrt(m1)) if m1 + m2 > 0 else 0.0

    def unit(v, fallback):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else fallback
    b = math.cos(eta) * c1 + math.sin(eta) * c2
    b2 = math.cos(eta) * c1 - math.sin(eta) * c2
    return {"alice": [unit(T @ c1, c1), unit(T @ c2, c2)], "bob": [b, b2]}


def behaviour(rho, dirs_a, dirs_b):
    """p[a][b][x][y]: Alice measures along dirs_a[a], Bob along dirs_b[b]; x, y = 0 for +1."""
    def proj(d, x):
        return 0.5 * (PAULI["I"] + (1 if x == 0 else -1) * spin(d))
    return [[[[trace_re(rho @ np.kron(proj(dirs_a[a], x), proj(dirs_b[b], y))) for y in (0, 1)]
              for x in (0, 1)] for b in (0, 1)] for a in (0, 1)]


def random_unitary2(r) -> np.ndarray:
    """Product of rotations exp(-i theta P) about all 15 non-identity Pauli
    products, angles uniform in [0, pi) from r (same order as Node)."""
    S = [PAULI["I"], PAULI["X"], PAULI["Y"], PAULI["Z"]]
    U = np.eye(4, dtype=complex)
    for a in range(4):
        for b in range(4):
            if a == 0 and b == 0:
                continue
            U = pauli_rotation(np.kron(S[a], S[b]), math.pi * r.random()) @ U
    return U
