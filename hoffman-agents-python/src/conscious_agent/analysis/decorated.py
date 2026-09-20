"""Decorated permutations of Markov chains (Hoffman, Prakash & Prentner,
"Fusions of Consciousness", Entropy 2023, Definitions 1-2).

A decorated permutation on {1..n} is a map sigma: {1..n} -> {1..2n} with
a <= sigma(a) <= a + n and sigma(a) mod n a permutation. The paper maps a
Markov kernel on states 1..n (in a given order) to one:
  transient a            -> a
  recurrent singleton a  -> a + n          (absorbing state)
  other recurrent a      -> the first b > a such that the cyclic interval
                            (a, a+1, ..., b), read mod n, contains a's class
Decorated permutations index cells of the positive Grassmannian, the geometry
behind the amplituhedron; the paper conjectures this links agent dynamics to
particle physics. The map uses only which states communicate, not the
probabilities, and it depends on the order (labelling) of the states.

Port of hoffman-agents-node/src/analysis/decorated.js.
"""
from __future__ import annotations

import numpy as np

from ..math import markov


def decorated_permutation(P) -> list[int]:
    """P: kernel whose row i is state i+1. Returns sigma as a list (1-based values)."""
    P = markov.as_matrix(P)
    n = len(P)
    recurrent = {i for c in markov.closed_classes(P) for i in c}
    class_of = {i: c for c in markov.communicating_classes(P) for i in c}
    out = []
    for i in range(n):
        a = i + 1
        if i not in recurrent:
            out.append(a)
            continue
        cls = [j + 1 for j in class_of[i]]
        if len(cls) == 1:
            out.append(a + n)
            continue
        for b in range(a + 1, a + n + 1):
            covered = {(c - 1) % n + 1 for c in range(a, b + 1)}
            if all(x in covered for x in cls):
                out.append(b)
                break
        else:
            raise RuntimeError("unreachable: a class is always covered within n steps")
    return out


def relabel(P, perm) -> np.ndarray:
    """The kernel with states relabelled: new state perm[i] is old state i."""
    P = markov.as_matrix(P)
    Q = np.zeros_like(P)
    perm = list(perm)
    Q[np.ix_(perm, perm)] = P
    return Q


def conjugate(sigma, perm) -> list[int]:
    """What sigma would become under a relabelling if it were a property of the
    chain rather than of the numbering (underlying permutation conjugated,
    decoration carried along)."""
    n = len(sigma)
    out = [0] * n
    for i, s in enumerate(sigma):
        a = i + 1
        target = (s - 1) % n
        na, nt = perm[i] + 1, perm[target] + 1
        if s == a:
            out[perm[i]] = na
        elif s == a + n:
            out[perm[i]] = na + n
        else:
            out[perm[i]] = nt if nt > na else nt + n
    return out


def is_covariant(P, perm) -> bool:
    """Does sigma just rename under this relabelling (or change shape)?"""
    return decorated_permutation(relabel(P, perm)) == conjugate(decorated_permutation(P), perm)
