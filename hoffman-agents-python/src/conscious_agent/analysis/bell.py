"""Bell-CHSH analysis of a two-party, two-setting, two-outcome experiment.

A behaviour is p[a][b][x][y] = P(x, y | a, b), settings and outcomes in {0, 1}.
Correlators E(a, b) = sum (-1)^(x xor y) p.

Regions (for unbiased marginals):
  local          all CHSH variants <= 2               (Bell / Fine)
  quantum        TLM condition holds, CHSH <= 2*sqrt(2) (Tsirelson)
  post-quantum   no-signalling but outside the quantum set, up to 4 (PR box)
  signalling     one party's marginal depends on the other's setting

Port of hoffman-agents-node/src/analysis/bell.js.
"""
from __future__ import annotations

import math

TSIRELSON = 2 * math.sqrt(2)


def correlators(p) -> list[list[float]]:
    return [[sum((1 if x == y else -1) * p[a][b][x][y] for x in (0, 1) for y in (0, 1)) for b in (0, 1)]
            for a in (0, 1)]


def chsh_variants(E) -> list[float]:
    """The four CHSH expressions (minus sign on each of the four terms)."""
    terms = [E[0][0], E[0][1], E[1][0], E[1][1]]
    return [sum(-e if i == k else e for i, e in enumerate(terms)) for k in range(4)]


def chsh(E) -> float:
    return max(abs(v) for v in chsh_variants(E))


def is_quantum(E, tol: float = 1e-9) -> bool:
    """Tsirelson-Landau-Masanes: unbiased correlators are quantum iff for every
    placement of the minus sign |sum +/- asin E(a, b)| <= pi."""
    a = [math.asin(max(-1.0, min(1.0, e))) for e in (E[0][0], E[0][1], E[1][0], E[1][1])]
    return all(abs(sum(-v if i == k else v for i, v in enumerate(a))) <= math.pi + tol for k in range(4))


def signalling(p) -> float:
    """Largest change in one party's outcome distribution caused by the other's setting."""
    worst = 0.0
    for b in (0, 1):
        for y in (0, 1):
            pb = [p[a][b][0][y] + p[a][b][1][y] for a in (0, 1)]
            worst = max(worst, abs(pb[0] - pb[1]))
    for a in (0, 1):
        for x in (0, 1):
            pa = [p[a][b][x][0] + p[a][b][x][1] for b in (0, 1)]
            worst = max(worst, abs(pa[0] - pa[1]))
    return worst


def classify(p, tol: float = 1e-9) -> str:
    if signalling(p) > tol:
        return "signalling"
    E = correlators(p)
    if chsh(E) <= 2 + tol:
        return "local"
    return "quantum" if is_quantum(E, tol) else "post-quantum"


def from_correlators(E):
    """Behaviour with uniform marginals and the given correlators."""
    return [[[[(1 + (1 if x == y else -1) * E[a][b]) / 4 for y in (0, 1)] for x in (0, 1)]
             for b in (0, 1)] for a in (0, 1)]
