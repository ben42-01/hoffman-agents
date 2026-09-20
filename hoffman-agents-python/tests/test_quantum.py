import importlib.util
import math
from pathlib import Path

import numpy as np

from conscious_agent import bell, mulberry32
from conscious_agent import quantum as q

_spec = importlib.util.spec_from_file_location(
    "exp17", Path(__file__).resolve().parents[1] / "examples" / "17_quantum_agents" / "quantum_agents.py")
exp17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp17)
TSIRELSON = 2 * math.sqrt(2)


def test_bell_state_reaches_tsirelson():
    s = math.sqrt(0.5)
    rho = q.density_from_state([s, 0, 0, s])
    assert abs(q.max_chsh(rho) - TSIRELSON) < 1e-12
    p = q.behaviour(rho, [[0, 0, 1], [1, 0, 0]], [[s, 0, s], [-s, 0, s]])
    assert abs(bell.chsh(bell.correlators(p)) - TSIRELSON) < 1e-12
    assert bell.classify(p) == "quantum"


def test_xx_interaction_closed_form():
    for t in (0, 0.1, 0.4, math.pi / 4, 1.2):
        assert abs(q.max_chsh(exp17.interact(exp17.ZERO2, t)) - 2 * math.sqrt(1 + math.sin(2 * t) ** 2)) < 1e-12


def test_product_and_random_interactions():
    r = mulberry32(9)
    for _ in range(200):
        assert q.max_chsh(exp17.product_state(exp17.random_qubit(r), exp17.random_qubit(r))) <= 2 + 1e-9
        rho = q.conjugate(exp17.product_state(exp17.random_qubit(r), exp17.random_qubit(r)), q.random_unitary2(r))
        assert abs(q.trace_re(rho) - 1) < 1e-12
        m = exp17.measured(rho)
        assert q.max_chsh(rho) <= TSIRELSON + 1e-9
        assert abs(m["chsh"] - q.max_chsh(rho)) < 1e-9
        assert m["signalling"] < 1e-12
        assert m["cls"] in ("local", "quantum")


def test_markov_channel_embedding():
    r = mulberry32(4)
    P, p = np.array(exp17.random_stochastic(3, r)), np.array(exp17.random_stochastic(3, r)[0])
    K = q.markov_channel(P)
    assert q.kraus_deviation(K) < 1e-12
    out = q.apply_channel(np.diag(p).astype(complex), K)
    assert np.allclose(np.real(np.diag(out)), p @ P, atol=1e-12)


def test_noise_closed_forms():
    assert q.kraus_deviation(q.depolarizing(0.37)) < 1e-12
    assert q.kraus_deviation(q.dephasing(0.2)) < 1e-12
    rho = rho2 = exp17.interact(exp17.ZERO2, math.pi / 4)
    for k in range(1, 4):
        rho = exp17.local(rho, q.depolarizing(0.1))
        rho2 = exp17.local(rho2, q.dephasing(0.2))
        assert abs(q.max_chsh(rho) - TSIRELSON * 0.9 ** (2 * k)) < 1e-12
        assert abs(q.max_chsh(rho2) - 2 * math.sqrt(1 + 0.6 ** (4 * k))) < 1e-12
    assert abs(q.max_chsh(exp17.local(exp17.interact(exp17.ZERO2, math.pi / 4), q.dephasing(0.5))) - 2) < 1e-12
