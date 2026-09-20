import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conscious_agent import FormalConsciousAgent, bell, markov

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.stem, EXAMPLES / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp13 = _load(Path("13_bell_through_the_headset/bell_through_the_headset.py"))
exp14 = _load(Path("14_spacetime_in_the_headset/spacetime_in_the_headset.py"))


def test_irreversibility_and_dobrushin():
    assert markov.irreversibility(exp14.torus(10, 1)) == pytest.approx(0, abs=1e-12)
    assert markov.irreversibility([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) == pytest.approx(1)
    assert markov.dobrushin([[0.9, 0.1], [0.5, 0.5]]) == pytest.approx(0.4)
    assert markov.dobrushin([[0.3, 0.7], [0.3, 0.7]]) == pytest.approx(0)


def test_spectral_dimension_lattices_and_products():
    assert markov.spectral_dimension(exp14.torus(200, 1), t_min=4, t_max=64)["dimension"] == pytest.approx(1, abs=0.1)
    assert markov.spectral_dimension(exp14.torus(30, 2), t_min=4, t_max=64)["dimension"] == pytest.approx(2, abs=0.1)
    ring = exp14.torus(40, 1)
    assert markov.spectral_dimension(np.kron(ring, ring), t_min=4, t_max=64)["dimension"] == pytest.approx(2, abs=0.15)
    with pytest.raises(ValueError, match="unknown"):
        markov.return_probabilities(exp14.torus(5, 1), start_distribution="x")


def test_dimension_adds_under_combination():
    agent = exp14.ring_agent(10)
    for k in (1, 2, 3):
        if k > 1:
            agent = FormalConsciousAgent.combine(agent, exp14.ring_agent(10))
        d = markov.spectral_dimension(agent.joint_kernel().matrix, t_min=2, t_max=10)["dimension"]
        assert d == pytest.approx(k, abs=0.2)


def test_interaction_binds_dimensions():
    def dim(c):
        Q = exp14.coupled_rings(8, c)
        pi = markov.stationary(Q)["pi"]
        return markov.spectral_dimension(Q, t_min=2, t_max=8, start_distribution="stationary", pi=pi,
                                         max_starts=64, lazy=False)["dimension"]
    assert dim(0) > 2.4 and dim(0.9) < 1.5


def test_bell_classification():
    s = 2 ** -0.5
    assert bell.classify(bell.from_correlators([[1, 1], [1, 1]])) == "local"
    assert bell.classify(bell.from_correlators([[s, s], [s, -s]])) == "quantum"
    assert bell.classify(bell.from_correlators([[1, 1], [1, -1]])) == "post-quantum"
    signalling = [[[[0.5 if y == a else 0.0 for y in (0, 1)] for _ in (0, 1)] for _ in (0, 1)] for a in (0, 1)]
    assert bell.signalling(signalling) == pytest.approx(1)


def test_bell_bound_behind_the_headset():
    for seed, st in ((100, 0.0), (102, 0.8), (7, 0.3)):
        net = exp13.hidden_network(seed, st)
        for k in (0, 1, 3):
            Mk = markov.mat_pow(net["Q"], k)
            bound = 2 + 2 * markov.dobrushin(Mk)
            chsh = exp13.optimise_chsh(Mk, net["pi"], True, seed=7 + k)["chsh"]
            assert chsh <= bound + 1e-9
            assert chsh == pytest.approx(bound, abs=1e-6)
        assert exp13.optimise_chsh(markov.mat_pow(net["Q"], 1), net["pi"], False)["chsh"] <= 2 + 1e-9
