import numpy as np
import pytest

from conscious_agent import (
    ConsciousAgent, FormalConsciousAgent, MarkovKernel, StochasticMatrix,
    build_decision_kernel, DECISION_STATES, markov, mulberry32, fnv1a32,
)


def test_rng_reference_values():
    r = mulberry32(42)
    assert r.random() == 0.6011037519201636
    assert fnv1a32("") == 0x811C9DC5
    assert fnv1a32("a") == 0xE40C292C
    assert fnv1a32("foobar") == 0xBF9CF968


def test_stationary_periodic_and_analytic():
    r = markov.stationary([[0, 1], [1, 0]])
    assert r["converged"]
    np.testing.assert_allclose(r["pi"], [0.5, 0.5], atol=1e-9)
    P = [[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]]
    np.testing.assert_allclose(markov.stationary(P)["pi"], [0.25, 0.5, 0.25], atol=1e-9)
    np.testing.assert_allclose(markov.solve_stationary(P), [0.25, 0.5, 0.25], atol=1e-9)
    a, b = 0.3, 0.1
    np.testing.assert_allclose(markov.stationary([[1 - a, a], [b, 1 - b]])["pi"], [b / (a + b), a / (a + b)], atol=1e-9)
    assert markov.solve_stationary([[1, 0], [0, 1]]) is None


def test_chain_structure():
    R = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0.3, 0, 0.2, 0.5], [0, 0, 0, 1]]
    assert markov.communicating_classes(R) == [[0, 1], [2], [3]]
    assert markov.closed_classes(R) == [[0, 1], [3]]
    assert not markov.is_irreducible(R)
    assert markov.period([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) == 3
    assert markov.period([[0.5, 0.5], [1, 0]]) == 1
    assert markov.is_ergodic([[0.5, 0.5], [1, 0]]) and not markov.is_ergodic([[0, 1], [1, 0]])
    assert abs(markov.second_eigenvalue_modulus([[0.9, 0.1], [0.5, 0.5]]) - 0.4) < 1e-6
    assert markov.prune_unobserved_rows([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) == []
    assert markov.prune_unobserved_rows([[0, 1, 0], [1, 0, 1], [0, 0, 0]]) == [0, 1]


def test_markov_kernel():
    K = MarkovKernel(["a", "b"], [[0.9, 0.1], [0.5, 0.5]])
    with pytest.raises(ValueError, match="sums to"):
        MarkovKernel(["a"], [[0.5]])
    T = K.tensor(K)
    assert markov.is_stochastic(T.matrix)
    pi = K.stationary()["pi"]
    assert abs(T.stationary()["distribution"]["a|b"] - pi[0] * pi[1]) < 1e-9
    assert abs(K.power(2).prob("a", "a") - (0.81 + 0.05)) < 1e-12
    D = StochasticMatrix(["x", "y"], ["a", "b"], [[1, 0], [0.5, 0.5]])
    assert abs(D.compose(K).prob("y", "a") - 0.7) < 1e-12


def test_decision_kernel():
    with pytest.raises(ValueError, match="<= 1"):
        build_decision_kernel(0.9, 0.2, 0.1)
    with pytest.raises(ValueError, match="<= 1"):
        ConsciousAgent(agent_id="bad", p_stable=0.9, p_lexicon=0.2)
    d = build_decision_kernel().diagnostics()
    assert d["ergodic"] and tuple(d["states"]) == DECISION_STATES

    K = build_decision_kernel(0.6, 0.2, 0.1)
    pi = K.stationary()["distribution"]
    r = mulberry32(11)
    counts = dict.fromkeys(DECISION_STATES, 0)
    s, n = "core", 50000
    for _ in range(n):
        s = K.sample(s, r)
        counts[s] += 1
    assert sum(abs(counts[k] / n - pi[k]) for k in DECISION_STATES) / 2 < 0.015


def _formal(seed):
    return FormalConsciousAgent(
        ["calm", "alert"], ["stay", "move"], ["left", "right"],
        {"left": [[0.9, 0.1], [0.6, 0.4]], "right": [[0.3, 0.7], [0.1, 0.9]]},
        [[0.8, 0.2], [0.2, 0.8]],
        {"stay": [[1, 0], [0, 1]], "move": [[0, 1], [1, 0]]},
        rng=mulberry32(seed),
    )


def test_formal_agent_ergodic_theorem():
    c = _formal(7)
    assert markov.is_stochastic(c.joint_kernel().matrix)
    pi = c.diagnostics()["stationary"]
    counts, w, n = {}, "left", 50000
    for _ in range(n):
        o = c.step(w)
        w = o["w"]
        key = f"{o['x']}|{w}"
        counts[key] = counts.get(key, 0) + 1
    for k, p in pi.items():
        assert abs(counts.get(k, 0) / n - p) < 0.015
    assert c.N == n


def test_formal_agent_combination_roundtrip():
    cc = FormalConsciousAgent.combine(_formal(1), _formal(2))
    assert len(cc.X) == 4 and len(cc.G) == 4
    assert markov.is_stochastic(cc.joint_kernel().matrix)
    again = FormalConsciousAgent.from_dict(cc.to_dict())
    np.testing.assert_allclose(again.joint_kernel().matrix, cc.joint_kernel().matrix)
