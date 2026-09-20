import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conscious_agent import MarkovKernel, markov, mulberry32, trace

_spec = importlib.util.spec_from_file_location(
    "exp15", Path(__file__).resolve().parents[1] / "examples" / "15_time_in_the_traces" / "time_in_the_traces.py")
exp15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp15)


def _kernel(n, seed):
    r = mulberry32(seed)
    P = np.array([[r.random() for _ in range(n)] for _ in range(n)])
    return P / P.sum(axis=1, keepdims=True)


P = _kernel(7, 3)
PI = markov.stationary(P)["pi"]


def test_stochastic_and_full_window():
    assert markov.is_stochastic(trace.trace_chain(P, [0, 3, 4]))
    np.testing.assert_allclose(trace.trace_chain(P, list(range(7))), P, atol=1e-15)


def test_transitive():
    outer, inner = [0, 2, 3, 5, 6], [2, 6]
    via = trace.trace_chain(trace.trace_chain(P, outer), [outer.index(s) for s in inner])
    np.testing.assert_allclose(via, trace.trace_chain(P, inner), atol=1e-12)


def test_restricted_stationary_and_kac():
    S = [1, 4, 5]
    pi_t = markov.stationary(trace.trace_chain(P, S))["pi"]
    mass = trace.clock_rate(PI, S)
    np.testing.assert_allclose(pi_t, [PI[s] / mass for s in S], atol=1e-10)
    for window in ([0], [1, 2], [0, 3, 5, 6]):
        assert trace.mean_return_time(P, window, PI) == pytest.approx(1 / trace.clock_rate(PI, window), abs=1e-9)


def test_simulated_observer_matches_trace():
    states, Q = exp15.agent_network(15)
    S = exp15.window_of(states, lambda s: s[2] == "L")
    sim = exp15.simulate(Q, 60000, 1, [S])[0]
    emp, _ = markov.normalize_rows(sim["counts"])
    assert np.max(np.abs(emp - trace.trace_chain(Q, S))) < 0.03
    assert sim["ticks"] / 60000 == pytest.approx(trace.clock_rate(markov.stationary(Q)["pi"], S), abs=0.01)


def test_arrow_of_time():
    cycle = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    assert markov.irreversibility(cycle) == pytest.approx(1)
    assert markov.irreversibility(trace.trace_chain(cycle, [0, 1])) == pytest.approx(0)
    R = exp15.reversible_network(8, 4)
    T = trace.trace_chain(R, [0, 1, 2])
    assert markov.irreversibility(T) == pytest.approx(0, abs=1e-9)
    h = trace.conditional_entropy_profile(T, markov.stationary(T)["pi"], 6)
    assert all(h[i] >= h[i - 1] - 1e-12 for i in range(1, len(h)))


def test_invalid_windows():
    with pytest.raises(ValueError, match="no trace"):
        trace.trace_chain([[1, 0, 0], [0, 1, 0], [0.5, 0, 0.5]], [2])
    for bad, msg in (([], "non-empty"), ([0, 0], "repeated"), ([9], "out of range")):
        with pytest.raises(ValueError, match=msg):
            trace.trace_chain(P, bad)


def test_kernel_methods():
    K = MarkovKernel(["a", "b", "c"], [[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]])
    T = K.trace(["a", "c"])
    assert T.states == ["a", "c"] and K.has_trace(T)
    assert not K.has_trace(MarkovKernel(["a", "c"], [[0.5, 0.5], [0.5, 0.5]]))
