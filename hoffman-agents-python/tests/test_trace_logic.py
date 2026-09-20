import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from conscious_agent import decorated, markov, mulberry32, trace

_spec = importlib.util.spec_from_file_location(
    "exp16", Path(__file__).resolve().parents[1] / "examples" / "16_trace_logic_and_decorated_permutations" / "trace_logic.py")
exp16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp16)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fusions-appendix-b.json"


def test_paper_example():
    P = exp16.cycle_kernel(9, [[1, 5, 8], [2], [3, 4], [6], [7, 9]])
    assert decorated.decorated_permutation(P) == [8, 11, 4, 12, 10, 15, 9, 14, 16]


def test_appendix_b_27_of_27_and_14_distinct():
    vertices = json.loads(FIXTURE.read_text())["vertices"]
    assert len(vertices) == 27
    distinct = set()
    for v in vertices:
        sigma = decorated.decorated_permutation(v["M"])
        assert sigma == v["sigma"], v["M"]
        distinct.add(tuple(sigma))
    assert len(distinct) == 14


def test_transient_absorbing_and_probabilities_ignored():
    assert decorated.decorated_permutation([[1, 0], [0.5, 0.5]]) == [3, 2]
    assert decorated.decorated_permutation([[0.99, 0.01], [0.01, 0.99]]) == \
        decorated.decorated_permutation([[0.01, 0.99], [0.99, 0.01]])


def test_covariance_rule():
    perms = list(itertools.permutations(range(6)))
    for P in (exp16.cycle_kernel(6, [[1, 3, 5], [2, 6], [4]]), exp16.cycle_kernel(6, [[1, 2, 3, 4, 5, 6]]),
              exp16.random_kernel(6, mulberry32(3))):
        for p in perms:
            assert decorated.is_covariant(P, p) == exp16.preserves_cyclic_order(P, p)


def test_locally_boolean():
    N = exp16.labelled(["a", "b", "c", "d"], exp16.random_kernel(4, mulberry32(5)))
    windows = exp16.subsets(N["states"])
    traces = [exp16.trace_on(N, w) for w in windows]
    for i, wi in enumerate(windows):
        for j, wj in enumerate(windows):
            assert exp16.trace_leq(traces[i], traces[j]) == all(s in wj for s in wi)


def test_not_a_lattice():
    K1 = exp16.labelled(["a", "b"], [[0.9, 0.1], [0.4, 0.6]])
    K2 = exp16.labelled(["a", "b"], [[0.2, 0.8], [0.7, 0.3]])
    la, lb = exp16.labelled(["a"], [[1.0]]), exp16.labelled(["b"], [[1.0]])
    for l in (la, lb):
        assert exp16.trace_leq(l, K1) and exp16.trace_leq(l, K2)
    assert not (exp16.trace_leq(K1, K2) or exp16.trace_leq(K2, K1) or exp16.trace_leq(la, lb) or exp16.trace_leq(lb, la))


def test_lebesgue_homomorphism():
    r = mulberry32(11)
    for _ in range(50):
        n = 3 + int(r.random() * 4)
        B = exp16.labelled(["a", "b", "c", "d", "e", "f"][:n], exp16.random_kernel(n, r))
        win = [s for s in B["states"] if r.random() < 0.6]
        if not win:
            continue
        A = exp16.trace_on(B, win)
        assert exp16.trace_leq(A, B)
        assert trace.lebesgue_leq(trace.stationary_measure(A["states"], A["P"]), trace.stationary_measure(B["states"], B["P"]))
    nu = {"a": 0.5, "b": 0.5}
    assert trace.lebesgue_leq(nu, {"a": 0.25, "b": 0.25, "c": 0.5})
    assert not trace.lebesgue_leq(nu, {"a": 0.1, "b": 0.3, "c": 0.6})


def test_qualia_kernel_stochastic():
    Q = trace.qualia_kernel([[0.3, 0.7], [1, 0]], [[0.5, 0.5, 0], [0, 0.2, 0.8]], [[1, 0], [0.4, 0.6], [0, 1]])
    assert markov.is_stochastic(Q)


def test_entropy_rate_additive():
    r = mulberry32(17)
    P1, P2 = exp16.random_kernel(3, r), exp16.random_kernel(4, r)
    assert abs(markov.entropy_rate(markov.kron(P1, P2)) - markov.entropy_rate(P1) - markov.entropy_rate(P2)) < 1e-10
    assert abs(markov.entropy_rate([[0.5, 0.5], [0.5, 0.5]]) - np.log(2)) < 1e-12


def test_hitting_and_commute_times():
    H = markov.hitting_times([[0.7, 0.3], [0.2, 0.8]])
    assert abs(H[0, 1] - 1 / 0.3) < 1e-12 and abs(H[1, 0] - 1 / 0.2) < 1e-12
    K = markov.commute_times(exp16.random_kernel(6, mulberry32(2)))
    assert np.allclose(K, K.T)
    for i, j, k in itertools.product(range(6), repeat=3):
        assert K[i, k] <= K[i, j] + K[j, k] + 1e-9
