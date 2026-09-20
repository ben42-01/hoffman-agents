import numpy as np
import pytest

from conscious_agent import ConsciousAgent, combine, fuse, product_kernel, experience_space_distance, markov
from conscious_agent.io import serialize, deserialize
from helpers import HomeWorld


def _trained(agent_id, seed, steps=900):
    a = ConsciousAgent(agent_id=agent_id, seed=seed, world=HomeWorld(seed, 0.9, 8))
    a.run(steps)
    return a


@pytest.fixture(scope="module")
def abc():
    return _trained("A", 1), _trained("B", 2), _trained("C", 3)


def _counts(a):
    ids, counts = a.experience.meta_trie.transition_counts()
    return ids, counts.tolist()


def test_identity_commutative_associative(abc):
    A, B, C = abc
    assert combine(A, B).agent_id == combine(B, A).agent_id
    assert combine(combine(A, B), C).agent_id == combine(A, combine(B, C)).agent_id


def test_symmetric_structure_and_fresh_lock(abc):
    A, B, _ = abc
    AB, BA = combine(A, B), combine(B, A)
    assert sorted(AB.experience.meta_trie._registry) == sorted(BA.experience.meta_trie._registry)
    assert [e.to_state for e in AB.experience.trace_buffer] == [e.to_state for e in BA.experience.trace_buffer]
    assert A.is_i_locked and B.is_i_locked and not AB.is_i_locked
    assert AB.experience.meta_trie.ergodic_diagnostics()["class_size"] == 0


def test_nested_provenance_is_collision_free(abc):
    A, B, C = abc
    prov = combine(combine(A, B), C).experience.meta_trie._provenance
    assert len({(p["constituent_id"], p["local_id"]) for p in prov.values()}) == len(prov)


def test_product_kernel(abc):
    A, B, _ = abc
    K = product_kernel(A, B)
    assert markov.is_stochastic(K.matrix)
    pa = A.experience.meta_trie.ergodic_diagnostics()["pi"]
    pb = B.experience.meta_trie.ergodic_diagnostics()["pi"]
    a0, b0 = next(iter(pa)), next(iter(pb))
    assert abs(K.stationary()["distribution"][f"{a0}|{b0}"] - pa[a0] * pb[b0]) < 1e-6


def test_weighted_combination():
    x = ConsciousAgent(agent_id="x", p_stable=0.9, p_lexicon=0.05, p_explore=0.05)
    y = ConsciousAgent(agent_id="y", p_stable=0.5, p_lexicon=0.3, p_explore=0.1)
    xy = combine(x, y, weights=(3, 1))
    assert abs(xy.p_stable - 0.8) < 1e-12
    assert markov.is_stochastic(xy.decision_kernel.matrix)
    with pytest.raises(ValueError, match="non-negative"):
        combine(x, y, weights=(-1, 1))


def test_fuse_restores_constituents(abc, tmp_path):
    A, B, C = abc
    parts = fuse(combine(A, B))
    assert [p.agent_id for p in parts] == ["A", "B"]
    for part, orig in zip(parts, (A, B)):
        assert _counts(part) == _counts(orig)
        assert part.is_i_locked == orig.is_i_locked
        assert part.experience.self_token.referent_meta_state_id == orig.experience.self_token.referent_meta_state_id
    assert parts[0].experience.lexicon is not parts[1].experience.lexicon

    AB = combine(A, B)
    nested = fuse(combine(AB, C))
    ab = next(p for p in nested if p.agent_id == AB.agent_id)
    assert sorted(ab.constituent_ids) == ["A", "B"]
    a2, b2 = fuse(ab)
    assert _counts(a2) == _counts(A) and _counts(b2) == _counts(B)

    path = tmp_path / "ab.soul"
    serialize(AB, str(path))
    a3, _ = fuse(deserialize(str(path)))
    assert _counts(a3) == _counts(A)


def test_kernel_distance(abc):
    A, B, _ = abc
    assert experience_space_distance(A.experience, A.experience, mode="kernel") == 0
    assert 0 <= experience_space_distance(A.experience, B.experience, mode="kernel") <= 1
    with pytest.raises(ValueError, match="unknown mode"):
        experience_space_distance(A.experience, B.experience, mode="nope")
