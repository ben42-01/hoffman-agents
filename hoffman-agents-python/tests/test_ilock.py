from conscious_agent import ConsciousAgent, MetaTrie, SelfTokenState
from helpers import ConstantWorld, HomeWorld, SwitchingWorld, noise_world


def _meta_trie(edges, history):
    mt = MetaTrie()
    for a, b in edges:
        mt._trie.insert([a, b])
    mt._history = list(history)
    mt._last_meta_state = history[-1]
    return mt


def test_newest_meta_state_is_not_absorbing():
    d = _meta_trie([[1, 2], [2, 3]], [1, 2, 3]).ergodic_diagnostics()
    assert d["class_size"] == 0 and d["dominant"] is None


def test_periodic_chain():
    d = _meta_trie([[1, 2], [2, 1], [3, 1]], [3, 1, 2, 1]).ergodic_diagnostics()
    assert d["states"] == [1, 2]
    assert abs(d["pi"][1] - 0.5) < 1e-9 and d["period"] == 2 and not d["aperiodic"]


def test_self_transitions_count():
    d = _meta_trie([[1, 1], [1, 1], [1, 2], [2, 1]], [1, 1, 1, 2, 1]).ergodic_diagnostics()
    assert abs(d["pi"][1] - 0.75) < 1e-9 and d["dominant"] == 1 and d["n_transitions"] == 4


def test_meta_state_id_ignores_lock_flag():
    mt = MetaTrie()
    assert mt._compute_meta_state_id((1, 2, 3), 0.1, "core", False) == mt._compute_meta_state_id((1, 2, 3), 0.1, "core", True)


def test_self_loops_recorded():
    agent = ConsciousAgent(agent_id="const", seed=1, world=ConstantWorld())
    agent.run(200)
    mt = agent.experience.meta_trie
    assert mt.trie.lookup([mt.last_meta_state, mt.last_meta_state]).visit_count > 0


def test_null_world_does_not_lock():
    locks = 0
    for seed in range(1, 9):
        a = ConsciousAgent(agent_id=f"n{seed}", seed=seed, world=noise_world(seed))
        a.run(2000)
        locks += bool(a.experience.self_token.lock_history)
    assert locks == 0


def test_attractor_locks():
    for seed in range(1, 9):
        a = ConsciousAgent(agent_id=f"h{seed}", seed=seed, world=HomeWorld(seed, 0.9))
        a.run(1500)
        assert a.is_i_locked, f"seed {seed}"


def test_no_lock_before_evidence():
    a = ConsciousAgent(agent_id="early", seed=1, world=HomeWorld(1, 0.99))
    a.run(61)
    assert not a.is_i_locked


def test_unlock_after_regime_change():
    a = ConsciousAgent(agent_id="s", seed=1, world=SwitchingWorld(1, 1500))
    events = [o.interrupt for o in a.run(6000) if o.interrupt]
    assert events[0]["event"] == "lock" and events[0]["generation"] < 75
    assert any(e["event"] == "unlock" for e in events)
    assert a.experience.self_token.lock_history == events


def test_ergodic_stats_and_config():
    a = ConsciousAgent(agent_id="stats", seed=3, world=HomeWorld(3, 0.95))
    a.run(1500)
    s = a.ergodic_stats()
    assert s["math_version"] == "v3" and s["decision"]["ergodic"] and s["meta"]["class_size"] > 0
    assert sorted(s["lock"]["criteria"]) == ["dominance", "ergodic", "evidence", "occupancy", "stable"]
    assert a.experience.self_token.lock_generation < 100

    c = ConsciousAgent.from_config("cfg", {"agent": {"self_token": {"lock_margin": 0.3, "min_transitions": 40}, "seed": 5}})
    assert c.experience.self_token.lock_margin == 0.3 and c.experience.self_token.min_transitions == 40
    st = SelfTokenState.from_dict(c.experience.self_token.to_dict())
    assert st.lock_margin == 0.3 and st.math_version == "v3"
