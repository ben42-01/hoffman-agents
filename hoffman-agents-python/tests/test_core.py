from conscious_agent import (
    ConsciousAgent,
    StepOutput,
    SimpleWorld,
    TraceBuffer,
    TraceEvent,
    ExperienceTrie,
    MetaTrie,
    SelfTokenState,
    ExperienceLexicon,
    combine,
    AgentNetwork,
    World,
    WorldBuilder,
    CoinTossWorld,
    SelfWorld,
)

import numpy as np


def test_imports():
    assert ConsciousAgent is not None
    assert StepOutput is not None
    assert SimpleWorld is not None
    assert TraceBuffer is not None
    assert TraceEvent is not None
    assert ExperienceTrie is not None
    assert MetaTrie is not None
    assert SelfTokenState is not None
    assert ExperienceLexicon is not None
    assert combine is not None
    assert AgentNetwork is not None
    assert World is not None
    assert WorldBuilder is not None
    assert CoinTossWorld is not None


def test_trace_buffer():
    buf = TraceBuffer(maxlen=5)
    assert len(buf) == 0
    buf.append(TraceEvent(from_state=0, to_state=1, timestamp=0, prediction=1, prediction_correct=True, prediction_error=0.0))
    assert len(buf) == 1
    assert buf.is_full() is False
    for i in range(10):
        buf.append(TraceEvent(from_state=i, to_state=i+1, timestamp=i, prediction=i+1, prediction_correct=True, prediction_error=0.0))
    assert buf.is_full() is True
    assert len(buf) == 5


def test_trace_buffer_resize_grow():
    buf = TraceBuffer(maxlen=3)
    for i in range(5):
        buf.append(TraceEvent(from_state=i, to_state=i, timestamp=i, prediction=i, prediction_correct=True, prediction_error=0.0))
    assert len(buf) == 3
    buf.resize(10)
    assert buf.maxlen == 10
    assert len(buf) == 3


def test_trace_buffer_resize_shrink():
    buf = TraceBuffer(maxlen=10)
    for i in range(10):
        buf.append(TraceEvent(from_state=i, to_state=i, timestamp=i, prediction=i, prediction_correct=True, prediction_error=0.0))
    assert len(buf) == 10
    buf.resize(3)
    assert buf.maxlen == 3
    assert len(buf) == 3
    assert buf.as_state_sequence() == [7, 8, 9]


def test_trace_buffer_resize_invalid():
    buf = TraceBuffer(maxlen=5)
    import pytest
    with pytest.raises(ValueError, match=">= 1"):
        buf.resize(0)


def test_experience_trie():
    trie = ExperienceTrie(max_depth=5)
    trie.insert([1, 2, 3], prediction_error=0.1)
    trie.insert([1, 2, 3], prediction_error=0.2)
    node = trie.lookup([1, 2, 3])
    assert node is not None
    assert node.visit_count == 2
    import pytest
    assert node.mean_prediction_error == pytest.approx(0.15)
    assert trie.predict_next([1, 2]) == 3


def test_self_token():
    st = SelfTokenState()
    assert st.locked is False
    assert st.token == "I"
    assert st.lock_threshold == 0.25


def test_combine():
    a1 = ConsciousAgent(agent_id="test_1")
    a2 = ConsciousAgent(agent_id="test_2")
    combined = combine(a1, a2)
    assert combined.agent_id != "test_1"
    assert combined.agent_id != "test_2"
    assert combined.cycle_level == 1


def test_combine_nary():
    a1 = ConsciousAgent(agent_id="nary_1")
    a2 = ConsciousAgent(agent_id="nary_2")
    a3 = ConsciousAgent(agent_id="nary_3")
    combined = combine(a1, a2, a3)
    assert combined.cycle_level == 2


def test_combine_nary_5():
    agents = [ConsciousAgent(agent_id=f"N_{i}") for i in range(5)]
    combined = combine(*agents)
    assert combined.cycle_level >= 2


def test_simple_world_step():
    world = SimpleWorld(n_states=5, seed=42)
    state = world.step()
    assert state is not None
    assert state.sequences is not None


def test_agent_step():
    world = SimpleWorld(n_states=5, seed=42)
    agent = ConsciousAgent(agent_id="step_test")
    agent.set_world(world)
    output = agent.step()
    assert isinstance(output, StepOutput)
    assert output.sequence is not None
    assert isinstance(output.sequence_str, str)


def test_agent_run():
    world = SimpleWorld(n_states=5, seed=42)
    agent = ConsciousAgent(agent_id="run_test")
    agent.set_world(world)
    outputs = agent.run(n_steps=100)
    assert len(outputs) == 100


def test_action_distribution():
    agent = ConsciousAgent(agent_id="action_dist_test")
    world = SimpleWorld(n_states=5, seed=42)
    output = agent.step(world.step())
    assert hasattr(output, 'action_distribution')
    assert isinstance(output.action_distribution, dict)


def test_allowable_tokens():
    agent = ConsciousAgent(agent_id="allowable_test")
    agent.set_allowable_tokens({"I", "notice"})
    world = SimpleWorld(n_states=5, seed=42)
    for _ in range(10):
        output = agent.step(world.step())
        for token in output.sequence:
            assert token in ("I", "notice"), f"Unexpected token: {token}"


def test_agent_metrics():
    agent = ConsciousAgent(agent_id="metrics_test")
    m = agent.metrics
    assert "prediction_error" in m
    assert "i_locked" in m
    assert "loop_depth" in m
    assert "output_tokens" in m


def test_network_get_metrics():
    network = AgentNetwork(n_agents=4, seed=42)
    m = network.get_metrics()
    assert m["agent_count"] == 4


def test_network_get_agent_metrics():
    network = AgentNetwork(n_agents=4, seed=42)
    m = network.get_agent_metrics("CA_000")
    assert m is not None
    assert "prediction_error" in m
    assert network.get_agent_metrics("nonexistent") is None


def test_clear_memory():
    agent = ConsciousAgent(agent_id="clear_mem_test")
    agent.step(SimpleWorld(n_states=5, seed=42).step())
    agent.clear_memory()
    assert agent.step_count == 0
    assert agent.generation == 0
    assert len(agent.experience.trace_buffer) == 0


def test_clear_memory_preserves_structure():
    agent = ConsciousAgent(agent_id="clear_struct_test")
    world = SimpleWorld(n_states=5, seed=42)
    for _ in range(10):
        agent.step(world.step())
    trie_before = agent.experience.trie.size()
    assert trie_before > 0
    agent.clear_memory()
    assert agent.experience.trie.size() == trie_before


def test_set_mode_frozen():
    agent = ConsciousAgent(agent_id="frozen_test")
    agent.set_mode("frozen")
    assert agent.mode == "frozen"
    world = SimpleWorld(n_states=5, seed=42)
    size_before = agent.experience.trie.size()
    for _ in range(10):
        agent.step(world.step())
    assert agent.experience.trie.size() == size_before
    assert len(agent.experience.trace_buffer) == 10


def test_set_mode_invalid():
    agent = ConsciousAgent(agent_id="bad_mode")
    import pytest
    with pytest.raises(ValueError, match="Invalid mode"):
        agent.set_mode("invalid")


def test_thaw():
    agent = ConsciousAgent(agent_id="thaw_test")
    agent.set_mode("frozen")
    assert agent.mode == "frozen"
    agent.thaw()
    assert agent.mode == "learning"
    world = SimpleWorld(n_states=5, seed=42)
    size_before = agent.experience.trie.size()
    for _ in range(10):
        agent.step(world.step())
    assert agent.experience.trie.size() > size_before


def test_refreeze():
    agent = ConsciousAgent(agent_id="refreeze_test")
    agent.thaw()
    assert agent.mode == "learning"
    agent.refreeze()
    assert agent.mode == "frozen"


def test_world_builder():
    data = np.random.rand(100, 3)
    builder = WorldBuilder()
    builder.add_feature("f1", normalization="minmax", n_bins=4)
    builder.add_feature("f2", normalization="minmax", n_bins=4)
    builder.add_feature("f3", normalization="minmax", n_bins=4)
    world = builder.build(data)
    assert world.n_states > 0
    assert world.transition_matrix.shape[0] == world.n_states


def test_coin_toss_world():
    world = CoinTossWorld(n_coins=3)
    assert world.n_states == 8
    state = world.step()
    assert state is not None
    assert isinstance(state.get_state_id(), int)


def test_agent_network():
    network = AgentNetwork(n_agents=4, seed=42)
    state = network.step()
    assert "generation" in state
    assert state["generation"] == 0


def test_agent_network_step_all():
    network = AgentNetwork(n_agents=4, seed=42)
    world = SimpleWorld(n_states=5, seed=42).step()
    results = network.step_all(world)
    assert len(results) == 4
    for r in results:
        assert r.sequence is not None


def test_agent_network_agent_list():
    network = AgentNetwork(n_agents=4, seed=42)
    assert len(network.agent_list) == 4
    for a in network.agent_list:
        assert a.agent_id is not None


def test_serialization(tmp_path):
    agent = ConsciousAgent(agent_id="save_test")
    path = str(tmp_path / "test.soul")
    from conscious_agent.io import serialize, deserialize
    serialize(agent, path)
    loaded = deserialize(path)
    assert loaded.agent_id == "save_test"


def test_self_world():
    inner = SimpleWorld(n_states=5, seed=42)
    agent = ConsciousAgent(agent_id="sw_test")
    sw = SelfWorld(inner, agent)
    ws = sw.step()
    assert "self" in ws.sequences
    assert len(ws.sequences["self"]) > 0


def test_self_world_custom_fn():
    inner = SimpleWorld(n_states=5, seed=42)
    agent = ConsciousAgent(agent_id="sw_custom")
    sw = SelfWorld(inner, agent, get_state_fn=lambda a: {"custom": 42.0})
    ws = sw.step()
    assert any(t.startswith("custom:") for t in ws.sequences["self"])


def test_self_world_no_step():
    import pytest
    with pytest.raises(ValueError, match="step"):
        SelfWorld({}, None)


def test_lexicon():
    lex = ExperienceLexicon(embedding_dim=16)
    sig = np.random.rand(16).astype(np.float64)
    sig = sig / np.linalg.norm(sig)
    entry = lex.bind(label="test_word", output_token="test", trace_signature=sig)
    assert entry is not None
    assert lex.vocabulary_size() == 1
    found = lex.lookup_by_label("test_word")
    assert found is not None
    assert found.output_token == "test"


def test_meta_trie():
    mt = MetaTrie(snapshot_window=3, max_depth=5)
    buf = TraceBuffer(maxlen=10)
    for i in range(6):
        buf.append(TraceEvent(from_state=i, to_state=i+1, timestamp=i, prediction=i+1, prediction_correct=True, prediction_error=0.1))
    meta_id = mt.observe_self(buf, timestamp=5)
    assert meta_id > 0
    assert mt.registry_size > 0
    snap = mt.get_meta_state_snapshot(meta_id)
    assert snap is not None


def test_shared_meaning():
    from conscious_agent import SharedMeaningTracker
    tracker = SharedMeaningTracker()
    lex1 = ExperienceLexicon()
    lex2 = ExperienceLexicon()
    sig = np.ones(64, dtype=np.float64) / np.linalg.norm(np.ones(64))
    lex1.bind(label="a", output_token="foo", trace_signature=sig)
    lex2.bind(label="b", output_token="foo", trace_signature=sig)
    result = tracker.snapshot({"a1": lex1, "a2": lex2}, generation=1)
    assert result["sharedness"] > 0.0


def test_predict_next_none_when_no_experience():
    agent = ConsciousAgent(agent_id="pred_none")
    assert agent.predict_next() is None


def test_predict_next_returns_prediction():
    from conscious_agent import Prediction
    agent = ConsciousAgent(agent_id="pred_test")
    world = SimpleWorld(n_states=5, seed=42)
    for _ in range(40):
        agent.step(world.step())
    result = agent.predict_next()
    assert isinstance(result, Prediction)
    assert isinstance(result.state_id, int)
    assert isinstance(result.state_label, str)
    assert 0.0 <= result.confidence <= 1.0
    top3 = result.top_k(3)
    assert len(top3) <= 3
    for t in top3:
        assert "state_id" in t
        assert "confidence" in t


def test_prediction_top_k():
    from conscious_agent import Prediction
    top_k = [
        {"state_id": 42, "state_label": "42", "confidence": 0.7},
        {"state_id": 7, "state_label": "7", "confidence": 0.3},
    ]
    p = Prediction(state_id=42, state_label="42", confidence=0.7, _top_k=top_k)
    assert p.state_id == 42
    assert p.confidence == 0.7
    assert len(p.top_k(1)) == 1
    assert len(p.top_k(10)) == 2


def test_from_config():
    agent = ConsciousAgent.from_config("config_test", {
        "agent": {
            "self_token": {"lock_threshold": 0.5, "lock_consecutive_required": 5},
            "meta_observation_interval": 10,
            "expression_temp": 0.5,
        },
    })
    assert agent.agent_id == "config_test"
    assert agent.experience.self_token.lock_threshold == 0.5
    assert agent.experience.self_token.lock_consecutive_required == 5
    assert agent.meta_observation_interval == 10
    assert agent.expression_temp == 0.5


def test_agent_rng_is_deterministic():
    """Regression test: agent._rng must actually drive decision-time
    randomness (decide()). Two agents seeded with identical rng streams,
    fed identical world states, must produce identical output sequences.
    Previously decide() silently used the global `random` module instead
    of the agent's configured rng, making ConsciousAgent(rng=...) a no-op.
    """
    import random as _random

    world_a = SimpleWorld(n_states=5, seed=42)
    world_b = SimpleWorld(n_states=5, seed=42)

    agent_a = ConsciousAgent(agent_id="rng_a")
    agent_b = ConsciousAgent(agent_id="rng_b")
    agent_a._rng = _random.Random(123)
    agent_b._rng = _random.Random(123)

    sequences_a = []
    sequences_b = []
    for _ in range(200):
        out_a = agent_a.step(world_a.step())
        out_b = agent_b.step(world_b.step())
        sequences_a.append(out_a.sequence_str)
        sequences_b.append(out_b.sequence_str)

    assert sequences_a == sequences_b
