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


def test_serialization(tmp_path):
    agent = ConsciousAgent(agent_id="save_test")
    path = str(tmp_path / "test.soul")
    from conscious_agent.io import serialize, deserialize
    serialize(agent, path)
    loaded = deserialize(path)
    assert loaded.agent_id == "save_test"


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
