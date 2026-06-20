# conscious-agent — Session Context

This is a self-contained Python library for building Conscious Agents
(a computational implementation of Hoffman's Conscious Realism).

## Directory Structure

```
hoffman-agents-python/
├── src/
│   └── conscious_agent/          # The importable package
│       ├── __init__.py           # Public API exports
│       ├── core/                 # Components 1-8
│       │   ├── trace_buffer.py
│       │   ├── experience_trie.py
│       │   ├── meta_trie.py
│       │   ├── self_token.py
│       │   ├── strange_loop.py
│       │   ├── trie_compression.py
│       │   └── experience_lexicon.py
│       │   └── token_inventor.py
│       ├── agent/
│       │   ├── conscious_agent.py  # Main class
│       │   ├── world_state.py       # WorldState hash-based state ID
│       │   ├── experience_space.py  # MemorySpace dataclass
│       │   ├── perceptual_map.py    # perceive() — update trie from world
│       │   ├── decision_map.py      # decide() — ergodic output generation
│       │   └── simple_world.py      # Test Markov world
│       ├── io/
│       │   └── serialization.py     # Component 9: save/load/clone
│       ├── combination/
│       │   └── operator.py          # ⊗ operator (combine two agents)
│       ├── network/
│       │   └── agent_network.py     # Multi-agent network + topology
│       ├── world/
│       │   └── world_builder.py     # WorldBuilder, World, CoinTossWorld
│       └── meaning/
│           └── shared_meaning.py    # SharedMeaningTracker
├── tests/
│   └── test_core.py               # 15 tests covering all modules
├── examples/
│   ├── 01_fitness_beats_truth/
│   ├── 02_quantum_signature/
│   ├── 03_weather_benchmark/
│   └── 04_stop_lights/            # Web UI dashboard
├── pyproject.toml
├── SETUP.md
├── README.md
└── .context/
    └── SESSION_CONTEXT.md         # This file
```

## Quick Start (after setup)

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld

world = CoinTossWorld(n_coins=4)
agent = ConsciousAgent(agent_id="my_agent", world=world)
outputs = agent.run(n_steps=1000)
print(f'"I" locked: {agent.is_i_locked}')
```

## Key Design Decisions

- **State IDs are hashes** of `WorldState.sequences` — not raw integers. Never
  pass raw ints to `agent.step()`. Always use `WorldState.from_sequence(...)`.
- **ExperienceSpace** bundles the 4 core structures:
  `TraceBuffer` + `ExperienceTrie` + `MetaTrie` + `SelfTokenState`
- **Perception** updates the trie; **meta-observation** (every 20 steps) updates
  the meta-trie; **decision** generates output via ergodic Markov chain.
- **Minimal deps**: only numpy + scipy. No pandas, no pytorch, no sklearn.

## Commands

```bash
# Install (editable dev mode)
uv sync --group dev        # uv (recommended)
pip install -e .           # pip

# Test
uv run pytest tests/ -v
python -m pytest tests/ -v

# Run examples
uv run python examples/04_stop_lights/stop_lights.py

# Publish to PyPI
uv build && uv publish
python -m build && twine upload dist/*
```

## Architecture (Data Flow)

```
WorldState.step() → WorldState (sequences dict)
                       ↓
agent.step(ws) ───────→ perceive() → update TraceBuffer + ExperienceTrie
                              ↓ (every 20 steps)
                          MetaTrie.observe_self() → update SelfTokenState
                              ↓
                          decide() → output tokens (ergodic Markov chain)
```

## Important Files to Know

| File | Purpose |
|------|---------|
| `src/conscious_agent/agent/conscious_agent.py` | Main class — entry point for all agent operations |
| `src/conscious_agent/agent/perceptual_map.py` | `perceive()` — the P function in Hoffman's 6-tuple |
| `src/conscious_agent/agent/decision_map.py` | `decide()` — the D function, ergodic output |
| `src/conscious_agent/core/meta_trie.py` | Self-model — trie over trace buffer snapshots |
| `src/conscious_agent/core/self_token.py` | "I" attractor — identity formation |
| `src/conscious_agent/combination/operator.py` | ⊗ operator — combines two agents |
| `src/conscious_agent/io/serialization.py` | Component 9 — portable agent format |
| `src/conscious_agent/world/world_builder.py` | World construction from raw data |
