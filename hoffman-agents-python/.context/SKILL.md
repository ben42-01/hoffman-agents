# conscious-agent Python Lib — Agent Skill

## Overview

This is a minimal-dependency Python library for building self-referential agents (Conscious Agents). Agents learn by *inhabiting* Markov worlds — building compressed tries over state sequences and meta-tries over their own trace buffers.

## Architecture (Data Flow)

```
WorldState.step() → WorldState (sequences dict)
                       ↓
agent.step(ws) ───────→ perceive() → update TraceBuffer + ExperienceTrie
                              ↓ (every meta_observation_interval steps)
                          MetaTrie.observeSelf() → update SelfTokenState
                              ↓
                          decide() → output tokens (ergodic Markov chain)
```

## Key Conventions

1. **State IDs are hashes** of `WorldState.sequences`. Never pass raw ints. Always use `WorldState.from_sequence(agent_id, sequence)` or `WorldState(sequences={...})`.

2. **MemorySpace → ExperienceSpace**: The old `agent.memory_space` is now `agent.experience`. Migration:
   - `agent.memory_space.trie` → `agent.experience.trie`
   - `agent.memory_space.meta_trie` → `agent.experience.meta_trie`
   - `agent.memory_space.self_token` → `agent.experience.self_token`
   - `agent.memory_space.trace_buffer` → `agent.experience.trace_buffer`
   - `agent.memory_space.lexicon` → `agent.experience.lexicon`

3. **No numpy/scipy fallback**: Required for spectral decomposition (quantum analysis). Node.js port uses row-concentration proxy instead.

## Commands

```bash
uv sync --group dev         # install (add --group dev for pytest)
uv run pytest tests/ -v     # run tests
uv run python examples/...  # run example
uv build && uv publish      # publish to PyPI
```

## Common Patterns

### Create agent in a world
```python
agent = ConsciousAgent(agent_id="my_agent")
for _ in range(100):
    ws = WorldState.from_sequence("world", ["state_1"])
    agent.step(ws)
```

### Set custom lock threshold (ablation)
```python
st = SelfTokenState(lock_threshold=1.5)  # never locks
exp = ExperienceSpace(self_token=st)
agent = ConsciousAgent(agent_id="ablated", experience=exp)
```

### Combine two agents
```python
from conscious_agent import combine
combined = combine(agent_a, agent_b)
```

### Serialize
```python
from conscious_agent.io import save_agent, load_agent, clone_agent
save_agent(agent, "./souls")
loaded = load_agent("./souls/agent_...soul")
cloned = clone_agent(agent, "clone_id")
```

### Analyze meta-trie spectral gap
```python
P = extract_meta_matrix(agent)
gap = spectral_gap(P)  # 1.0 = classical, ~0.0 = quantum-like
```

## Creating a New Example

1. Create `examples/NN_name/` directory
2. Add a `main()` function with argument parser for config
3. Use SSE (Server-Sent Events) for real-time web dashboards
4. Test with: `uv run python examples/NN_name/script.py`

## File Map

| File | Purpose |
|------|---------|
| `src/conscious_agent/__init__.py` | Public API exports |
| `src/conscious_agent/agent/conscious_agent.py` | Main agent class |
| `src/conscious_agent/agent/perceptual_map.py` | `perceive()` — P function |
| `src/conscious_agent/agent/decision_map.py` | `decide()` — D function |
| `src/conscious_agent/core/meta_trie.py` | Self-model |
| `src/conscious_agent/core/self_token.py` | "I" attractor |
| `src/conscious_agent/combination/operator.py` | ⊗ combine operator |
| `src/conscious_agent/io/serialization.py` | Save/load/clone |
| `src/conscious_agent/world/world_builder.py` | World construction |
