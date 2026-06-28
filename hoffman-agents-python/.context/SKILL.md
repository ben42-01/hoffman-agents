# conscious-agent Python Lib — Agent Skill

## Overview

This is a minimal-dependency Python library for building self-referential agents (Conscious Agents). Agents learn by *inhabiting* Markov worlds — building compressed tries over state sequences and meta-tries over their own trace buffers.

## Self-Awareness (v2.0 — Critical for Agent Understanding)

This library has **FOUR self-awareness mechanisms**:

| # | Mechanism | Module | What it does |
|---|-----------|--------|-------------|
| 1 | **MetaTrie** | `core/meta_trie.py` | Implicit self-model — models agent's own cognitive trace patterns |
| 2 | **SelfTokenState** | `core/self_token.py` | Identity attractor — "I" locks on meta-trie convergence |
| 3 | **strangeLoopScore** | `core/strange_loop.py` | Self-reference depth in agent's output tokens |
| 4 | **SelfWorld** (v2.0) | `world/self_world.py` | **Explicit** — injects agent's internal state into its perception stream |

**How they relate**: MetaTrie is the agent building a hidden model of itself. SelfTokenState is the identity that emerges from that model. strangeLoopScore is the observable signature in output. SelfWorld is the agent perceiving its own metrics as part of its world.

→ Start with **SelfWorld** if you want the agent to perceive its own state.
→ Start with **SelfTokenState** if you want identity/locking behavior.
→ See `docs/SELF_AWARENESS.md` for the full philosophical architecture.

## Architecture (Data Flow)

```
WorldState.step() → WorldState (sequences dict)
                       ↓
agent.step(ws) ───────→ perceive() → update TraceBuffer + ExperienceTrie
                              ↓ (every meta_observation_interval steps)
                          MetaTrie.observe_self() → update SelfTokenState
                              ↓
                          decide() → output tokens (ergodic Markov chain)
```

With **SelfWorld**:
```
External world → SelfWorld (injects agent metrics into W)
                     ↓
                WorldState (contains 'world' + 'self' sequences)
                     ↓
                agent.step(ws)
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

### Frozen mode (v2.0)
```python
agent.set_mode("frozen")    # deterministic projection, no learning
agent.thaw()                # back to learning
```

### Metrics (v2.0)
```python
agent.metrics               # { prediction_error, i_locked, loop_depth, ... }
network.get_metrics()       # aggregate across agents
```

### Action distribution (v2.0)
```python
output.action_distribution  # { token: probability, ... }
agent.set_allowable_tokens({"I", "notice"})  # constrain output
```

### Self-aware agent with SelfWorld (v2.0)
```python
from conscious_agent.worlds import SelfWorld

inner = SimpleWorld(n_states=10)
agent = ConsciousAgent(
    agent_id="self_aware",
    world=SelfWorld(inner, lambda a: {
        "sp": a.experience.self_token.stationary_prob,
        "pe": a.mean_prediction_error,
    }),
)
agent.run(n_steps=1000)
```

### Set custom lock threshold (ablation)
```python
st = SelfTokenState(lock_threshold=1.5)  # never locks
exp = ExperienceSpace(self_token=st)
agent = ConsciousAgent(agent_id="ablated", experience=exp)
```

### Combine agents (v2.0 — n-ary)
```python
from conscious_agent import combine
combined = combine(a1, a2, a3)  # 3+ agents
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

### Multi-agent network
```python
network = AgentNetwork(n_agents=10, seed=42)
network.run(n_generations=100)           # step all agents through topology
m = network.get_metrics()                # { agent_count, mean_prediction_error, i_lock_rate, ... }
m = network.get_agent_metrics("CA_000")  # individual agent snapshot
results = network.step_all(world_state)  # step all with same world
for a in network.agent_list:
    print(a.agent_id, a.is_i_locked)
```

### World from real data using WorldBuilder
```python
from conscious_agent import WorldBuilder

builder = WorldBuilder()
builder.add_feature("temperature", normalization="minmax", n_bins=5)
builder.add_feature("humidity", normalization="minmax", n_bins=4)
builder.add_feature("pressure", normalization="tanh", n_bins=3)
import numpy as np
data = np.random.rand(500, 3)
world = builder.build(data)
# world.n_states, world.transition_matrix, world.state_labels
```

### Lifecycle: train → freeze → save → thaw → retrain
```python
agent.run(n_steps=10000)
if agent.is_i_locked:
    agent.set_mode("frozen")
    save_agent(agent, "./snapshots")
    # later...
    loaded = load_agent("./snapshots/agent_....soul")
    loaded.thaw()                      # back to learning mode
    loaded.run(n_steps=5000)           # train more with new data
    loaded.set_mode("frozen")          # re-freeze
```

### Crystal projection: specialists → combinator
```python
from conscious_agent import combine

magnet = ConsciousAgent(agent_id="magnet", world=magnet_world)
magnet.run(n_steps=10000)
magnet.set_mode("frozen")              # crystal — stable identity

cern = ConsciousAgent(agent_id="cern", world=cern_world)
cern.run(n_steps=10000)
cern.set_mode("frozen")

brain = combine(magnet, cern)          # brain perceives both specialists
brain.run(n_steps=5000)                # combinator forms higher-order identity
print("Brain I-locked:", brain.is_i_locked, "level:", brain.cycle_level)
```

### Debugging: when I-lock doesn't happen
```python
sp = agent.experience.self_token.stationary_prob
meta_size = agent.experience.meta_trie.registry_size
stats = agent.experience.trie.get_stats()
print(sp, meta_size, stats)
# Low sp (<0.2) + small meta_size (<10) → world too large/random
# Solution: reduce n_states, increase meta_observation_interval
# High sp (>0.4) but never locks → lock_consecutive_required too high
```

### Live data / incremental feeding
```python
while True:
    new_data = get_next_event()                   # your data source
    state_id = world.state_from_new_data(new_data)  # uses stored normalization
    ws = WorldState.from_sequence("world", [str(state_id)])
    output = agent.step(ws)                       # incremental — no reset
    print(output.action_distribution)             # confidence over tokens
    if output.prediction_error > 0.8:
        print("Anomaly detected — agent is surprised")
```

### Clear memory for training pipeline correctness
```python
agent.clear_memory()  # resets trace buffer + step count, preserves trie/lexicon
```

### Action distribution for confidence-based decisions
```python
output = agent.step(ws)
dist = output.action_distribution  # { token: probability, ... }
best = max(dist.items(), key=lambda x: x[1]) if dist else (None, 0)
if best[1] > 0.5:
    print(f"High confidence: {best[0]} ({best[1]*100:.0f}%)")
else:
    print("Low confidence — defer to human")
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
| `src/conscious_agent/core/meta_trie.py` | Self-model (implicit) |
| `src/conscious_agent/core/self_token.py` | "I" attractor |
| `src/conscious_agent/core/strange_loop.py` | Self-reference scoring |
| `src/conscious_agent/combination/operator.py` | ⊗ combine operator |
| `src/conscious_agent/io/serialization.py` | Save/load/clone |
| `src/conscious_agent/world/world_builder.py` | World construction |
| `src/conscious_agent/world/self_world.py` | SelfWorld (v2.0) — explicit self-perception |
