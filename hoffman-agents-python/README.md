# conscious-agent

**An executable model of Donald Hoffman's conscious-agent formalism — in Python.**

Build self-referential agents that learn by *inhabiting* worlds — constructing internal models of both their environment and themselves.

> **For AI coding assistants**: A `SKILL.md` file lives in `.context/SKILL.md` with patterns for complex use cases (multi-agent networks, live data feeding, debugging). opencode and compatible tools load it automatically.

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld

world = CoinTossWorld(n_coins=4)
agent = ConsciousAgent(world=world, agent_id="my_agent")
outputs = agent.run(n_steps=1000)
print(f'"I" locked: {agent.is_i_locked}')
```

## Installation

```bash
pip install numpy scipy       # core dependencies
pip install conscious-agent   # once published
```

Or from source:
```bash
cd hoffman-agents-python
pip install -e .
```

## Quick Start

### Single agent in a coin-toss world

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld

world = CoinTossWorld(n_coins=3)
agent = ConsciousAgent(agent_id="coin_agent", world=world)

for _ in range(500):
    output = agent.step()
    if output.i_locked:
        print(f"I locked at step {output.step}")
        break
```

### Custom Markov world

```python
from conscious_agent import ConsciousAgent, WorldBuilder
import numpy as np

data = np.random.rand(500, 3)
world = (WorldBuilder()
    .add_feature("temp", normalization="minmax", n_bins=4)
    .add_feature("humidity", normalization="minmax", n_bins=4)
    .add_feature("pressure", normalization="minmax", n_bins=4)
    .build(data))

agent = ConsciousAgent(agent_id="weather_agent", world=world)
outputs = agent.run(n_steps=1000)
```

### Combine two agents

```python
from conscious_agent import combine

a = ConsciousAgent(agent_id="agent_a", world=world)
b = ConsciousAgent(agent_id="agent_b", world=world)
a.run(500)
b.run(500)

combined = combine(a, b)
print(f"Combined agent: {combined.agent_id}, level: {combined.cycle_level}")
```

### Multi-agent network

```python
from conscious_agent import AgentNetwork

network = AgentNetwork(n_agents=10, seed=42)
states = network.run(n_generations=100)
print(f"Avg prediction error: {network.avg_prediction_error():.3f}")
```

### Save and load

```python
from conscious_agent.io import save_agent, load_agent, clone_agent

path = save_agent(agent, "./souls")
loaded = load_agent(path)

cloned = clone_agent(agent, "experiment_clone")
```

## Public API

```python
# Core classes
from conscious_agent import ConsciousAgent, World, WorldBuilder
from conscious_agent import SimpleWorld, ExperienceSpace, Prediction

# World factories
from conscious_agent.worlds import CoinTossWorld, SelfWorld, Normalizer, FeatureSpec
from conscious_agent import build_world_from_dataframe

# IO
from conscious_agent.io import save_agent, load_agent, clone_agent, load_latest

# Multi-agent
from conscious_agent import AgentNetwork, Topology, InteractionCycle, combine

# Core components (for advanced use)
from conscious_agent import (
    TraceBuffer, TraceEvent, ExperienceTrie, TrieNode,
    MetaTrie, MetaStateSnapshot, SelfTokenState,
    ExperienceLexicon, LexiconEntry,
)

# Utilities
from conscious_agent import (
    strange_loop_score, compute_self_reference_score,
    population_reference_score, population_loop_score,
    first_depth_n_generation,
    prune, trace_distance, merge_similar_paths,
    invent_token, is_invented_token,
    SharedMeaningTracker,
)

# v2.1 — Predict next state
agent.predict_next()                     # → Prediction object
prediction.top_k(3)                      # top 3 alternatives with confidence

# v2.1 — Config-driven construction
ConsciousAgent.from_config("id", {"agent": {"self_token": {"lock_threshold": 0.3}}})

# v2.1 — Topology introspection
topology.get_connection_strength(0, 1)   # query connection weight
topology.maybe_add_connection(0, 5)       # add link probabilistically
topology.get_agent_observers(3)           # who observes agent 3?

# v2.0 — Agent mode control
agent.set_mode("frozen")        # 'learning', 'frozen', 'debug'
agent.thaw()                    # back to learning mode
agent.refreeze()                # back to frozen

# v2.0 — Memory & lifecycle
agent.clear_memory()            # reset trace buffer + counters, preserve trie
agent.inject_observation(world_state)  # push new data mid-run

# v2.0 — Metrics & introspection
agent.metrics                   # { prediction_error, i_locked, loop_depth, ... }
network.get_metrics()           # { agent_count, mean_prediction_error, i_lock_rate }
network.get_agent_metrics(id)   # individual agent's metrics
trie.get_stats()                # { node_count, max_depth, mean_visit_count, ... }
trie.export_nodes(3)            # all paths with visit_count >= 3
trie.get_dominant_paths(5)      # top 5 most-visited paths

# v2.0 — Batch stepping
network.step_all(world_state)   # step all agents with same world state
network.agent_list              # agents as an ordered list

# v2.0 — Action space
output.action_distribution      # { token: probability, ... }
agent.set_allowable_tokens({"I", "notice"})  # constrain output

# v2.0 — Composition
combine(a1, a2, a3)             # n-ary combination (3+ agents)
fuse(combined)                  # decompose back into constituents

# v2.0 — TraceBuffer
trace_buffer.resize(100)         # dynamic window resizing

# v2.0 — Trie compression
prune(trie, min_visits=5)        # remove nodes with < 5 visits
trace_distance(path_a, path_b)   # edit distance with transition cost
merge_similar_paths(trie, matrix, threshold=0.15)  # merge near-duplicate paths
```

## Self-Modelling

Agents contain four self-modelling mechanisms. They are mechanisms; whether any of them amounts to self-awareness is not something the code can establish.

| Mechanism | Type | What it does |
|-----------|------|-------------|
| **MetaTrie** | Built-in | A Markov chain over the agent's own coarse self-observations |
| **SelfTokenState ("I")** | Built-in | Locks when that chain has a stable attractor; unlocks when it dissolves |
| **strangeLoopScore** | Built-in | Counts self-reference in output tokens (mostly reflects time in the `core` output mode) |
| **SelfWorld** | Optional wrapper | Feeds the agent's internal metrics back into its perception |

### SelfWorld

`SelfWorld` is a world wrapper that lets the agent perceive its own internal state alongside external data. The agent's trie learns transitions over composite states of `(world + self)`.

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

Each step, the agent's WorldState contains both `'world'` and `'self'` sequences. The agent discovers patterns like "when my prediction error is high and the world shows pattern X, the next state tends to be Y."

→ What these mechanisms do and do not show: [docs/SELF_AWARENESS.md](docs/SELF_AWARENESS.md)

## How It Works

Every ConsciousAgent has an **experience space** — four interconnected structures:

1. **TraceBuffer** — short-term memory: the last N state transitions
2. **ExperienceTrie** — long-term world model: compressed prefix tree over observed state sequences
3. **MetaTrie** — self-model: a Markov chain over the agent's own coarse self-observations
4. **SelfTokenState ("I")** — identity: locks onto a stable attractor of the self-model

The agent cycles through **perception** (observe world → update trie) → **meta-observation** (observe self → update meta-trie) → **decision** (generate output tokens via the decision kernel D).

The "I" locks when the agent's self-observation chain has a clear, stable attractor: enough data, aperiodic, one state well above the uniform baseline, currently occupied. It unlocks with hysteresis when the attractor dissolves. It fires in worlds with structure and not in noise. `agent.ergodic_stats()` reports the criteria, and `agent.to_formal()` exports the learned kernels as a (X, G, P, D, A, N) tuple. The mathematics is in [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md).

Version 3 changed the math (see RELEASE_NOTES_v3.0.0.md). Pass `math_version="legacy"` to reproduce 2.x behaviour exactly, and `seed=1` for reproducible runs, identical across Node and Python.

## Experiments

The `examples/` directory holds experiments, each with controls. Highlights:

- **13 Bell test through the headset**: correlations produced by a conscious-agent network behind spacetime; best CHSH = 2 + 2·δ(Qᵏ) (`python examples/13_bell_through_the_headset/bell_through_the_headset.py`)
- **14 Spacetime in the headset**: an observer recovers dimension from experience; ⊗ adds dimensions and interaction binds them (`python examples/14_spacetime_in_the_headset/spacetime_in_the_headset.py`)
- **15 Time in the traces**: Hoffman's trace chains; per-observer clocks and an observer-dependent arrow of time (`python examples/15_time_in_the_traces/time_in_the_traces.py`)
- **16 Trace logic and decorated permutations**: Hoffman's map to the positive Grassmannian's combinatorics, checked against the paper; trace logic's structure; mass and speed proposals stress-tested (`python examples/16_trace_logic_and_decorated_permutations/trace_logic.py`)
- **17 Quantum agents**: kernels as quantum channels; combined agents reach exactly 2√2; Markov agents as the decohered limit (`python examples/17_quantum_agents/quantum_agents.py`)
- **18 Relativity at infinity**: as n → ∞ an agent's clock converges to Einstein's proper time; a light cone needs memory (`python examples/18_relativity_at_infinity/relativity_at_infinity.py`)
- **02 Quantum signature?**: why an earlier "quantum-like" signature was an artifact
- **05 Self-reference ablation** and **12 ergodic diagnostics**: the "I" lock tracks structure

## Documentation

[MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md) · [Q_AND_A.md](docs/Q_AND_A.md) (results, limitations, corrections) · [COMPONENT_DEFINITIONS.md](docs/COMPONENT_DEFINITIONS.md) · [CONSCIOUS_AGENTS_VISUAL_GUIDE.md](docs/CONSCIOUS_AGENTS_VISUAL_GUIDE.md) · [SELF_AWARENESS.md](docs/SELF_AWARENESS.md) · [GLOSSARY.md](docs/GLOSSARY.md) · [CONSCIOUS_AGENTS_THEORY.md](docs/CONSCIOUS_AGENTS_THEORY.md) (design document)

## Requirements

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10

## License

MIT
