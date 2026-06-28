# conscious-agent

**A computational implementation of Hoffman's Conscious Realism.**

Build self-referential agents that learn by *inhabiting* worlds — constructing internal models of both their environment and themselves.

> **For AI coding assistants**: A `SKILL.md` file lives in `.context/SKILL.md` with patterns for complex use cases (multi-agent networks, crystal projection, live data feeding, debugging). opencode and compatible tools load it automatically.

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
from conscious_agent import SimpleWorld, ExperienceSpace

# World factories
from conscious_agent.worlds import CoinTossWorld, build_world_from_dataframe

# IO
from conscious_agent.io import save_agent, load_agent, clone_agent, load_latest

# Multi-agent
from conscious_agent import AgentNetwork, combine

# Core components (for advanced use)
from conscious_agent import (
    TraceBuffer, TraceEvent, ExperienceTrie, MetaTrie,
    SelfTokenState, ExperienceLexicon, strange_loop_score,
)

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

# v2.0 — TraceBuffer
trace_buffer.resize(100)         # dynamic window resizing
```

## Self-Awareness

This library provides **four self-awareness mechanisms**, three built-in and one optional:

| Mechanism | Type | What it does |
|-----------|------|-------------|
| **MetaTrie** | Built-in (implicit) | Models the agent's own trace buffer patterns — a hidden self-model |
| **SelfTokenState ("I")** | Built-in (implicit) | Tracks identity stability; locks on meta-trie convergence |
| **strangeLoopScore** | Built-in (explicit) | Measures self-referential depth in output tokens |
| **SelfWorld** | Optional wrapper | Injects agent's internal metrics into its perception stream |

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

→ Full philosophical architecture: `docs/SELF_AWARENESS.md`

## How It Works

Every ConsciousAgent has an **experience space** — four interconnected structures:

1. **TraceBuffer** — short-term memory: the last N state transitions
2. **ExperienceTrie** — long-term world model: compressed prefix tree over observed state sequences
3. **MetaTrie** — self-model: a second trie over the agent's own trace buffer snapshots (thinking about thinking)
4. **SelfTokenState ("I")** — identity: the dominant meta-state that forms a stable attractor

The agent cycles through **perception** (observe world → update trie) → **meta-observation** (observe self → update meta-trie) → **decision** (generate output tokens via ergodic Markov chain).

When the meta-trie's stationary distribution converges on a single meta-state, the "I" locks — the agent has formed a stable identity.

## Requirements

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10

## License

MIT
