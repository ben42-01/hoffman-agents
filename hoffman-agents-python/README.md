# conscious-agent

**A computational implementation of Hoffman's Conscious Realism.**

Build self-referential agents that learn by *inhabiting* worlds — constructing internal models of both their environment and themselves.

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

# v2.0 — New APIs
# Agent mode control: agent.set_mode("frozen"), agent.thaw(), agent.refreeze()
# Memory control: agent.clear_memory()
# Metrics: agent.metrics, network.get_metrics(), network.get_agent_metrics(id)
# Batch stepping: network.step_all(world_state), network.agent_list
# Action distribution: output.action_distribution
# Token constraints: agent.set_allowable_tokens({...})
# Incremental injection: agent.inject_observation(world_state)
# N-ary combine: combine(a1, a2, a3)
# Trie introspection: trie.get_stats(), trie.export_nodes(3), trie.get_dominant_paths(5)
# Trace buffer: trace_buffer.resize(new_size)
```

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
