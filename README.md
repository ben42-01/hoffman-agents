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

## Installation  Python

```bash
pip install numpy scipy       # core dependencies
pip install conscious-agent   # once published
```

Or from source:
```bash
cd hoffman-agents-python
pip install -e .
```

Or install directly from GitHub without publishing to PyPI:
```bash
pip install git+https://github.com/ben42-01/hoffman-agents.git@main#subdirectory=hoffman-agents-python
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




## Installation

```bash
npm install conscious-agent   # once published
```

or directly From Git 

```bash
npm install github:ben42-01/hoffman-agents
```

Or from source:
```bash
cd hoffman-agents-node
npm link                     # or copy src/ into your project
```

## Quick Start

```javascript
const { ConsciousAgent } = require('conscious-agent');
const { CoinTossWorld } = require('conscious-agent/worlds');

const world = new CoinTossWorld(4);
const agent = new ConsciousAgent({ agentId: 'my_agent', world });
const outputs = agent.run(1000);
console.log(`"I" locked: ${agent.isILocked}`);
```

### Single agent in a coin-toss world

```javascript
const { ConsciousAgent } = require('conscious-agent');
const { CoinTossWorld } = require('conscious-agent/worlds');

const world = new CoinTossWorld(3);
const agent = new ConsciousAgent({ agentId: 'coin_agent', world });

for (let i = 0; i < 500; i++) {
  const output = agent.step();
  if (output.iLocked) {
    console.log(`I locked at step ${output.step}`);
    break;
  }
}
```

### Custom Markov world

```javascript
const { ConsciousAgent, WorldBuilder } = require('conscious-agent');

const data = Array.from({ length: 500 }, () => [Math.random(), Math.random(), Math.random()]);
const world = new WorldBuilder()
  .addFeature('temp', 'minmax', 4)
  .addFeature('humidity', 'minmax', 4)
  .addFeature('pressure', 'minmax', 4)
  .build(data);

const agent = new ConsciousAgent({ agentId: 'weather_agent', world });
const outputs = agent.run(1000);
```

### Combine two agents

```javascript
const { combine } = require('conscious-agent');

const a = new ConsciousAgent({ agentId: 'agent_a', world });
const b = new ConsciousAgent({ agentId: 'agent_b', world });
a.run(500);
b.run(500);

const combined = combine(a, b);
console.log(`Combined agent: ${combined.agentId}, level: ${combined.cycleLevel}`);
```

### Multi-agent network

```javascript
const { AgentNetwork } = require('conscious-agent');

const network = new AgentNetwork({ nAgents: 10, seed: 42 });
const states = network.run(100);
console.log(`Avg prediction error: ${network.avgPredictionError().toFixed(3)}`);
```

### Save and load

```javascript
const { saveAgent, loadAgent, cloneAgent } = require('conscious-agent/io');

const path = saveAgent(agent, './souls');
const loaded = loadAgent(path);

const cloned = cloneAgent(agent, 'experiment_clone');
```

## Public API

```javascript
// Core classes
const { ConsciousAgent, World, WorldBuilder } = require('conscious-agent');
const { SimpleWorld, ExperienceSpace } = require('conscious-agent');

// World factories
const { CoinTossWorld } = require('conscious-agent/worlds');

// IO
const { saveAgent, loadAgent, cloneAgent, loadLatest } = require('conscious-agent/io');

// Multi-agent
const { AgentNetwork, combine } = require('conscious-agent');

// Core components
const { TraceBuffer, ExperienceTrie, MetaTrie, SelfTokenState, ExperienceLexicon } = require('conscious-agent');

// v2.0 — Agent mode
agent.setMode('frozen');       // deterministic projection, no learning
agent.thaw();                  // back to learning
agent.refreeze();              // back to frozen

// v2.0 — Memory & lifecycle
agent.clearMemory();           // reset short-term, preserve long-term
agent.injectObservation(worldState); // push data mid-run

// v2.0 — Metrics & introspection
agent.metrics;                 // { predictionError, iLocked, loopDepth, ... }
network.getMetrics();          // aggregate across all agents
network.getAgentMetrics(id);   // per-agent snapshot
trie.getStats();               // { nodeCount, maxDepth, meanVisitCount, ... }
trie.exportNodes(3);           // export paths with visitCount >= 3
trie.getDominantPaths(5);      // top 5 most-visited

// v2.0 — Batch stepping
network.stepAll(worldState);   // same world for all agents
network.agentList;             // agents as array

// v2.0 — Action space
output.actionDistribution;     // { token: probability, ... }
agent.setAllowableTokens(['I', 'notice']);  // constrain output
new ConsciousAgent({ allowableTokens: [...] }); // at construction

// v2.0 — Composition
combine(a, b, c);              // n-ary (3+ agents)

// v2.0 — TraceBuffer
traceBuffer.resize(100);
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

# v2.0 — Agent mode
agent.set_mode("frozen")        # deterministic projection, no learning
agent.thaw()                    # back to learning
agent.refreeze()                # back to frozen

# v2.0 — Memory & lifecycle
agent.clear_memory()            # reset short-term, preserve long-term
agent.inject_observation(world_state)  # push data mid-run

# v2.0 — Metrics & introspection
agent.metrics                   # { prediction_error, i_locked, loop_depth, ... }
network.get_metrics()           # aggregate across all agents
network.get_agent_metrics(id)   # per-agent snapshot
trie.get_stats()                # { node_count, max_depth, mean_visit_count, ... }
trie.export_nodes(3)            # export paths with visit_count >= 3
trie.get_dominant_paths(5)      # top 5 most-visited

# v2.0 — Batch stepping
network.step_all(world_state)   # same world for all agents
network.agent_list              # agents as list

# v2.0 — Action space
output.action_distribution      # { token: probability, ... }
agent.set_allowable_tokens({"I", "notice"})  # constrain output

# v2.0 — Composition
combine(a1, a2, a3)             # n-ary (3+ agents)

# v2.0 — TraceBuffer
trace_buffer.resize(100)
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

## Documentation

This repository includes a single shared docs folder at `docs/`.

The docs cover:
- [CONSCIOUS_AGENTS_THEORY.md](docs/CONSCIOUS_AGENTS_THEORY.md) — the underlying theory
- [COMPONENT_DEFINITIONS.md](docs/COMPONENT_DEFINITIONS.md) — component descriptions and data flow
- [CA_RUNTIME_API.md](docs/CA_RUNTIME_API.md) — runtime API and usage patterns
- [CONSCIOUS_AGENTS_VISUAL_GUIDE.md](docs/CONSCIOUS_AGENTS_VISUAL_GUIDE.md) — visual explanation
- [GLOSSARY.md](docs/GLOSSARY.md) — key terms
- [Q_AND_A.md](docs/Q_AND_A.md) — frequently asked questions

## License

MIT