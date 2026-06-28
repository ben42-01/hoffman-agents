# conscious-agent

**A computational implementation of Hoffman's Conscious Realism — in Node.js.**

Build self-referential agents that learn by *inhabiting* worlds. Zero external dependencies.

> **For AI coding assistants**: A `SKILL.md` file lives in `.context/SKILL.md` with patterns for complex use cases (multi-agent networks, crystal projection, live data feeding, debugging). opencode and compatible tools load it automatically.

```javascript
const { ConsciousAgent } = require('conscious-agent');
const { CoinTossWorld } = require('conscious-agent/worlds');

const world = new CoinTossWorld(4);
const agent = new ConsciousAgent({ agentId: 'my_agent', world });
const outputs = agent.run(1000);
console.log(`"I" locked: ${agent.isILocked}`);
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

// v2.0 — Agent mode control
const { setMode } = agent;   // 'learning', 'frozen', 'debug'
agent.setMode('frozen');     // deterministic projection, no trie/meta updates
agent.thaw();                // back to learning mode
agent.refreeze();            // back to frozen

// v2.0 — Memory & lifecycle
agent.clearMemory();         // reset trace buffer + counters, preserve trie/lexicon
agent.injectObservation(worldState);  // push new data mid-run without reset

// v2.0 — Metrics & introspection
agent.metrics;               // { predictionError, iLocked, loopDepth, outputTokens }
network.getMetrics();        // { agentCount, meanPredictionError, iLockRate, ... }
network.getAgentMetrics(id); // individual agent's metrics snapshot
trie.getStats();             // { nodeCount, maxDepth, meanVisitCount, depthDistribution }
trie.exportNodes(3);         // all paths with visitCount >= 3
trie.getDominantPaths(5);    // top 5 most-visited paths

// v2.0 — Batch stepping
network.stepAll(worldState); // step all agents with same world state
network.agentList;           // agents as an ordered array

// v2.0 — Action space
output.actionDistribution;   // { token: probability, ... } — full distribution
new ConsciousAgent({ allowableTokens: ['I', 'notice'] });  // constrain output
agent.setAllowableTokens(['I', 'notice', 'familiar']);

// v2.0 — Composition
combine(a, b, c);            // n-ary combination (3+ agents)

// v2.0 — TraceBuffer
traceBuffer.resize(100);     // dynamic window resizing
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

```javascript
const inner = new SimpleWorld({ nStates: 10 });
const agent = new ConsciousAgent({
  agentId: 'self_aware',
  world: new SelfWorld(inner, (self) => ({
    sp: self.experience.selfToken.stationaryProb,
    pe: self.meanPredictionError,
  })),
});
agent.run(1000);
```

Each step, the agent's WorldState contains both `'world'` and `'self'` sequences. The agent discovers patterns like "when my prediction error is high and the world shows pattern X, the next state tends to be Y."

→ Full philosophical architecture: `docs/SELF_AWARENESS.md`

## How It Works

Every ConsciousAgent has an **experience space** — four interconnected structures:

1. **TraceBuffer** — short-term memory: the last N state transitions
2. **ExperienceTrie** — long-term world model: compressed prefix tree over observed state sequences
3. **MetaTrie** — self-model: a second trie over the agent's own trace buffer snapshots (thinking about thinking)
4. **SelfTokenState ("I")** — identity: the dominant meta-state that forms a stable attractor

The agent cycles through **perception** → **meta-observation** → **decision** (generate output tokens via ergodic Markov chain).

When the meta-trie's stationary distribution converges on a single meta-state, the "I" locks — the agent has formed a stable identity.

## Requirements

- Node.js >= 18
- No external dependencies

## License

MIT
