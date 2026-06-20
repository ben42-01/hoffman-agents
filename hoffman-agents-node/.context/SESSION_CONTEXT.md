# conscious-agent — Session Context (Node.js)

Self-contained Node.js library for building Conscious Agents (Hoffman's
Conscious Realism). Zero external dependencies.

## Directory Structure

```
hoffman-agents-node/
├── src/
│   ├── index.js                    # Public API exports
│   ├── core/
│   │   ├── trace-buffer.js         # TraceEvent, TraceBuffer
│   │   ├── experience-trie.js      # ExperienceTrie, TrieNode
│   │   ├── meta-trie.js            # MetaTrie, MetaStateSnapshot
│   │   ├── self-token.js           # SelfTokenState ("I" attractor)
│   │   ├── strange-loop.js         # Self-reference scoring
│   │   ├── trie-compression.js     # prune, traceDistance, mergeSimilarPaths
│   │   ├── experience-lexicon.js   # ExperienceLexicon, LexiconEntry
│   │   └── token-inventor.js       # inventToken, isInventedToken
│   ├── agent/
│   │   ├── conscious-agent.js      # ConsciousAgent, StepOutput
│   │   ├── world-state.js          # WorldState, sequenceToStateId
│   │   ├── experience-space.js     # ExperienceSpace dataclass
│   │   ├── perceptual-map.js       # perceive() — trie update from world
│   │   ├── decision-map.js         # decide() — ergodic output generation
│   │   └── simple-world.js         # Test Markov world
│   ├── io/
│   │   ├── index.js                # Re-exports
│   │   └── serialization.js        # Component 9: save/load/clone
│   ├── combination/
│   │   ├── index.js                # Re-exports
│   │   └── operator.js             # ⊗ operator (combine two agents)
│   ├── network/
│   │   ├── index.js                # Re-exports
│   │   └── agent-network.js        # AgentNetwork, Topology, InteractionCycle
│   ├── world/
│   │   ├── index.js                # Re-exports
│   │   └── world-builder.js        # WorldBuilder, World, CoinTossWorld
│   └── meaning/
│       ├── index.js                # Re-exports
│       └── shared-meaning.js       # SharedMeaningTracker
├── test/
│   └── core.test.js                # 15 tests covering all modules
├── examples/
│   ├── 01_fitness_beats_truth/
│   ├── 02_quantum_signature/
│   ├── 03_weather_benchmark/
│   └── 04_stop_lights/             # Web UI dashboard
├── docs/
│   ├── CONSCIOUS_AGENTS_THEORY.md
│   ├── COMPONENT_DEFINITIONS.md
│   └── CA_RUNTIME_API.md
├── package.json
├── README.md
└── .context/
    └── SESSION_CONTEXT.md          # This file
```

## Quick Start

```javascript
const { ConsciousAgent } = require('./src/index');
const { CoinTossWorld } = require('./src/world');

const world = new CoinTossWorld(4);
const agent = new ConsciousAgent({ agentId: 'my_agent', world });
const outputs = agent.run(1000);
console.log(`"I" locked: ${agent.isILocked}`);
```

## Key Design Decisions

- **Zero dependencies** — only Node.js stdlib (`crypto`, `fs`, `path`, `http`)
- **State IDs are hashes** of WorldState sequences — never pass raw ints.
  Always use `WorldState.fromSequence(agentId, sequence)`.
- **Float64Array** is used instead of regular arrays for numerical ops
  (stationary distribution, matrix power iteration).
- **Same architecture as Python lib** — same API, same data flow.

## Commands

```bash
# No installation needed — just Node.js 18+
node --test test/core.test.js        # Run tests
node examples/04_stop_lights/stop_lights.js  # Run an example

# For npm publishing
npm pack                              # Create tarball
npm publish                           # Publish to npm
```

## Architecture (Data Flow)

```
WorldState.step() → WorldState (sequences object)
                       ↓
agent.step(ws) ───────→ perceive() → update TraceBuffer + ExperienceTrie
                              ↓ (every 20 steps)
                          MetaTrie.observeSelf() → update SelfTokenState
                              ↓
                          decide() → output tokens (ergodic Markov chain)
```

## Important Files

| File | Purpose |
|------|---------|
| `src/agent/conscious-agent.js` | Main class — all agent operations |
| `src/agent/perceptual-map.js` | `perceive()` — the P function |
| `src/agent/decision-map.js` | `decide()` — the D function |
| `src/core/meta-trie.js` | Self-model over trace buffer |
| `src/core/self-token.js` | "I" attractor — identity formation |
| `src/combination/operator.js` | ⊗ combine operator |
| `src/io/serialization.js` | Component 9 — portable format |
| `src/world/world-builder.js` | World construction from data |
| `src/index.js` | Public API exports |

## Public API

```javascript
const { ConsciousAgent, World, WorldBuilder } = require('./src/index');
const { CoinTossWorld } = require('./src/world');
const { saveAgent, loadAgent, cloneAgent } = require('./src/io');
const { combine, AgentNetwork } = require('./src/index');
```

## Examples

```bash
node examples/01_fitness_beats_truth/fitness_beats_truth.js
node examples/02_quantum_signature/quantum_signature.js
node examples/03_weather_benchmark/weather_benchmark.js
node examples/04_stop_lights/stop_lights.js     # http://localhost:8765
```

## Note

This is a Node.js port of the Python `conscious-agent` library. The
Python version has additional features (numpy/scipy for matrix ops).
The Node.js version implements the same algorithms with stdlib-only
Float64Array operations.
