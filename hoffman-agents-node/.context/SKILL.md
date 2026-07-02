# conscious-agent Node.js Lib — Agent Skill

## Overview

Zero-dependency Node.js library for building self-referential agents (Hoffman's Conscious Realism). Same architecture as the Python lib but implemented entirely with stdlib (`crypto`, `fs`, `http`, `Float64Array`).

## Self-Awareness (v2.0 — Critical for Agent Understanding)

This library has **FOUR self-awareness mechanisms**. If you're exploring self-awareness, check these first:

| # | Mechanism | File | What it does |
|---|-----------|------|-------------|
| 1 | **MetaTrie** | `src/core/meta-trie.js` | Implicit self-model — models agent's own cognitive trace patterns |
| 2 | **SelfTokenState** | `src/core/self-token.js` | Identity attractor — "I" locks on meta-trie convergence |
| 3 | **strangeLoopScore** | `src/core/strange-loop.js` | Self-reference depth in agent's output tokens |
| 4 | **SelfWorld** (v2.0) | `src/world/self-world.js` | **Explicit** — injects agent's internal state into its perception stream |

**How they relate**: MetaTrie is the agent building a hidden model of itself. SelfTokenState is the identity that emerges from that model. strangeLoopScore is the observable signature in output. SelfWorld is the agent perceiving its own metrics as part of its world.

→ Start with **SelfWorld** if you want the agent to perceive its own state.
→ Start with **SelfTokenState** if you want identity/locking behavior.
→ See `docs/SELF_AWARENESS.md` for the full philosophical architecture.

## Architecture (Data Flow)

```
WorldState.step() → WorldState (sequences object)
                       ↓
agent.step(ws) ───────→ perceive() → update TraceBuffer + ExperienceTrie
                              ↓ (every meta_observation_interval steps)
                          MetaTrie.observeSelf() → update SelfTokenState
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

1. **State IDs are hashes** of `WorldState.sequences`. Always use `WorldState.fromSequence(agentId, sequence)` or `new WorldState({agentId: sequence})`.

2. **CamelCase**: JavaScript conventions — `agent.getOutput()`, `metaTrie.observeSelf()`, `traceBuffer.predictionErrorMean()`. See Python's snake_case equivalents.

3. **Spectral gap**: Real eigenvalue-based via power iteration + Hotelling deflation. Uses Float64Array.
   ```
   B = P - 1·πᵀ      (deflate eigenvalue 1)
   λ₂ = power_iteration(B)
   gap = 1 - |λ₂|
   ```
   Range: 0 (deterministic cycle) to 1 (maximal mixing).

4. **Float64Array** for matrix operations (stationary distribution, power iteration).

## Commands

```bash
node --test test/*.test.js      # run all tests
node --test --test-name-pattern="AgentNetwork"  # run specific tests
node examples/04_stop_lights/stop_lights.js  # run an example (has web UI)
node examples/01_*/fitness_beats_truth.js    # run example 1
node -e "console.log(require('./src/index'))" # verify exports
npm pack                        # create .tgz for local install
npm publish                     # publish to npm
```

## Common Patterns

### Create agent
```javascript
const agent = new ConsciousAgent({ agentId: 'my_agent' });
for (let i = 0; i < 100; i++) {
  const ws = WorldState.fromSequence('world', ['state_1']);
  agent.step(ws);
}
```

### Frozen mode (v2.0)
```javascript
agent.setMode('frozen');  // deterministic projection, no learning
agent.thaw();             // back to learning
```

### Metrics (v2.0)
```javascript
agent.metrics;                  // { predictionError, iLocked, loopDepth, ... }
network.getMetrics();           // aggregate across agents
```

### Action distribution (v2.0)
```javascript
output.actionDistribution;      // { token: probability, ... }
```

### Self-aware agent with SelfWorld (v2.0)
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

### Custom lock threshold (ablation)
```javascript
const st = new SelfTokenState({ lockThreshold: 1.5 });
const exp = new ExperienceSpace({ selfToken: st });
const agent = new ConsciousAgent({ agentId: 'ablated', experience: exp });
```

### N-ary Combine (v2.0)
```javascript
const combined = combine(agentA, agentB, agentC);  // 3+ agents
```

### Fuse (decompose combined agent)
```javascript
const { fuse } = require('conscious-agent');
const [a, b] = fuse(combined);  // splits L1 agent back into L0 constituents
// Fused agents retain: shared experience trie, split meta-trie, fresh self-token
```

### Save/load
```javascript
const { saveAgent, loadAgent, cloneAgent } = require('conscious-agent/io');
saveAgent(agent, './souls');
const loaded = loadAgent('./souls/agent_...soul');
const cloned = cloneAgent(agent, 'clone_id');
```

### SSE Web Dashboard (pattern)
```javascript
const http = require('http');
http.createServer((req, res) => {
  if (req.url === '/stream') {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    setInterval(() => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    }, 500);
  } else {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
  }
}).listen(8765);
```

### Multi-agent network
```javascript
const net = new AgentNetwork({ nAgents: 10, seed: 42 });
net.run(100);                        // step all agents through topology
const m = net.getMetrics();          // { agentCount, meanPredictionError, iLockRate, ... }
net.getAgentMetrics('CA_000');       // individual agent snapshot
net.stepAll(someWorldState);         // step all with same world (bypasses topology)
net.agentList.forEach(a => console.log(a.agentId, a.isILocked));
```

### World from real data using WorldBuilder
```javascript
const builder = new WorldBuilder();
builder.addFeature('temperature', 'minmax', 5);   // normalize to 0-1, 5 bins
builder.addFeature('humidity', 'minmax', 4);       // 4 bins
builder.addFeature('pressure', 'tanh', 3);          // tanh for outliers
const data = [[23.5, 65, 1013], [24.1, 63, 1011], ...];
const world = builder.build(data);
// world.nStates, world.transitionMatrix, world.stateLabels
```

### Lifecycle: train → freeze → save → thaw → retrain
```javascript
agent.run(10000);
if (agent.isILocked) {
  agent.setMode('frozen');
  saveAgent(agent, './snapshots');
  // later...
  const loaded = loadAgent('./snapshots/agent_....soul');
  loaded.thaw();                    // back to learning mode
  loaded.run(5000);                 // train more with new data
  loaded.setMode('frozen');         // re-freeze
}
```

### Crystal projection: specialists → combinator
```javascript
// Train specialists, freeze them, then combine
const magnetAgent = new ConsciousAgent({ agentId: 'magnet', world: magnetWorld });
magnetAgent.run(10000);
magnetAgent.setMode('frozen');       // crystal — stable identity, no drift

const cernAgent = new ConsciousAgent({ agentId: 'cern', world: cernWorld });
cernAgent.run(10000);
cernAgent.setMode('frozen');

const brain = combine(magnetAgent, cernAgent);
// brain perceives BOTH specialists' outputs as its world
brain.run(5000);                     // combinator forms higher-order identity
console.log('Brain I-locked:', brain.isILocked, 'level:', brain.cycleLevel);
```

### Debugging: when I-lock doesn't happen
```javascript
const sp = agent.experience.selfToken.stationaryProb;
const metaSize = agent.experience.metaTrie.registrySize;
const trieStats = agent.experience.trie.getStats();
console.log({ sp, metaSize, trieStats });
// Low sp (<0.2) + small metaSize (<10) → world too large or too random
// Solution: reduce nStates, increase metaObservationInterval, lower lockThreshold
// High sp (>0.4) but never locks → lockConsecutiveRequired too high
// Solution: reduce lockConsecutiveRequired
```

### Live data / incremental feeding
```javascript
// Agent already trained on historical data, now receiving live stream
while (true) {
  const newData = await getNextEvent();            // your data source
  const stateId = world.stateFromNewData(newData); // uses stored normalization
  const ws = WorldState.fromSequence('world', [String(stateId)]);
  const output = agent.step(ws);                   // incremental — no reset needed
  console.log(output.actionDistribution);          // confidence over tokens
  if (output.predictionError > 0.8) {
    console.log('Anomaly detected — agent is surprised');
  }
}
```

### Clear memory for training pipeline correctness
```javascript
agent.clearMemory();   // resets trace buffer + step count, preserves trie/lexicon
// Use between unrelated training runs to avoid state leakage
```

### Action distribution for confidence-based decisions
```javascript
const output = agent.step(ws);
const dist = output.actionDistribution;  // { token: probability, ... }
const best = Object.entries(dist).sort((a, b) => b[1] - a[1])[0];
if (best && best[1] > 0.5) {
  console.log(`High confidence: ${best[0]} (${(best[1]*100).toFixed(0)}%)`);
} else {
  console.log('Low confidence — defer to human');
}
```

## File Map

| File | Purpose |
|------|---------|
| `src/index.js` | Public API exports |
| `src/agent/conscious-agent.js` | Main class |
| `src/agent/perceptual-map.js` | `perceive()` — P function |
| `src/agent/decision-map.js` | `decide()` — D function |
| `src/core/meta-trie.js` | Self-model (implicit) |
| `src/core/self-token.js` | "I" attractor |
| `src/core/strange-loop.js` | Self-reference scoring |
| `src/combination/operator.js` | ⊗ combine (n-ary) + fuse decomposition |
| `src/io/serialization.js` | Save/load/clone |
| `src/world/world-builder.js` | World construction |
| `src/world/self-world.js` | SelfWorld (v2.0) — explicit self-perception |
