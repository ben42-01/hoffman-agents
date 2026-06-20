# conscious-agent Node.js Lib — Agent Skill

## Overview

Zero-dependency Node.js library for building self-referential agents. Same architecture as the Python lib but implemented entirely with stdlib (`crypto`, `fs`, `http`, `Float64Array`).

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

## Key Conventions

1. **State IDs are hashes** of `WorldState.sequences`. Always use `WorldState.fromSequence(agentId, sequence)` or `new WorldState({agentId: sequence})`.

2. **CamelCase**: JavaScript conventions — `agent.getOutput()`, `metaTrie.observeSelf()`, `traceBuffer.predictionErrorMean()`. See Python's snake_case equivalents.

3. **Spectral gap proxy**: No numpy eigenvalues. Uses row-concentration proxy:
   ```
   gap = (avgMaxRowProb - 1/n) / (1 - 1/n)
   ```
   Range: 0 (uniform) to 1 (deterministic).

4. **Float64Array** for matrix operations (stationary distribution, power iteration).

## Commands

```bash
node --test test/*.test.js      # run tests
node examples/04_stop_lights/stop_lights.js  # run example
npm pack                        # create tarball
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

### Custom lock threshold (ablation)
```javascript
const st = new SelfTokenState({ lockThreshold: 1.5 });
const exp = new ExperienceSpace({ selfToken: st });
const agent = new ConsciousAgent({ agentId: 'ablated', experience: exp });
```

### Combine
```javascript
const { combine } = require('conscious-agent');
const combined = combine(agentA, agentB);
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

## File Map

| File | Purpose |
|------|---------|
| `src/index.js` | Public API exports |
| `src/agent/conscious-agent.js` | Main class |
| `src/agent/perceptual-map.js` | `perceive()` — P function |
| `src/agent/decision-map.js` | `decide()` — D function |
| `src/core/meta-trie.js` | Self-model |
| `src/core/self-token.js` | "I" attractor |
| `src/combination/operator.js` | ⊗ combine |
| `src/io/serialization.js` | Save/load/clone |
| `src/world/world-builder.js` | World construction |
