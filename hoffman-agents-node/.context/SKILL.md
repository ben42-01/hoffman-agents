# conscious-agent Node.js Lib — Agent Skill

## Overview

Zero-dependency Node.js library for building self-referential agents (Hoffman's Conscious Realism). Same architecture as the Python lib but implemented entirely with stdlib (`crypto`, `fs`, `http`, `Float64Array`).

### What This Library IS

A **philosophical simulation toolkit** for studying self-referential structure-learning agents — agents that build compressed models of their world AND of themselves (meta-cognition). The core algorithm is Hoffman's Markov chain formalism for conscious agents: perceive → meta-observe → decide.

### What This Library Is NOT

This is **NOT** a general-purpose ML prediction engine. It is not a replacement for LSTMs, transformers, ARIMA, or gradient-boosted trees for time-series forecasting. Key architectural differences:

| ML Prediction | conscious-agent |
|---|---|
| Predicts continuous/output values (regression) | Predicts discrete state-ID hashes (categorical) |
| Optimizes MSE, cross-entropy, etc. | Optimizes surprise (prediction error) |
| Can learn magnitude (price up 2% vs 5%) | All magnitude info is lost in hashing |
| Ground truth is the raw data | Ground truth is the agent's own interface token |
| Train/test split on held-out data | Incremental, online learning only |

---

## Verdict

### When to use this library

This library shines in **one specific role**: as a **decorrelated auxiliary signal generator** that augments existing models. Its value is that it sees the world differently — hash-based, categorical, self-referential — so it fails on different inputs than gradient-based models. In an ensemble, that decorrelation lifts overall performance.

Use it when you need:

| Need | What the library provides |
|------|--------------------------|
| **Anomaly / regime change detection** without labeled data | `predictionError` spikes when input patterns shift — a free surprise detector |
| **Multi-observer consensus** on state | `SharedMeaningTracker` measures how aligned multiple agents are — low sharedness = regime fragmentation |
| **Well-calibrated confidence** for discrete state prediction | `predictNext().confidence` tracks actual accuracy (80% confident ≈ 80% right) |
| **Self-modeling / meta-cognition** research | `MetaTrie` + `SelfTokenState` = how "I" locks emerge from self-observation |
| **"Fitness Beats Truth" experiments** | Compare compressed vs granular feature representations — fewer features can predict better |
| **Online / incremental learning** with no retraining | Trie insertions are O(log n), no gradient computation, no training epochs |

### When NOT to use this library

| Situation | Why it fails |
|-----------|-------------|
| **Regression or continuous-value prediction** | All outputs are discrete categorical hashes. Magnitude is destroyed on ingestion. |
| **Standalone prediction engine** | `predictNext()` accuracy on noisy data is ~40-50% — no better than a naive baseline. |
| **Large-vocabulary or high-cardinality problems** | Trie scales with unique state count. >1000 states becomes memory-heavy and slow. |
| **Supervised learning from labeled data** | No loss function, no labels, no train/test split. The agent is unsupervised. |
| **Real-time / low-latency production systems** | Step execution is ~0.1ms — fast enough for 1-minute bars, not for tick-level HFT. |
| **Any task where you need to know "how much" not just "what"** | The hash discards magnitude entirely. "Up 2%" and "Up 10%" look identical. |

### The hybrid approach

This library is **never the star of the show**. It's the supporting actor that provides signals your main model cannot:

```
Your existing model (LSTM, XGBoost, etc.) → primary predictions
                              +
conscious-agent signals (surprise, sharedness, confidence) → risk modulation
                              =
Better ensemble because error modes are decorrelated
```

The most proven pattern: use `predictionError` as a **regime-change gate**. When the agent is surprised, reduce conviction in your primary model. When the agent is unsurprised (low error, stable confidence), proceed normally. This is a ~20-line integration that adds no latency, requires no labeled data, and cannot hurt because it only ever reduces exposure — it never generates a false positive trade entry. It can only prevent bad trades. This is the safest and most effective entry point for any domain. If your domain is well-suited to the good-fit use cases above, invest further. If not, this single pattern already provides meaningful risk management with minimal cost.

---

## 🚫 Deep Dive: Why Trading Failed

### The core mismatch

Trading is fundamentally about predicting **continuous numeric values** (price direction × magnitude × time horizon). This library operates on **discrete categorical hashes** of interface tokens.

```
Price $100.52 → hashed to bucket "price_bucket_201"
Price $100.53 → hashed to bucket "price_bucket_201"  (same!)
Price $105.00 → hashed to bucket "price_bucket_210"  (5% move)
Price $95.00  → hashed to bucket "price_bucket_190"  (5% move)
→ Agent sees: 201 → 201 → 210 → 190
→ It learns that bucket 201 sometimes stays 201, sometimes jumps
→ It CANNOT learn magnitude: a $0.01 move and a $5.00 move look identical
```

### What goes wrong specifically

1. **Magnitude blindness**: "Up 0.1%" and "Up 5%" hash to the same state if they fall in the same bin. The entire continuous price signal is destroyed on ingestion.

2. **No regression objective**: The agent optimizes for `predictionError` (surprise about the next hash). It does not optimize for MSE, MAE, directional accuracy, Sharpe ratio, or any trading-relevant metric.

3. **No profit/loss feedback**: The agent cannot learn "that prediction was costly" — it only learns "that prediction was surprising." There's no reward signal tied to actual trading outcomes.

4. **Cold-start problem on novel prices**: Every time price enters a previously unseen bucket, predictionError spikes to 1.0. For a volatile asset, this happens constantly — the agent is perpetually "surprised" and never converges.

5. **Tick-level unpredictability**: Financial markets are nearly-random at the tick level. `predictNext()` accuracy on a uniform-random world is ~40-50% — and financial data at micro-timescales is even noisier.

---

## 🔷 Hybrid Trading Architecture: conscious-agent + Conventional Tools

### The insight

The library should **never** replace your price-prediction stack (LSTM, XGBoost, ARIMA). Instead, it provides **decorrelated auxiliary signals** that no conventional ML model produces, because its internal structure is fundamentally different:

| Conventional ML | conscious-agent |
|---|---|
| Gradient descent / backprop | Incremental trie insertion |
| Continuous floating-point math | Discrete hash-based state machine |
| Optimizes for accuracy (MSE, etc.) | Optimizes for surprise minimization |
| Single model, single view | Multi-agent with meta-cognition |
| Black-box weights | Fully interpretable trie paths |

This decorrelation is valuable: an ensemble benefits most when its members make **different kinds of mistakes**. A trie-based agent and an LSTM will fail on different data regimes — combining them can outperform either alone.

### What conscious-agent brings to a trading stack

| Signal | Source | What it measures | How to use it |
|--------|--------|-----------------|---------------|
| **Regime change alarm** | `predictionError` moving average | Market entering unfamiliar territory — structure has shifted | Trigger model retraining, switch strategy, reduce position size |
| **Surprise spike** | `predictionError` per step | Individual events that don't fit learned patterns | Flag for manual review, override automated execution |
| **Confidence calibration** | `predictNext().confidence` | How predictable the next state is under learned patterns | One input to position sizing — low confidence → reduce exposure |
| **Multi-agent consensus** | `SharedMeaningTracker.sharedness` | How aligned multiple agents are on market structure | Low sharedness = regime fragmentation = reduce leverage |
| **Market coherence** | `agent.isILocked`, `selfToken.stationaryProb` | Whether market behavior is consistent enough for a stable self-model to form | I-locked = coherent regime; never locking = regime too chaotic |
| **Fitness beat truth** | Compare prediction error across discretizations | Optimal feature granularity for current market | Adaptive feature selection — use coarser bins in volatile markets |

### Reference architecture

```
                      ┌─────────────────────────────────┐
                      │         RAW MARKET DATA          │
                      │  (price, volume, order book, ...) │
                      └──────────┬──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌─────────────────┐       ┌─────────────────┐
          │  Conventional    │       │  conscious-agent │
          │  ML pipeline     │       │  swarm (5-20     │
          │  (LSTM/XGBoost)  │       │  agents, varying │
          │                  │       │  discretization) │
          │  → price pred    │       │                  │
          │  → direction     │       │  → regime change │
          │  → volatility    │       │  → surprise flag │
          │  → volume        │       │  → confidence    │
          └────────┬────────┘       │  → consensus     │
                   │                │  → coherence     │
                   │                └────────┬─────────┘
                   │                         │
                   └──────────┬──────────────┘
                              ▼
                   ┌────────────────────┐
                   │   ENSEMBLE FUSION   │
                   │   (weighted voting  │
                   │    + regime-gated   │
                   │    position sizing) │
                   └────────┬───────────┘
                            ▼
                   ┌────────────────────┐
                   │   TRADING DECISION  │
                   │  (entry, exit,      │
                   │   position size,    │
                   │   risk override)    │
                   └────────────────────┘
```

### Pattern 1: Regime-gated position sizing

Use the agent's surprise level to modulate allocations from your primary model:

```javascript
const agent = new ConsciousAgent({ agentId: 'regime_sizer' });
const primaryModel = loadLSTMModel(); // your existing model

// Each market tick: feed both models
for (const tick of marketStream) {
  const ws = discretizeTick(tick);          // your discretization
  const output = agent.step(ws);

  const lstmSignal = primaryModel.predict(tick);
  const surprise = output.predictionError;

  // Gating logic
  let baseAllocation = 1.0;
  if (surprise > 0.6) baseAllocation *= 0.3;   // regime shift → reduce 70%
  else if (surprise > 0.4) baseAllocation *= 0.7; // moderate surprise → reduce 30%

  const finalSignal = lstmSignal * baseAllocation;
  executeTrade(finalSignal, { size: baseAllocation * defaultPositionSize });
  log({ surprise, lstmSignal, baseAllocation, finalSignal });
}
```

**Why this works**: The agent's surprise metric is free (no labels, no retraining), instantaneous, and decorrelated from gradient-based loss. When the LSTM is confidently wrong (e.g., during a regime it hasn't seen), the agent's surprise spikes and reduces position size — acting as a safety circuit breaker.

### Pattern 2: Multi-agent consensus for regime classification

Use multiple agents with different feature views to detect agreement/fragmentation:

```javascript
const { SharedMeaningTracker } = require('conscious-agent');

// Create agents with different market views
const agents = [
  new ConsciousAgent({ agentId: 'price_only' }),   // price buckets only
  new ConsciousAgent({ agentId: 'vol_aware' }),    // price + volatility
  new ConsciousAgent({ agentId: 'volume_aware' }), // price + volume
  new ConsciousAgent({ agentId: 'fast' }),          // high-frequency discretization
  new ConsciousAgent({ agentId: 'slow' }),          // low-frequency discretization
];
const tracker = new SharedMeaningTracker();

for (const tick of marketStream) {
  // Each agent gets its own interface to the same world
  agents[0].step(wsPriceOnly(tick));
  agents[1].step(wsPricePlusVol(tick));
  agents[2].step(wsPricePlusVolume(tick));
  agents[3].step(wsFast(tick));
  agents[4].step(wsSlow(tick));
}

const lexicons = {};
agents.forEach((a, i) => { lexicons[`agent_${i}`] = a.experience.lexicon; });
const consensus = tracker.snapshot(lexicons, generation);

if (consensus.sharedness > 0.5) {
  // Agents agree on market structure → high confidence trading regime
  // Increase conviction in primary model's signals
} else {
  // Agents disagree → market is fragmented or transitioning
  // Reduce position sizes, widen stops
  log(`Low consensus (${consensus.sharedness.toFixed(2)}) — reducing exposure`);
}
```

**Why this works**: Sharedness is a unique signal no conventional model produces. It measures whether market behavior is consistent enough that independent observers build similar internal models. Low sharedness → conflicting narratives → high uncertainty → reduce.

### Pattern 3: "Fitness Beats Truth" feature selection

Hoffman's core result: an agent with **fewer, compressed features** can predict better than one with complete information. Apply this to feature engineering:

```javascript
// Truth agent: all features, full precision
const truthWorld = buildWorldFromDataFrame(rawMarketData, [
  { name: 'price', normalization: 'minmax', nBins: 50 },
  { name: 'volume', normalization: 'minmax', nBins: 50 },
  { name: 'volatility', normalization: 'tanh', nBins: 20 },
]);

// Interface agent: compressed, fitness-optimized features
const ifaceWorld = buildWorldFromDataFrame(rawMarketData, [
  { name: 'price', normalization: 'minmax', nBins: 8 },
  { name: 'volume', normalization: 'percentile', nBins: 4, params: { window: 100 } },
]);

const truthAgent = new ConsciousAgent({ agentId: 'truth' });
const ifaceAgent = new ConsciousAgent({ agentId: 'interface' });

// Train both on (different) training window
for (const row of trainingData) {
  truthAgent.step(truthWorld.step());
  ifaceAgent.step(ifaceWorld.step());
}

// Compare: interface agent often has LOWER prediction error
// despite having FEWER features — this is Fitness Beats Truth.
// Use this to guide feature granularity: in volatile markets,
// fewer/broader bins may produce more robust predictions.
```

**Why this works**: This is the core Hoffman insight validated empirically. It directly challenges the assumption that more features = better predictions. Use this to dynamically select feature resolution.

### Important caveats for production

1. **Never use `predictNext()` alone for trade direction**. Its accuracy on noisy data is ~40-50%. Always combine with a conventional price-prediction model.

2. **predictionError is a relative, not absolute, signal**. It spikes on novelty, but "novelty" doesn't mean "bad" — it could be a genuine opportunity. Use it to modulate, not to veto.

3. **Multi-agent consensus requires training time**. Agents need enough steps to build shared lexicons (~500+ steps minimum).

4. **The library is not real-time safe**. Step execution is ~0.1ms in Node, so it's fast enough for 1-minute bars but not for tick-level HFT.

5. **No backtest integration**. The library has no concept of P&L, slippage, or transaction costs. You must implement that in your conventional stack.

### Summary

| Use pattern | conscious-agent role | Conventional tools | Combined value |
|------------|---------------------|-------------------|----------------|
| Regime gate | Surprise-based position sizing | Price prediction (LSTM) | Survive regime changes without retraining |
| Consensus filter | Sharedness as uncertainty metric | Trend/mean-reversion signals | Avoid false signals during regime transitions |
| Feature selection | Compare interface vs truth accuracy | Feature engineering | Optimize bin count for current volatility regime |
| Ensemble diversity | Hash-based state machine predictions | Gradient-based predictions | Decorrelated error modes → better ensemble |

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

---

## Use Case Patterns

### 1. Anomaly / Regime Change Detection (✅ Best fit)

```javascript
// Train on "normal" data
const agent = new ConsciousAgent({ agentId: 'anomaly_detector' });
for (const event of normalEvents) {
  const ws = WorldState.fromSequence('world', [event]);
  agent.step(ws);
}

// Now monitor live stream — predictionError spikes = anomaly
while (true) {
  const ws = WorldState.fromSequence('world', [await getNextEvent()]);
  const output = agent.step(ws);
  if (output.predictionError > 0.6) {
    console.log(`REGIME CHANGE detected at step ${output.step}`);
    // predictionError > 0.6: state was never seen or pattern shifted
  } else if (output.predictionError > 0.4) {
    console.log(`Subtle anomaly at step ${output.step}`);
    // 0.4-0.6: state was seen but transition was surprising
  }
}
```

**Pros**: Zero labeled data, works on categorical streams, natural surprise metric
**Cons**: Cannot distinguish "novel normal" from "actual anomaly", no magnitude info

---

### 2. Consensus / Shared Meaning (✅ Good fit)

```javascript
const { SharedMeaningTracker } = require('conscious-agent');

const tracker = new SharedMeaningTracker();
const agents = [new ConsciousAgent({ agentId: 'A' }),
                new ConsciousAgent({ agentId: 'B' })];
const world = new SimpleWorld({ nStates: 10 });

// Both agents experience the same world
for (let gen = 0; gen < 100; gen++) {
  const ws = world.step();
  agents.forEach(a => a.step(ws));

  // Measure sharedness every 20 steps
  if (gen % 20 === 0) {
    const result = tracker.snapshot(
      { a: agents[0].experience.lexicon,
        b: agents[1].experience.lexicon },
      gen
    );
    console.log(`Generation ${gen}: sharedness = ${result.sharedness.toFixed(2)}`);
  }
}
```

**Pros**: Unique capability — no other lib measures shared meaning formation
**Cons**: Only works for agents on same world, Lexicon needs sufficient vocabulary

---

### 3. Identity Formation (✅ Core design)

```javascript
// Track how "I" emerges from meta-cognitive self-modeling
const agent = new ConsciousAgent({
  agentId: 'self',
  metaObservationInterval: 10,
});

for (let step = 0; step < 1000; step++) {
  const ws = world.step();
  const output = agent.step(ws);
  if (output.iLocked) {
    console.log(`"I" locked at step ${output.step}, gen ${output.generation}`);
    console.log(`  stationary prob: ${agent.experience.selfToken.stationaryProb.toFixed(3)}`);
    console.log(`  meta states: ${agent.experience.metaTrie.registrySize}`);
    break;
  }
}

// Ablation: prevent I-lock
const ablated = new ConsciousAgent({
  agentId: 'ablated',
  experience: new ExperienceSpace({
    selfToken: new SelfTokenState({ lockThreshold: 1.5 }), // never locks
  }),
});
```

**Pros**: Unique to this library, scientifically interesting
**Cons**: Not useful for practical prediction tasks

---

### 4. Fitness Beats Truth (✅ Good fit)

```javascript
const { buildWorldFromDataFrame } = require('conscious-agent');

// Create an "interface" world (compressed = fitness-optimized)
const interfaceWorld = buildWorldFromDataFrame(rawData, [
  { name: 'feature_1', normalization: 'minmax', nBins: 4 },
  { name: 'feature_2', normalization: 'minmax', nBins: 4 },
]);

// Create a "truth" world (all granular states)
const truthWorld = buildWorldFromDataFrame(rawData, [
  { name: 'feature_1', normalization: 'minmax', nBins: 20 },
  { name: 'feature_2', normalization: 'minmax', nBins: 20 },
]);

// Interface agent predicts better despite having LESS information
// See examples/01_fitness_beats_truth/ for the full implementation
```

**Pros**: Demonstrates Hoffman's core theory, compelling pedagogical tool
**Cons**: Requires careful data preparation, result is known in advance

---

### 5. Discrete State Forecast (✅ Good with caveats)

```javascript
// Only use when states are truly categorical with few (<50) values
const agent = new ConsciousAgent({ agentId: 'forecaster' });
const knownStates = ['idle', 'processing', 'warning', 'error', 'fatal'];

// Train on known transition patterns
for (const state of trainingSequence) {
  const ws = WorldState.fromSequence('world', [state]);
  agent.step(ws);
}

// Predict next state
const prediction = agent.predictNext();
if (prediction && prediction.confidence > 0.8) {
  console.log(`Confident next state: ${prediction.stateLabel}`);
  // Act on high-confidence prediction
} else if (prediction) {
  console.log(`Top candidates: ${prediction.topK(3).map(t => t.stateLabel).join(', ')}`);
  // Use only as one input among many
}
```

**Pros**: Well-calibrated confidence, top-K alternatives, incremental learning
**Cons**: Only works for small discrete state spaces (< 50 states). Each state must have been seen ~50+ times for reliable probabilities. No magnitude or temporal distance.

---

### 6. Multi-Agent Cognitive Simulation (✅ Good fit)

```javascript
// Create specialists, freeze them, combine into higher-order agent
const specialistA = new ConsciousAgent({ agentId: 'vision', world: visionWorld });
specialistA.run(5000);
specialistA.setMode('frozen');

const specialistB = new ConsciousAgent({ agentId: 'audio', world: audioWorld });
specialistB.run(5000);
specialistB.setMode('frozen');

// Higher-order agent perceives both specialists' outputs
const brain = combine(specialistA, specialistB);
console.log(`Brain cycle level: ${brain.cycleLevel}`);
console.log(`Brain constituents: ${[...brain.constituentIds]}`);

// The brain's SelfWorld blends both modalities
const brainWorld = new SelfWorld(brainWorld, brain);
brain.setWorld(brainWorld);
brain.run(3000);
console.log(`Brain I-locked: ${brain.isILocked}`);
```

**Pros**: Unique hierarchical composition, no other lib does this
**Cons**: Complexity grows with cycle level, hard to interpret

---

## Common Patterns (utility)

### Create agent
```javascript
const agent = new ConsciousAgent({ agentId: 'my_agent' });
for (let i = 0; i < 100; i++) {
  const ws = WorldState.fromSequence('world', ['state_1']);
  agent.step(ws);
}
```

### Predict next state (v2.1)
```javascript
const prediction = agent.predictNext();
// Returns Prediction { stateId, stateLabel, confidence, topK(n) }
// or null if no experience yet.
if (prediction) {
  console.log(`Expecting state ${prediction.stateId} (conf: ${prediction.confidence.toFixed(2)})`);
  const top3 = prediction.topK(3);    // [{ stateId, stateLabel, confidence }, ...]
}
```

**How it works**: Looks up `experience.lastWorldStateId` in the trie, returns
the child (next state) with the highest visit count. Confidence = top child's
visit count / total children visits — a well-calibrated Bayesian probability.

**Performance**: Accuracy equals the world's transition predictability.
- Deterministic cycle: **100%** accuracy, **100%** confidence
- Strongly biased (90%): **~95%** accuracy, **~95%** confidence
- Uniform random (e.g., SimpleWorld default): **~40-50%** — reflects true
  world randomness, not a bug. The agent cannot predict noise, but knows it.
- Confidence is well-calibrated: 80%-confidence predictions are right ~80%.

**Use with `topK(n)`** for robust decision-making:
```javascript
const top = prediction.topK(3);
if (top[0].confidence > 0.8) {
  // High certainty — act on best guess
} else {
  // Low certainty — sample from top K or defer
}
```

### From config (v2.1)
```javascript
const agent = ConsciousAgent.fromConfig('my_agent', {
  agent: {
    selfToken: { lockThreshold: 0.3, lockConsecutiveRequired: 5 },
    metaObservationInterval: 15,
    expressionTemp: 0.7,
  },
});
```

### Build world from data rows (v2.1)
```javascript
const { buildWorldFromDataFrame } = require('conscious-agent');
const rows = [{ temp: 1.2, humidity: 65 }, { temp: 2.5, humidity: 70 }];
const world = buildWorldFromDataFrame(rows);
// Auto-detects numeric columns. Or pass explicit specs:
const world2 = buildWorldFromDataFrame(rows, [
  { name: 'temp', normalization: 'minmax', nBins: 4 },
]);
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

### Topology introspection (v2.1)
```javascript
const { Topology } = require('conscious-agent');
const top = new Topology({ nAgents: 10, seed: 42 });

top.getConnectionStrength(0, 1);        // get weight between agents
top.maybeAddConnection(0, 9);           // probabilistically add new link
top.getAgentObservers(5);               // which agents observe agent 5
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
| `src/index.js` | Public API exports (47 exports: all core classes + utilities) |
| `src/agent/conscious-agent.js` | Main class — ConsciousAgent, StepOutput, Prediction |
| `src/agent/perceptual-map.js` | `perceive()` — P function |
| `src/agent/decision-map.js` | `decide()` — D function |
| `src/core/experience-trie.js` | ExperienceTrie, TrieNode |
| `src/core/meta-trie.js` | Self-model (implicit) — MetaTrie, MetaStateSnapshot |
| `src/core/self-token.js` | "I" attractor — SelfTokenState |
| `src/core/strange-loop.js` | Self-reference scoring + population metrics |
| `src/core/trie-compression.js` | prune, traceDistance, mergeSimilarPaths |
| `src/core/token-inventor.js` | inventToken, isInventedToken |
| `src/core/experience-lexicon.js` | ExperienceLexicon, LexiconEntry |
| `src/combination/operator.js` | ⊗ combine (n-ary) + fuse decomposition |
| `src/io/serialization.js` | Save/load/clone — serialize/deserialize |
| `src/network/agent-network.js` | AgentNetwork, Topology, InteractionCycle |
| `src/world/world-builder.js` | World, WorldBuilder, CoinTossWorld, Normalizer, FeatureSpec, buildWorldFromDataFrame |
| `src/world/self-world.js` | SelfWorld (v2.0) — explicit self-perception |
| `src/meaning/shared-meaning.js` | SharedMeaningTracker |
