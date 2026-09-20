# conscious-agent Python Lib — Agent Skill

## v3.0 changes (read first)

- **Math.** Agents default to `math_version="v3"`. `math_version="legacy"` reproduces 2.x exactly. Full spec: `docs/MATHEMATICAL_MODEL.md`.
- **"I" lock.** It now requires a real attractor in the self-chain: evidence, ergodicity, dominance above uniform, occupancy and stability. It can unlock. Tune it with `lock_margin` (default 0.15), `min_transitions` (20), `min_transitions_per_state` (2), `min_occupancy` (0.6), `unlock_margin` (0.05). `lock_threshold` is used only by `"legacy"`. To disable locking in an ablation, set `lock_margin=2`. Inspect with `agent.ergodic_stats()["lock"]["criteria"]`.
- **Earliest lock.** At default settings no lock can happen before step ~460, and structureless worlds never lock.
- **Reproducibility.** Pass `seed` for mulberry32; seeded runs are identical in Node and Python.
- **Combination.** `combine` is commutative and associative, and `fuse` restores constituents exactly. `product_kernel(a, b)` gives M_a ⊗ M_b.
- **New modules.** `kernels/` (`MarkovKernel`, `FormalConsciousAgent`), `math/markov.py` (stationary, period, mixing, `spectral_dimension`, `irreversibility`, `dobrushin`), `analysis/bell.py` (CHSH, quantum-set test, signalling), `legacy/` (2.x math).
- **Spectral gap is not a quantum signature.** A small gap means slow mixing; a classical clock has one. See example 02 and `docs/Q_AND_A.md#corrections`.

## Overview

Minimal-dependency Python library for building self-referential agents (Hoffman's Conscious Realism). Agents learn by *inhabiting* Markov worlds — building compressed tries over state sequences and meta-tries over their own trace buffers.

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
| **Anomaly / regime change detection** without labeled data | `prediction_error` spikes when input patterns shift — a free surprise detector |
| **Multi-observer consensus** on state | `SharedMeaningTracker` measures how aligned multiple agents are — low sharedness = regime fragmentation |
| **Well-calibrated confidence** for discrete state prediction | `predict_next().confidence` tracks actual accuracy (80% confident ≈ 80% right) |
| **Self-modeling / meta-cognition** research | `MetaTrie` + `SelfTokenState` = how "I" locks emerge from self-observation |
| **"Fitness Beats Truth" experiments** | Compare compressed vs granular feature representations — fewer features can predict better |
| **Online / incremental learning** with no retraining | Trie insertions are O(log n), no gradient computation, no training epochs |

### When NOT to use this library

| Situation | Why it fails |
|-----------|-------------|
| **Regression or continuous-value prediction** | All outputs are discrete categorical hashes. Magnitude is destroyed on ingestion. |
| **Standalone prediction engine** | `predict_next()` accuracy on noisy data is ~40-50% — no better than a naive baseline. |
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

The most proven pattern: use `prediction_error` as a **regime-change gate**. When the agent is surprised, reduce conviction in your primary model. When the agent is unsurprised (low error, stable confidence), proceed normally. This is a ~20-line integration that adds no latency, requires no labeled data, and cannot hurt because it only ever reduces exposure — it never generates a false positive trade entry. It can only prevent bad trades. This is the safest and most effective entry point for any domain. If your domain is well-suited to the good-fit use cases above, invest further. If not, this single pattern already provides meaningful risk management with minimal cost.

---

## 🚫 Deep Dive: Why Trading Failed

### The core mismatch

Trading is fundamentally about predicting **continuous numeric values** (price direction × magnitude × time horizon). This library operates on **discrete categorical hashes** of interface tokens.

```python
price = 100.52       # → hashed to bucket "price_bucket_201"
price = 100.53       # → hashed to bucket "price_bucket_201"  (same!)
price = 105.00       # → hashed to bucket "price_bucket_210"  (5% move)
price = 95.00        # → hashed to bucket "price_bucket_190"  (5% move)
# Agent sees: 201 → 201 → 210 → 190
# It learns that bucket 201 sometimes stays 201, sometimes jumps
# It CANNOT learn magnitude: a $0.01 move and a $5.00 move look identical
```

### What goes wrong specifically

1. **Magnitude blindness**: "Up 0.1%" and "Up 5%" hash to the same state if they fall in the same bin. The entire continuous price signal is destroyed on ingestion.

2. **No regression objective**: The agent optimizes for `prediction_error` (surprise about the next hash). It does not optimize for MSE, MAE, directional accuracy, Sharpe ratio, or any trading-relevant metric.

3. **No profit/loss feedback**: The agent cannot learn "that prediction was costly" — it only learns "that prediction was surprising." There's no reward signal tied to actual trading outcomes.

4. **Cold-start problem on novel prices**: Every time price enters a previously unseen bucket, prediction_error spikes to 1.0. For a volatile asset, this happens constantly — the agent is perpetually "surprised" and never converges.

5. **Tick-level unpredictability**: Financial markets are nearly-random at the tick level. `predict_next()` accuracy on a uniform-random world is ~40-50% — and financial data at micro-timescales is even noisier.

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
| **Regime change alarm** | `prediction_error` moving average | Market entering unfamiliar territory — structure has shifted | Trigger model retraining, switch strategy, reduce position size |
| **Surprise spike** | `prediction_error` per step | Individual events that don't fit learned patterns | Flag for manual review, override automated execution |
| **Confidence calibration** | `predict_next().confidence` | How predictable the next state is under learned patterns | One input to position sizing — low confidence → reduce exposure |
| **Multi-agent consensus** | `SharedMeaningTracker.sharedness` | How aligned multiple agents are on market structure | Low sharedness = regime fragmentation = reduce leverage |
| **Market coherence** | `agent.is_i_locked`, `self_token.stationary_prob` | Whether market behavior is consistent enough for a stable self-model to form | I-locked = coherent regime; never locking = regime too chaotic |
| **Fitness beats truth** | Compare prediction error across discretizations | Optimal feature granularity for current market | Adaptive feature selection — use coarser bins in volatile markets |

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

```python
agent = ConsciousAgent(agent_id="regime_sizer")
# your existing model
primary_model = load_lstm_model()

for tick in market_stream:
    ws = discretize_tick(tick)
    output = agent.step(ws)

    lstm_signal = primary_model.predict(tick)
    surprise = output.prediction_error

    base_allocation = 1.0
    if surprise > 0.6:
        base_allocation *= 0.3      # regime shift → reduce 70%
    elif surprise > 0.4:
        base_allocation *= 0.7      # moderate surprise → reduce 30%

    final_signal = lstm_signal * base_allocation
    execute_trade(final_signal, size=base_allocation * default_position_size)
    log(f"surprise={surprise:.2f} lstm={lstm_signal:.3f} alloc={base_allocation:.2f}")
```

**Why this works**: The agent's surprise metric is free (no labels, no retraining), instantaneous, and decorrelated from gradient-based loss. When the LSTM is confidently wrong (e.g., during a regime it hasn't seen), the agent's surprise spikes and reduces position size — acting as a safety circuit breaker.

### Pattern 2: Multi-agent consensus for regime classification

Use multiple agents with different feature views to detect agreement/fragmentation:

```python
from conscious_agent import SharedMeaningTracker

agents = [
    ConsciousAgent(agent_id="price_only"),    # price buckets only
    ConsciousAgent(agent_id="vol_aware"),     # price + volatility
    ConsciousAgent(agent_id="volume_aware"),  # price + volume
    ConsciousAgent(agent_id="fast"),           # high-frequency discretization
    ConsciousAgent(agent_id="slow"),           # low-frequency discretization
]
tracker = SharedMeaningTracker()

for tick in market_stream:
    agents[0].step(ws_price_only(tick))
    agents[1].step(ws_price_plus_vol(tick))
    agents[2].step(ws_price_plus_volume(tick))
    agents[3].step(ws_fast(tick))
    agents[4].step(ws_slow(tick))

lexicons = {f"agent_{i}": a.experience.lexicon for i, a in enumerate(agents)}
consensus = tracker.snapshot(lexicons, generation=generation)

if consensus["sharedness"] > 0.5:
    # Agents agree on market structure → high confidence regime
    # Increase conviction in primary model's signals
else:
    # Agents disagree → market fragmented or transitioning
    log(f"Low consensus ({consensus['sharedness']:.2f}) — reducing exposure")
```

**Why this works**: Sharedness is a unique signal no conventional model produces. It measures whether market behavior is consistent enough that independent observers build similar internal models. Low sharedness → conflicting narratives → high uncertainty → reduce.

### Pattern 3: "Fitness Beats Truth" feature selection

Hoffman's core result: an agent with **fewer, compressed features** can predict better than one with complete information. Apply this to feature engineering:

```python
# Truth agent: all features, full precision
truth_world = build_world_from_dataframe(raw_market_data, [
    {"name": "price", "normalization": "minmax", "n_bins": 50},
    {"name": "volume", "normalization": "minmax", "n_bins": 50},
    {"name": "volatility", "normalization": "tanh", "n_bins": 20},
])

# Interface agent: compressed, fitness-optimized features
iface_world = build_world_from_dataframe(raw_market_data, [
    {"name": "price", "normalization": "minmax", "n_bins": 8},
    {"name": "volume", "normalization": "percentile", "n_bins": 4, "params": {"window": 100}},
])

truth_agent = ConsciousAgent(agent_id="truth")
iface_agent = ConsciousAgent(agent_id="interface")

for row in training_data:
    truth_agent.step(truth_world.step())
    iface_agent.step(iface_world.step())

# Interface agent often has LOWER prediction error despite FEWER features
# → Use this to guide feature granularity in volatile markets
```

**Why this works**: This is the core Hoffman insight validated empirically. It directly challenges the assumption that more features = better predictions. Use this to dynamically select feature resolution.

### Important caveats for production

1. **Never use `predict_next()` alone for trade direction**. Its accuracy on noisy data is ~40-50%. Always combine with a conventional price-prediction model.

2. **prediction_error is a relative, not absolute, signal**. It spikes on novelty, but "novelty" doesn't mean "bad" — it could be a genuine opportunity. Use it to modulate, not to veto.

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

## Self-Modelling (v2.0 — Critical for Agent Understanding)

This library has **four self-modelling mechanisms** (whether they amount to self-awareness is not something the code establishes):

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

3. **No numpy/scipy fallback**: Required for the Markov analysis (stationary distributions, spectral estimates). The Node.js port implements the same algorithms without dependencies.

## Commands

```bash
uv sync --group dev         # install (add --group dev for pytest)
uv run pytest tests/ -v     # run tests
uv run python examples/...  # run example
uv build && uv publish      # publish to PyPI
```

---

## Use Case Patterns

### 1. Anomaly / Regime Change Detection (✅ Best fit)

```python
# Train on "normal" data
agent = ConsciousAgent(agent_id="anomaly_detector")
for event in normal_events:
    ws = WorldState.from_sequence("world", [event])
    agent.step(ws)

# Now monitor live stream — prediction_error spikes = anomaly
while True:
    ws = WorldState.from_sequence("world", [get_next_event()])
    output = agent.step(ws)
    if output.prediction_error > 0.6:
        print(f"REGIME CHANGE detected at step {output.step}")
        # prediction_error > 0.6: state was never seen or pattern shifted
    elif output.prediction_error > 0.4:
        print(f"Subtle anomaly at step {output.step}")
        # 0.4-0.6: state was seen but transition was surprising
```

**Pros**: Zero labeled data, works on categorical streams, natural surprise metric
**Cons**: Cannot distinguish "novel normal" from "actual anomaly", no magnitude info

---

### 2. Consensus / Shared Meaning (✅ Good fit)

```python
from conscious_agent import SharedMeaningTracker

tracker = SharedMeaningTracker()
agents = [ConsciousAgent(agent_id="A"), ConsciousAgent(agent_id="B")]
world = SimpleWorld(n_states=10)

for gen in range(100):
    ws = world.step()
    for a in agents:
        a.step(ws)

    if gen % 20 == 0:
        result = tracker.snapshot(
            {"a": agents[0].experience.lexicon, "b": agents[1].experience.lexicon},
            gen,
        )
        print(f"Generation {gen}: sharedness = {result['sharedness']:.2f}")
```

**Pros**: Unique capability — no other lib measures shared meaning formation
**Cons**: Only works for agents on same world, Lexicon needs sufficient vocabulary

---

### 3. Identity Formation (✅ Core design)

```python
agent = ConsciousAgent(agent_id="self", meta_observation_interval=10)

for step in range(1000):
    ws = world.step()
    output = agent.step(ws)
    if output.i_locked:
        print(f'"I" locked at step {output.step}, gen {output.generation}')
        print(f"  stationary prob: {agent.experience.self_token.stationary_prob:.3f}")
        print(f"  meta states: {agent.experience.meta_trie.registry_size}")
        break

# Ablation: prevent I-lock
from conscious_agent import SelfTokenState, ExperienceSpace
ablated = ConsciousAgent(
    agent_id="ablated",
    experience=ExperienceSpace(
        self_token=SelfTokenState(lock_margin=2),  # never locks (dominance <= 1)
    ),
)
```

**Pros**: Unique to this library, scientifically interesting
**Cons**: Not useful for practical prediction tasks

---

### 4. Fitness Beats Truth (✅ Good fit)

```python
from conscious_agent import build_world_from_dataframe

# Create an "interface" world (compressed = fitness-optimized)
interface_world = build_world_from_dataframe(raw_data, [
    {"name": "feature_1", "normalization": "minmax", "n_bins": 4},
    {"name": "feature_2", "normalization": "minmax", "n_bins": 4},
])

# Create a "truth" world (all granular states)
truth_world = build_world_from_dataframe(raw_data, [
    {"name": "feature_1", "normalization": "minmax", "n_bins": 20},
    {"name": "feature_2", "normalization": "minmax", "n_bins": 20},
])

# Interface agent predicts better despite having LESS information
# See examples/01_fitness_beats_truth/ for the full implementation
```

**Pros**: Demonstrates Hoffman's core theory, compelling pedagogical tool
**Cons**: Requires careful data preparation, result is known in advance

---

### 5. Discrete State Forecast (✅ Good with caveats)

```python
agent = ConsciousAgent(agent_id="forecaster")
known_states = ["idle", "processing", "warning", "error", "fatal"]

for state in training_sequence:
    ws = WorldState.from_sequence("world", [state])
    agent.step(ws)

prediction = agent.predict_next()
if prediction and prediction.confidence > 0.8:
    print(f"Confident next state: {prediction.state_label}")
elif prediction:
    top3 = prediction.top_k(3)
    print(f"Top candidates: {[t['state_label'] for t in top3]}")
```

**Pros**: Well-calibrated confidence, top-K alternatives, incremental learning
**Cons**: Only works for small discrete state spaces (< 50 states). Each state must have been seen ~50+ times for reliable probabilities. No magnitude or temporal distance.

---

### 6. Multi-Agent Cognitive Simulation (✅ Good fit)

```python
specialist_a = ConsciousAgent(agent_id="vision", world=vision_world)
specialist_a.run(n_steps=5000)
specialist_a.set_mode("frozen")

specialist_b = ConsciousAgent(agent_id="audio", world=audio_world)
specialist_b.run(n_steps=5000)
specialist_b.set_mode("frozen")

brain = combine(specialist_a, specialist_b)
print(f"Brain cycle level: {brain.cycle_level}")
print(f"Brain constituents: {brain.constituent_ids}")

from conscious_agent.worlds import SelfWorld
brain_world = SelfWorld(brain_world, brain)
brain.set_world(brain_world)
brain.run(n_steps=3000)
print(f"Brain I-locked: {brain.is_i_locked}")
```

**Pros**: Unique hierarchical composition, no other lib does this
**Cons**: Complexity grows with cycle level, hard to interpret

---

## Common Patterns (utility)

### Create agent in a world
```python
agent = ConsciousAgent(agent_id="my_agent")
for _ in range(100):
    ws = WorldState.from_sequence("world", ["state_1"])
    agent.step(ws)
```

### Predict next state (v2.1)
```python
from conscious_agent import Prediction

prediction = agent.predict_next()
if prediction:
    print(f"Expecting state {prediction.state_id} (conf: {prediction.confidence:.2f})")
    top3 = prediction.top_k(3)   # [{state_id, state_label, confidence}, ...]
```

**How it works**: Looks up `last_world_state_id` in the trie, returns
the child (next state) with the highest visit count. Confidence = top child's
visit count / total children visits — a well-calibrated Bayesian probability.

**Performance**: Accuracy equals the world's transition predictability.
- Deterministic cycle: **100%** accuracy, **100%** confidence
- Strongly biased (90%): **~95%** accuracy, **~95%** confidence
- Uniform random (e.g., SimpleWorld default): **~40-50%** — reflects true
  world randomness, not a bug. The agent cannot predict noise, but knows it.
- Confidence is well-calibrated: 80%-confidence predictions are right ~80%.

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
st = SelfTokenState(lock_margin=2)  # never locks
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
from conscious_agent import markov
from conscious_agent.combination import meta_kernel
P = meta_kernel(agent).matrix  # the agent's own recurrent self-chain
gap = 1 - markov.second_eigenvalue_modulus(P)  # mixing speed; NOT a quantum signature
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

### Topology introspection (v2.1)
```python
from conscious_agent import Topology
top = Topology(n_agents=10, seed=42)
top.get_connection_strength(0, 1)     # get weight between agents
top.maybe_add_connection(0, 9)        # probabilistically add new link
top.get_agent_observers(5)            # which agents observe agent 5
```

---

## Creating a New Example

1. Create `examples/NN_name/` directory
2. Add a `main()` function with argument parser for config
3. Use SSE (Server-Sent Events) for real-time web dashboards
4. Test with: `uv run python examples/NN_name/script.py`

## File Map

| File | Purpose |
|------|---------|
| `src/conscious_agent/__init__.py` | Public API exports (45+ exports) |
| `src/conscious_agent/agent/conscious_agent.py` | Main agent class — ConsciousAgent, StepOutput, Prediction |
| `src/conscious_agent/agent/perceptual_map.py` | `perceive()` — P function |
| `src/conscious_agent/agent/decision_map.py` | `decide()` — D function |
| `src/conscious_agent/core/experience_trie.py` | ExperienceTrie, TrieNode |
| `src/conscious_agent/core/meta_trie.py` | Self-model (implicit) — MetaTrie, MetaStateSnapshot |
| `src/conscious_agent/core/self_token.py` | "I" attractor — SelfTokenState |
| `src/conscious_agent/core/strange_loop.py` | Self-reference scoring + population metrics |
| `src/conscious_agent/core/trie_compression.py` | prune, trace_distance, merge_similar_paths |
| `src/conscious_agent/core/token_inventor.py` | invent_token, is_invented_token |
| `src/conscious_agent/core/experience_lexicon.py` | ExperienceLexicon, LexiconEntry |
| `src/conscious_agent/combination/operator.py` | ⊗ combine + fuse decomposition |
| `src/conscious_agent/io/serialization.py` | Save/load/clone |
| `src/conscious_agent/network/agent_network.py` | AgentNetwork, Topology, InteractionCycle |
| `src/conscious_agent/world/world_builder.py` | World, WorldBuilder, CoinTossWorld, Normalizer, FeatureSpec, build_world_from_dataframe |
| `src/conscious_agent/world/self_world.py` | SelfWorld (v2.0) — explicit self-perception |
| `src/conscious_agent/meaning/shared_meaning.py` | SharedMeaningTracker |
