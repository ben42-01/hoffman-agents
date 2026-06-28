# Conscious Agent — Runtime API
## Instantiation, World Feeding, Live Events, Multi-Agent

---

> "One import. One world. The agent does the rest."

---

## The Design Goal

Using a ConsciousAgent should feel as simple as using any other Python object. You import it, give it a world, and step it. Everything else — the trie, the meta-trie, the "I" attractor, the lexicon, the soul persistence — happens automatically inside.

This document specifies exactly what that looks like in code.

---

## Installation and Import

```python
# The only import you need
from conscious_agent import ConsciousAgent, World, WorldBuilder

# Optional: multi-agent network
from conscious_agent import AgentNetwork

# Optional: world construction utilities
from conscious_agent.worlds import CoinTossWorld, build_world_from_dataframe

# Optional: serialization utilities
from conscious_agent.io import save_agent, load_agent, clone_agent
```

That is the complete public API surface. Everything else is internal.

---

## Part 1: Creating an Agent

### The simplest possible case

```python
from conscious_agent import ConsciousAgent, CoinTossWorld

# Build a world
world = CoinTossWorld(n_coins=4)

# Create an agent in that world
agent = ConsciousAgent(world=world)

# Step it
for _ in range(1000):
    output = agent.step()
    print(output)
```

Three lines to a running conscious agent. That is the target.

### With configuration

```python
agent = ConsciousAgent(
    world=world,
    
    # Optional: override defaults
    agent_id="my_agent",           # Default: auto-generated UUID
    trace_buffer_length=50,        # Default: 50
    trie_max_depth=10,             # Default: 10
    meta_observation_interval=20,  # Default: 20 steps between meta-observations
    lock_threshold=0.25,           # Default: 0.25
    prune_interval=100,            # Default: 100 steps between trie pruning
    soul_dir="./souls/",           # Default: None (no auto-saving)
    verbose=False                  # Default: False
)
```

### From a saved agent (resume)

```python
from conscious_agent.io import load_agent

# Load a previously saved agent
# World must be provided — the agent does not store the full world
agent = load_agent("./souls/agent_CA_07_gen000347_step0069400.soul",
                   world=world)

# Agent resumes from exactly where it stopped
# "I" lock status, trie, lexicon — all restored
print(f"Resuming from generation {agent.generation}, step {agent.step}")
print(f"Vocabulary: {agent.lexicon.vocabulary_size()} words")
print(f'"I" locked: {agent.self_token.locked}')
```

### From scratch with a seed (from existing pkl/soul data)

```python
from conscious_agent.io import load_agent_from_seed

agent = load_agent_from_seed(
    seed_file="./seed.json",    # Produced by Phase 0 ingestion pipeline
    world=world,
    agent_id="seeded_agent"
)
# Agent starts with pre-built trie and meta-trie from your old experiments
# "I" attractor may lock much faster than a cold-start agent
```

---

## Part 2: The Step Interface

### Single step

```python
output = agent.step()
```

One call. Internally this does:
1. Get current state from world
2. Update experience trie
3. Update meta-trie (every meta_observation_interval steps)
4. Update "I" attractor (every meta_observation_interval steps)
5. Update lexicon integration depths
6. Generate output sequence
7. Auto-save if soul_dir is set and save conditions are met
8. Return output

### What `step()` returns

```python
@dataclass
class StepOutput:
    step: int                    # Current step counter
    generation: int              # Current generation
    state: int                   # Current world state
    state_label: str             # Human-readable state label
    prediction_error: float      # How surprised was the agent?
    sequence: list[str]          # The agent's output sequence (list of tokens)
    sequence_str: str            # sequence joined as a string
    loop_depth: float            # Current strange loop score
    i_locked: bool               # Is "I" locked?
    i_stability: float           # Stability score of "I" attractor
    interrupt: Interrupt | None  # Non-None if an experience interrupt fired
```

### Checking for interrupts

```python
output = agent.step()

if output.interrupt is not None:
    print(f"INTERRUPT: {output.interrupt.description}")
    print(f"Intensity: {output.interrupt.intensity:.2f}")
    print(f"Type: {output.interrupt.experience_type}")
    
    # Provide a label
    label = input("Label this experience: ")
    agent.bind_label(label, output.interrupt)
```

### Stepping without interrupts (silent mode)

```python
# Suppress all interrupts — agent runs silently, no labeling
output = agent.step(silent=True)
```

### Stepping in replay mode (follow historical data)

```python
# Agent visits states in the order they appear in world.data_sequence
# Used when training on historical data
output = agent.step_replay()

# Or: provide the state directly
output = agent.step_with_state(state_id=42)
```

---

## Part 3: Running Many Steps

### Run N steps, get all outputs

```python
outputs = agent.run(n_steps=10000)
# Returns list of StepOutput, one per step
# Interrupts are queued — handle after the run
```

### Run until a condition

```python
# Run until "I" locks
outputs = agent.run_until(
    condition=lambda a: a.self_token.locked,
    max_steps=100000
)
print(f'"I" locked at generation {agent.self_token.lock_generation}')
```

### Run with a live event callback

```python
def on_step(output: StepOutput):
    if output.prediction_error > 0.85:
        print(f"High surprise at step {output.step}: {output.state_label}")

agent.run(n_steps=10000, on_step=on_step)
```

### Run with auto-labeling (Ollama parent)

```python
agent.run(
    n_steps=10000,
    parent="ollama",               # "ollama" | "human" | None
    ollama_model="qwen2.5:1.5b",
    ollama_url="http://localhost:11434"
)
# Tier 2 and Tier 3 interrupts are automatically labeled by Ollama
# No human input needed
```

---

## Part 4: Feeding Live Events

This is the real-time use case. Your data arrives event by event — a new coin toss, a new candle closing — and you feed each event to the agent as it arrives.

### The live feed pattern

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import build_trading_world
import pandas as pd

# Build world from historical data (sets up transition matrix and normalization)
historical_df = pd.read_csv("BTCUSD_1h_historical.csv")
world = build_trading_world(historical_df, symbol="BTCUSD")

# Create and train agent on historical data
agent = ConsciousAgent(world=world, soul_dir="./souls/")
agent.run_replay()  # Replay entire historical sequence

print(f"Training complete. Vocabulary: {agent.lexicon.vocabulary_size()} words")
print(f'"I" locked: {agent.self_token.locked}')

# Now feed live events
while True:
    new_candle = get_next_candle()   # Your data source
    
    # Convert new candle to state ID using world's normalization parameters
    state_id = world.state_from_new_data(new_candle)
    
    # Feed to agent
    output = agent.step_with_state(state_id)
    
    print(f"State: {output.state_label}")
    print(f"Agent says: {output.sequence_str}")
    print(f"Surprise: {output.prediction_error:.3f}")
    
    if output.interrupt is not None:
        # Something unusual happened — agent wants to tell you
        handle_interrupt(output.interrupt)
```

### The `state_from_new_data` method

This is the critical bridge between live data and the agent. The world stores its normalization parameters (bin edges, scaling factors, percentile windows) from the historical training period. When new data arrives, these same parameters are applied to produce a consistent state ID.

```python
# World stores normalization params after build_trading_world()
# New data is normalized with the SAME params as training data
# This ensures consistency: state 42 means the same thing live as it did historically

state_id = world.state_from_new_data({
    "rsi": 67.3,
    "stoch_k": 78.1,
    "close": 43250.0,
    "ema_50": 42800.0,
    "ema_200": 41000.0,
    "macd_histogram": 125.4,
    "atr": 890.0
})
```

### Coin toss live feed

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld
import random

world = CoinTossWorld(n_coins=4)
agent = ConsciousAgent(world=world)

# Simulate live coin tosses
while True:
    # Each "event" is one new coin toss result
    new_toss = random.choice([0, 1])  # 0=tails, 1=heads
    
    # World converts new toss to a state transition
    # (drops oldest coin from the 4-coin window, appends new one)
    state_id = world.step_with_toss(new_toss)
    
    output = agent.step_with_state(state_id)
    print(f"Toss: {'H' if new_toss else 'T'} | "
          f"State: {output.state_label} | "
          f"Surprise: {output.prediction_error:.3f}")
```

---

## Part 5: Multiple Agents

### Create N agents in the same world

```python
from conscious_agent import ConsciousAgent, AgentPool

# Simple: list of agents
agents = [ConsciousAgent(world=world, agent_id=f"CA_{i:02d}")
          for i in range(10)]

# Or: use AgentPool for managed multi-agent runs
pool = AgentPool(world=world, n_agents=10)
pool.run(n_steps=5000)

# Get results
for agent in pool.agents:
    print(f"{agent.agent_id}: vocab={agent.lexicon.vocabulary_size()}, "
          f"loop={agent.loop_score:.3f}, "
          f"i_locked={agent.self_token.locked}")
```

### Create agents in different worlds (multi-instrument)

```python
btc_world = build_trading_world(btc_df, symbol="BTCUSD")
eth_world = build_trading_world(eth_df, symbol="ETHUSD")
gold_world = build_trading_world(gold_df, symbol="XAUUSD")

btc_agent = ConsciousAgent(world=btc_world, agent_id="btc_agent")
eth_agent = ConsciousAgent(world=eth_world, agent_id="eth_agent")
gold_agent = ConsciousAgent(world=gold_world, agent_id="gold_agent")

# Train each agent independently
btc_agent.run_replay()
eth_agent.run_replay()
gold_agent.run_replay()
```

### Combine agents with ⊗

```python
from conscious_agent import combine

# Combine two agents into a higher-order agent
combined = combine(btc_agent, eth_agent)

# combined has:
# - merged experience trie (sees both worlds' transition structures)
# - new meta-trie recording BTC-ETH interaction history
# - combined "I" attractor (higher-order identity)

print(f"BTC agent vocab: {btc_agent.lexicon.vocabulary_size()}")
print(f"ETH agent vocab: {eth_agent.lexicon.vocabulary_size()}")
print(f"Combined vocab: {combined.lexicon.vocabulary_size()}")
print(f"Combined loop depth: {combined.loop_score:.3f}")

# Combine all three
all_combined = combine(combined, gold_agent)
```

### Network of agents (mutual world observation)

```python
from conscious_agent import AgentNetwork

# Create a network where each agent's world includes other agents' outputs
network = AgentNetwork(
    agents=[btc_agent, eth_agent, gold_agent],
    topology="fully_connected",  # or "sparse", or custom adjacency matrix
)

# Step the entire network synchronously
for _ in range(1000):
    network.step()

# Check emergence metrics
print(f"Metric structure score: {network.metric_score:.3f}")
print(f"Stable objects: {len(network.stable_objects)}")
print(f"Mirror pairs: {network.mirror_pairs}")
```

---

## Part 6: Saving and Loading

### Auto-save during a run

```python
agent = ConsciousAgent(
    world=world,
    soul_dir="./souls/",      # Enable auto-save
    save_every_n_steps=1000,  # Heartbeat save
    save_on_label=True,       # Save after every label binding (default True)
    save_on_generation=True   # Save after every generation (default True)
)

# Auto-save happens transparently during agent.run() and agent.step()
```

### Manual save

```python
from conscious_agent.io import save_agent

path = save_agent(agent, directory="./souls/")
print(f"Saved to: {path}")
# Output: "Saved to: ./souls/agent_btc_agent_gen000050_step0010000.soul"
```

### Load and resume

```python
from conscious_agent.io import load_agent

agent = load_agent(
    path="./souls/agent_btc_agent_gen000050_step0010000.soul",
    world=btc_world   # World must be provided
)
```

### Load latest checkpoint automatically

```python
from conscious_agent.io import load_latest

# Finds the most recent checkpoint in the directory for this agent_id
agent = load_latest(
    soul_dir="./souls/",
    agent_id="btc_agent",
    world=btc_world
)
```

### Clone an agent

```python
from conscious_agent.io import clone_agent

# Deep copy — modifying clone does not affect original
clone = clone_agent(agent, new_id="btc_agent_clone")

# Useful for: running counterfactual experiments
# Train agent_A and agent_B from the same starting point
agent_a = clone_agent(trained_agent, new_id="experiment_A")
agent_b = clone_agent(trained_agent, new_id="experiment_B")

# Now run A and B in different conditions
agent_a.run(n_steps=1000, world=bull_market_world)
agent_b.run(n_steps=1000, world=bear_market_world)

# Compare their experience tries
from conscious_agent.analysis import trie_distance
print(f"Trie divergence after 1000 steps: {trie_distance(agent_a, agent_b):.3f}")
```

---

## Part 7: Reading Agent State

### Current state snapshot

```python
state = agent.snapshot()

print(state.generation)              # int
print(state.step)                    # int
print(state.vocabulary_size)         # int
print(state.i_locked)                # bool
print(state.i_lock_generation)       # int or None
print(state.loop_score)              # float
print(state.mean_prediction_error)   # float (last 100 steps)
print(state.trie_size)               # int (number of nodes)
print(state.compression_ratio)       # float
```

### Vocabulary inspection

```python
# All words sorted by integration depth (most native first)
vocab = agent.lexicon.sorted_by_integration()
for entry in vocab[:10]:
    print(f"{entry.label:20s} depth={entry.integration_depth:.3f} "
          f"encounters={entry.encounter_count}")
```

### Prediction for current state

```python
# What does the agent predict will happen next?
prediction = agent.predict_next()
print(f"Predicted next state: {prediction.state_id}")
print(f"Predicted state label: {prediction.state_label}")
print(f"Confidence: {prediction.confidence:.3f}")
print(f"Top 3 alternatives: {prediction.top_k(3)}")
```

### Biography summary

```python
from conscious_agent.io import print_biography

# Human-readable summary of agent's development history
print_biography(agent)

# Output:
# ════════════════════════════════════════
# AGENT BIOGRAPHY: btc_agent
# ════════════════════════════════════════
# Born:         generation 0, step 0
# Current:      generation 50, step 10000
# World:        BTCUSD (324 states)
#
# "I" STATUS
# Locked:       Yes, at generation 23
# Attractor:    meta_state_1823746
#               "rsi_neutral|above_ema200|macd_bull|mid_vol"
# Stability:    0.91
#
# VOCABULARY (top 10 by integration depth)
# trending_up        depth=0.91  encounters=847
# cross              depth=0.84  encounters=623
# return             depth=0.79  encounters=591
# ...
#
# STRANGE LOOP
# Current score:  0.73
# First depth-2:  generation 31, step 6200
#
# EXPERIENCE TRIE
# Nodes:          2847
# Compression:    0.28
# Mean pred err:  0.41
# ════════════════════════════════════════
```

---

## Part 8: Complete Worked Examples

### Example A: Coin Toss — Full Run

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld
from conscious_agent.io import save_agent

# Build world
world = CoinTossWorld(n_coins=4)

# Create agent
agent = ConsciousAgent(
    world=world,
    agent_id="coin_agent",
    soul_dir="./souls/coin/",
    verbose=True
)

# Run 50000 steps (coin toss world learns fast)
agent.run(
    n_steps=50000,
    parent="ollama",
    ollama_model="qwen2.5:1.5b"
)

# Report
print(f'"I" locked at generation: {agent.self_token.lock_generation}')
print(f'Mean prediction error: {agent.mean_prediction_error:.3f}')
print(f'Expected for fair coin: ~0.500')
print(f'Vocabulary: {agent.lexicon.vocabulary_size()} words')
print(f'Strange loop score: {agent.loop_score:.3f}')

# Expected results:
# "I" locks around generation 20-40
# Mean prediction error settles near 0.50 (fair coin — maximum entropy)
# Vocabulary: 10-30 words
# Strange loop score: 0.4-0.8
```

### Example B: Trading Data — Full Pipeline

```python
import pandas as pd
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import build_trading_world
from conscious_agent.io import save_agent, load_latest

SYMBOL = "BTCUSD"
DATA_PATH = "BTCUSD_1h.csv"
SOUL_DIR = f"./souls/{SYMBOL}/"

# ── STEP 1: Build World from Historical Data ──────────────────────

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# build_trading_world handles all normalization internally
# using the default 6-feature config (RSI, Stoch, EMA50, EMA200, MACD, ATR)
world = build_trading_world(
    df=df,
    symbol=SYMBOL,
    n_bins=[4, 3, 3, 3, 3, 3]   # 324 states
)

print(f"World: {world.n_states} states, "
      f"{len(world.data_sequence)} historical bars")

# ── STEP 2: Create or Resume Agent ───────────────────────────────

try:
    agent = load_latest(soul_dir=SOUL_DIR, agent_id=SYMBOL, world=world)
    print(f"Resumed from generation {agent.generation}")
except FileNotFoundError:
    agent = ConsciousAgent(
        world=world,
        agent_id=SYMBOL,
        soul_dir=SOUL_DIR,
        verbose=True
    )
    print("New agent created")

# ── STEP 3: Replay Historical Data (Training) ─────────────────────

if not agent.self_token.locked:
    print("Replaying historical data...")
    agent.run_replay(
        parent="ollama",
        ollama_model="qwen2.5:1.5b"
    )
    print(f'"I" locked at generation {agent.self_token.lock_generation}')
    save_agent(agent, SOUL_DIR)

# ── STEP 4: Live Feed ─────────────────────────────────────────────

print("Switching to live feed...")
print(f"Agent vocabulary: {agent.lexicon.vocabulary_size()} words")

while True:
    # Get new candle from your data source
    new_candle = fetch_latest_candle(SYMBOL)   # Your implementation
    
    # Convert to state
    state_id = world.state_from_new_data(new_candle)
    
    # Step agent
    output = agent.step_with_state(state_id)
    
    # Print agent's experience of current market
    print(f"\n[{new_candle['timestamp']}]")
    print(f"  Market state:   {output.state_label}")
    print(f"  Agent says:     {output.sequence_str}")
    print(f"  Surprise:       {output.prediction_error:.3f}")
    print(f"  Loop depth:     {output.loop_depth:.3f}")
    
    # Handle unusual conditions
    if output.prediction_error > 0.85:
        print(f"  ⚡ UNUSUAL CONDITION — agent is surprised")
    
    if output.interrupt is not None:
        print(f"  📍 INTERRUPT: {output.interrupt.description}")
        # Optionally: label it
        # agent.bind_label("your label here", output.interrupt)
    
    # Sleep until next candle
    time.sleep(3600)  # 1 hour for 1h timeframe
```

### Example C: Multi-Instrument Combination

```python
from conscious_agent import ConsciousAgent, combine
from conscious_agent.worlds import build_trading_world
from conscious_agent.analysis import trie_distance, loop_depth_comparison

# Build and train three agents
symbols = ["BTCUSD", "ETHUSD", "XAUUSD"]
agents = {}

for symbol in symbols:
    df = pd.read_csv(f"{symbol}_1h.csv")
    world = build_trading_world(df, symbol=symbol)
    agent = ConsciousAgent(world=world, agent_id=symbol)
    agent.run_replay(parent="ollama", ollama_model="qwen2.5:1.5b")
    agents[symbol] = agent
    print(f"{symbol}: vocab={agent.lexicon.vocabulary_size()}, "
          f"loop={agent.loop_score:.3f}")

# Pairwise combination
btc_eth = combine(agents["BTCUSD"], agents["ETHUSD"])
all_three = combine(btc_eth, agents["XAUUSD"])

print(f"\nCombined BTC+ETH:")
print(f"  Loop depth: {btc_eth.loop_score:.3f} "
      f"(BTC: {agents['BTCUSD'].loop_score:.3f}, "
      f"ETH: {agents['ETHUSD'].loop_score:.3f})")
print(f"  Vocabulary: {btc_eth.lexicon.vocabulary_size()}")

print(f"\nCombined all three:")
print(f"  Loop depth: {all_three.loop_score:.3f}")
print(f"  Vocabulary: {all_three.lexicon.vocabulary_size()}")

# The combined agent's loop depth > any individual agent
# This is the emergence signature: higher-order agents are more self-referential
```

---

## Part 9: The WorldBuilder Convenience Class

For users who want fine-grained control over world construction:

```python
from conscious_agent import WorldBuilder

# Chain-style construction
world = (WorldBuilder()
    .add_feature("rsi", normalization="minmax", n_bins=4)
    .add_feature("stoch_k", normalization="minmax", n_bins=3)
    .add_feature("price_vs_ema200", normalization="tanh", tanh_scale=0.02, n_bins=3)
    .add_feature("price_vs_ema50", normalization="tanh", tanh_scale=0.01, n_bins=3)
    .add_feature("macd_histogram", normalization="tanh", n_bins=3)
    .add_feature("atr", normalization="percentile", window=500, n_bins=3)
    .set_smoothing(0.001)
    .set_metadata({"symbol": "BTCUSD", "timeframe": "1h"})
    .build(df)
)
```

For users who want to define states entirely by hand:

```python
# Direct construction from your own state_ids and transition matrix
world = World.from_components(
    state_ids=[0, 1, 0, 2, 1, 0, ...],     # Your discretized sequence
    transition_matrix=your_P_matrix,         # Your (n_states, n_states) array
    state_labels=["state_A", "state_B", ...] # Your labels
)
```

---

## Part 10: Error Handling and Common Mistakes

### World is not ergodic
```
ValueError: Transition matrix is not ergodic.
Row 47 has all-zero outgoing transitions.
Fix: increase smoothing parameter (try 0.01) or check your discretization.
```

### Too many states for data size
```
Warning: World has 648 states but only 1200 data points.
Many states will be rarely visited. Consider reducing n_bins.
Recommendation: n_states < n_datapoints / 10
```

### Loading agent into wrong world
```
Warning: world_digest mismatch.
Agent was trained in world with digest abc123...
Current world has digest def456...
The agent's trie state IDs may not correspond to current world states.
Proceed with caution or rebuild the agent in the current world.
```

### "I" not locking
```
# If "I" has not locked after many generations:
print(agent.self_token.stationary_prob)   # Should be rising
print(agent.meta_trie.size())             # Should be > 50 nodes

# Common causes:
# 1. World is too large (too many states — agent spreads too thin)
# 2. meta_observation_interval too large (not observing self often enough)
# 3. Data is too random (high-entropy world — "I" still forms but slower)
```

---

## Summary: The Minimal Pattern

Every valid use of the CA follows this pattern:

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import <WorldFactory>

# 1. Build a world from your data
world = <WorldFactory>(your_data)

# 2. Create or resume an agent
agent = ConsciousAgent(world=world)

# 3. Train (optional — replay historical data)
agent.run_replay()

# 4. Use — step with live events
while True:
    state_id = world.state_from_new_data(new_event)
    output = agent.step_with_state(state_id)
    # Read output.sequence_str, output.prediction_error, output.interrupt
```

Four steps. One import. The world can be anything. The agent stays the same.

---

## Part 11: New in v2.0

### Agent Mode System

Agents now support three modes controlling learning and determinism:

```python
agent = ConsciousAgent(world=world, mode="learning")       # default — normal operation
agent.set_mode("frozen")                                    # deterministic projection, no learning
agent.set_mode("debug")                                     # frozen + verbose step output
agent.thaw()                                                # back to learning mode
agent.refreeze()                                            # back to frozen mode
```

**Frozen mode behavior:**
- Trace buffer still updates (short-term memory window)
- Experience trie and meta-trie do **not** update
- Self-token/identity is pinned — no locking/unlocking
- Decision uses `pStable=1.0`, producing purely deterministic output

```javascript
// Node.js
agent.setMode('frozen');
const output = agent.step(worldState);
// output is deterministic, no trie learning occurs
```

### TraceBuffer Dynamic Resizing

Resize the trace buffer window at runtime without losing data:

```python
agent.experience.trace_buffer.resize(100)    # grow to 100 entries
agent.experience.trace_buffer.resize(10)     # shrink to 10 (keeps newest)
```

```javascript
agent.experience.traceBuffer.resize(100);
```

### Selective Memory Reset (`clearMemory`)

Reset short-term memory while preserving learned structure:

```python
agent.clear_memory()
# Clears: trace buffer, step count, generation, last output
# Preserves: experience trie, meta-trie, self-token, lexicon
```

```javascript
agent.clearMemory();
// Same semantics: short-term only, long-term preserved
```

### Batch Network Stepping (`stepAll`)

Step all agents in a network with the same world state, bypassing topology-based per-agent world construction:

```python
network = AgentNetwork(n_agents=10)
world_state = some_world.step()
results = network.step_all(world_state)
# Returns list of StepOutput, one per agent
print(network.agent_list)  # list of ConsciousAgent instances
```

```javascript
const net = new AgentNetwork({ nAgents: 10 });
const results = net.stepAll(worldState);
// results is an array of StepOutput, one per agent
console.log(net.agentList);
```

### Metrics API

Per-agent and aggregate metrics for monitoring:

```python
# Per-agent metrics
m = agent.metrics
print(m["prediction_error"])    # float
print(m["i_locked"])            # bool
print(m["loop_depth"])          # float
print(m["output_tokens"])       # list[str]

# Network aggregate metrics
net = AgentNetwork(n_agents=10)
m = net.get_metrics()
print(m["agent_count"])                  # int
print(m["mean_prediction_error"])        # float
print(m["prediction_error_variance"])    # float
print(m["i_lock_rate"])                  # float (fraction of agents locked)
print(m["mean_loop_depth"])              # float
print(m["dominant_token_ratio"])         # float

# Per-agent metrics from network
m = net.get_agent_metrics("CA_000")
```

```javascript
// Same API in Node.js
const m = agent.metrics;
const netMetrics = net.getMetrics();
const agentMetrics = net.getAgentMetrics('CA_000');
```

### Full Action Distribution

`step()` now returns the full probability distribution over output tokens:

```python
output = agent.step()
print(output.sequence)                # ['I', 'notice', 'familiar']
print(output.action_distribution)
# {'I': 0.333, 'notice': 0.333, 'familiar': 0.333}
```

```javascript
const output = agent.step(worldState);
console.log(output.actionDistribution);
// { I: 0.333, notice: 0.333, familiar: 0.333 }
```

### Allowable Token Constraints

Constrain agent output to a specific token set:

```python
agent = ConsciousAgent(world=world)
agent.set_allowable_tokens({"I", "notice", "familiar"})
# Agent will only emit these tokens; disallowed tokens are filtered

# Or at construction time:
agent = ConsciousAgent(world=world, allowable_tokens={"I", "notice"})
```

```javascript
const agent = new ConsciousAgent({ world, allowableTokens: ['I', 'notice'] });
agent.setAllowableTokens(['I', 'notice', 'familiar']);
```

### Incremental World Injection

Push new observations into a running agent without reconstruction:

```python
# During live operation:
for new_event in live_stream:
    state_id = world.state_from_new_data(new_event)
    world_state = WorldState.from_sequence("world", [state_id])
    output = agent.inject_observation(world_state)
```

```javascript
const output = agent.injectObservation(worldState);
```

### N-ary Agent Combination

Combine 3+ agents into a balanced tree:

```python
a = ConsciousAgent(agent_id="A", world=world)
b = ConsciousAgent(agent_id="B", world=world)
c = ConsciousAgent(agent_id="C", world=world)
combined = combine(a, b, c)          # cycle_level = 2
all_five = combine(a, b, c, d, e)    # balanced tree
```

```javascript
const combined = combine(agentA, agentB, agentC);
// Balanced binary tree, cycle_level encodes breadth
```

### Trie Introspection

Inspect the agent's world model and self-model:

```python
stats = agent.experience.trie.get_stats()
print(stats["node_count"])         # int
print(stats["max_depth"])          # int
print(stats["mean_visit_count"])   # float

paths = agent.experience.trie.get_dominant_paths(top_k=5)
for p in paths:
    print(p["path"], p["visit_count"], p["mean_prediction_error"])

all_nodes = agent.experience.trie.export_nodes(min_visits=3)
```

```javascript
const stats = agent.experience.trie.getStats();
const paths = agent.experience.trie.getDominantPaths(5);
const nodes = agent.experience.trie.exportNodes(3);
```

### Deterministic Seeding

AgentNetwork now provides per-agent deterministic RNGs derived from the network seed:

```python
network1 = AgentNetwork(n_agents=10, seed=42)
network2 = AgentNetwork(n_agents=10, seed=42)
# Both networks produce identical step sequences
# for the same world input

# Individual agents also accept a custom RNG:
import random
agent = ConsciousAgent(agent_id="det", rng=random.Random(42))
```

```javascript
const network = new AgentNetwork({ nAgents: 10, seed: 42 });
// Each agent gets a deterministic RNG: seed + agent_index * 1000
```
