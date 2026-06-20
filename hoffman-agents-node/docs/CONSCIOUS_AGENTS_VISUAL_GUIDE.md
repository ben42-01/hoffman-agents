# Conscious Agents — Visual Guide

> Mermaid diagrams explaining how Hoffman's Conscious Realism maps onto code.
> Examples use the **traffic light** (red/yellow/green) to keep it concrete.

---

## 1. The Traffic Light — Why Perception Is an Interface

Hoffman's central claim: **evolution shapes our perceptions to guide adaptive behavior, not to show reality as it is.**

A traffic light does not turn "red." Photons at ~700nm wavelength enter your eye, your visual cortex processes them, and you experience "red." But the *redness* is not in the light — it is in your experience. A mantis shrimp sees the same light entirely differently. The red is an **interface** that helps you *stop*, which helps you *survive*.

```mermaid
flowchart LR
    subgraph Reality["Physical Reality (unknowable)"]
        W["700nm photon<br/>(wavelength)"]
    end
    subgraph Interface["Perceptual Interface"]
        R["Red<br/>STOP"]
        Y["Yellow<br/>CAUTION"]
        G["Green<br/>GO"]
    end
    subgraph Agent["Conscious Agent"]
        D["Decision: Stop"]
    end
    W -->|"P: Perception map"| R
    R -->|"D: Decision map"| D
    Y -.->|"alternative state"| D
    G -.->|"alternative state"| D
```

**Key insight:** The agent never accesses the 700nm photon directly. It only ever knows `Red`. The `X` (experience space) of the agent contains colors, not wavelengths. The `W` (world space) is whatever generates those experiences.

---

## 2. The Conscious Agent Six-Tuple

Hoffman formally defines a conscious agent as:

```
CA = (X, G, P, W, A, D)
```

```mermaid
flowchart TD
    subgraph CA["Conscious Agent (X, G, P, W, A, D)"]
        direction TB
        X["X — Experience Space<br/>what it feels like to be this agent<br/>Example: Red, Yellow, Green"]
        G["G — Action Space<br/>what the agent can do<br/>Example: Stop, Go, Wait"]
        W["W — World Space<br/>other agents' experience spaces<br/>Example: the light's internal state"]
        P["P — Perceptual Map<br/>W x X to X<br/>how world changes experience"]
        A_map["A — Action Map<br/>X x G to G<br/>how experience shapes action"]
        D["D — Decision Map<br/>X to G<br/>what the agent does right now"]
    end
    W -->|"input"| P
    X -->|"input"| P
    P -->|"new experience"| X
    X -->|"input"| A_map
    G -->|"input"| A_map
    A_map -->|"action tendency"| D
    D -->|"action"| G
    G -.->|"affects"| W
```

**In our codebase:**

| Tuple | Implementation | File |
|-------|----------------|------|
| X — Experience Space | `ExperienceTrie` + `MetaTrie` + `SelfTokenState` | `conscious_agents/agent/perceptual_map.py` |
| G — Action Space | Token sequences the agent generates | `conscious_agents/agent/conscious_agent.py` |
| W — World Space | Other agents' output / market labels | In Prediction Engine: `market_stream.py` labels |
| P — Perceptual Map | `perceive()` — updates trie from observation | `conscious_agents/agent/perceptual_map.py:12` |
| A — Action Map | `step().action` — maps experience to next action | `conscious_agents/agent/conscious_agent.py` |
| D — Decision Map | `generate_output()` — produces token sequence | `conscious_agents/agent/conscious_agent.py` |

---

## 3. The Perception-Action Cycle (Traffic Light in Motion)

Every timestep, the agent runs this loop:

```mermaid
sequenceDiagram
    participant W as World (other agents)
    participant P as Perception Map
    participant X as Experience Space
    participant D as Decision Map
    participant G as Action (output)

    loop Every Step
        W->>P: World state W(t)
        P->>X: Update experience X(t+1) = P(W(t), X(t))
        X->>D: Current experience X(t+1)
        D->>G: Action G(t+1) = D(X(t+1))
        G->>W: Broadcast output (affects other agents)
    end
```

**Traffic light example:**
1. You see the light turn **Yellow** (W → P)
2. Your experience becomes "caution, might turn red" (P → X)
3. Your decision is "prepare to stop" (D → G)
4. You lift off the accelerator (G → W — you affect the car, which is another agent)

---

## 4. The Trace — How Experience Space Is Built

In our implementation, the experience space `X` has four components:

```
X = (T, M, I, L)
    T: Experience Trie — compressed world model
    M: Meta-Trie — self-model (trie over trace states)
    I: "I" Attractor — the agent's stable identity
    L: Experience Lexicon — labeled experiences
```

```mermaid
flowchart TD
    subgraph X["Experience Space X"]
        direction TB
        T["T — Experience Trie<br/>Learns: Red → Stop<br/>Green → Go<br/>(transition probabilities)"]
        M["M — Meta-Trie<br/>Observes: I often see Red<br/>I usually Stop<br/>(self-observation)"]
        I["I — Self-Token Attractor<br/>Locks when: stationary distribution<br/>converges (identity is stable)"]
        L["L — Experience Lexicon<br/>Labels: Red=Stop<br/>Green=Go<br/>(vocabulary)"]
    end

    W_obs["World Observation"] --> T
    T -->|trace states| M
    M -->|stationary distribution| I
    T -.->|labeled patterns| L
```

**Traffic light example:**
- **Trie** starts empty. You see Red → Stop enough times that the trie learns `P(Stop | Red) ≈ 0.99`.
- **Meta-Trie** watches your own trace: "I see Red, I Stop. I see Green, I Go." It models *you*.
- **Self-Token** locks when your meta-trie's stationary distribution converges — you have a consistent identity as a driver who responds to traffic lights.
- **Lexicon** labels the experience: "Red" means "stop," "Green" means "go."

---

## 5. The Strange Loop — Self-Observation (The "I")

The strange loop is the agent observing itself. The meta-trie records the agent's own trace, creating a self-model. When the meta-trie converges to a stationary distribution, the self-token locks — this is the birth of the "I."

```mermaid
flowchart TD
    subgraph Agent["The Agent"]
        direction TB
        T["Experience Trie<br/>'I see Red'"]
        M["Meta-Trie<br/>'I notice I see Red'"]
        I["Self-Token 'I'<br/>locked = stable identity"]
    end

    T -->|"trace snapshot<br/>every 20 steps"| M
    M -->|"stationary distribution<br/>converges"| I
    I -->|"protects meta-state"| M
    M -->|"influences perception"| T
```

**The depth of the loop is measurable:**

| Depth | Pattern | Meaning |
|-------|---------|---------|
| 0 | "Red" | No self-reference |
| 1 | "I see Red" | Self-awareness |
| 2 | "I notice I see Red" | Meta-self-awareness |
| 3 | "I wonder why I notice I see Red" | Introspection |

**In the codebase:** `conscious_agents/core/strange_loop.py:compute_self_reference_score()` counts these depths. The self-token locks in `conscious_agents/core/self_token.py` when the meta-trie's stationary distribution converges — the agent has proven to itself that it has a stable identity.

---

## 6. The Plus Circle ⊕ — Agent Combination

The **combination operator ⊕** (oplus, aka the "plus circle" or "Quotiented Fusion Simplex") combines two agents into a higher-order agent with an emergent experience space.

```mermaid
flowchart LR
    subgraph CA1["Agent 1"]
        X1["X1: Red/Green driver"]
        T1["T1: Red→Stop, Green→Go"]
    end
    subgraph CA2["Agent 2"]
        X2["X2: Car behavior model"]
        T2["T2: Brake→Slow, Gas→Fast"]
    end
    O["⊕<br/>Combine"]
    subgraph CA12["Higher-Order Agent"]
        X12["X12: Driving awareness<br/>(not reducible to X1 or X2 alone)"]
        T12["T12: Merged trie<br/>Red→Stop+Brake→Slow"]
        M12["M12: Joint meta-trie<br/>records interactions"]
        I12["I12: Combined identity<br/>'I am a driver'"]
    end

    CA1 --> O
    CA2 --> O
    O --> CA12
```

**Algebraic properties enforced in code:**

| Property | Rule | Code Location |
|----------|------|---------------|
| Associativity | (CA1 ⊕ CA2) ⊕ CA3 = CA1 ⊕ (CA2 ⊕ CA3) | `combination/operator.py:verify_associativity()` |
| Non-commutativity | CA1 ⊕ CA2 ≠ CA2 ⊕ CA1 | `combination/operator.py:verify_non_commutativity()` |
| Identity | CA ⊕ CA0 = CA (trivial agent) | `combination/operator.py:verify_identity()` |

**The combination performs four merges:**

| Merge | Function | File |
|-------|----------|------|
| T1 ⊕ T2 | `merge_tries()` — joint world model | `combination/trie_merge.py` |
| M1 ⊕ M2 | `build_joint_meta_trie()` — shared self-model | `combination/meta_merge.py` |
| I1 ⊕ I2 | `combine_attractors()` — fused identity | `combination/attractor_combine.py` |
| L1 ⊕ L2 | `merge_lexicons()` — shared vocabulary | `combination/lexicon_merge.py` |

---

## 7. Agent Network — Reality as Mutual Observation

The deepest claim: **the world W of any agent is the experience space X of other agents.** There is no agent-independent reality.

```mermaid
flowchart LR
    subgraph Reality["There is no external world"]
        CA1["Agent 1<br/>X1: driver experience"]
        CA2["Agent 2<br/>X2: car experience"]
        CA3["Agent 3<br/>X3: road experience"]
    end

    CA1 -->|"observes W2 = X2"| CA2
    CA2 -->|"observes W3 = X3"| CA3
    CA3 -->|"observes W1 = X1"| CA1
    CA1 -.->|"⊕ combine when ripe"| CA2
    CA2 -.->|"⊕ combine when ripe"| CA3
```

**What this means:**
- Agent 1's world is whatever Agent 2 is experiencing
- Agent 2's world is whatever Agent 3 is experiencing
- Agent 3's world is whatever Agent 1 is experiencing
- There is no traffic light "out there" — only agents experiencing each other's experiences

**In the Prediction Engine:** Each of the 10K agents observes market labels from the `HybridMarketStream` — these labels are the "world" for every agent. The agents don't access the market directly; they access a label that represents what another process (the WebSocket stream) experienced.

---

## 8. Full Architecture — How Everything Connects

```mermaid
flowchart TD
    subgraph Input["World / Market Data"]
        L["Labels from stream<br/>(ticker_up, ob_bid, etc.)"]
    end

    subgraph Agent["Conscious Agent"]
        direction TB
        T["Experience Trie<br/>compresses observations<br/>into transition model"]
        P["Perceptual Map<br/>perceive(): update from observation"]
        M["Meta-Trie<br/>observe_self(): models own trace"]
        S["Self-Token 'I'<br/>locks when identity stable"]
        D["Decision Map<br/>step(): predict next state"]
    end

    subgraph Output["Signal Generation"]
        PE["Prediction Error<br/>|expected - observed|"]
        DIR["Direction Bias<br/>up/down weight"]
        CONF["Confidence<br/>spiking agent ratio"]
    end

    L --> P
    P --> T
    T -->|"trace states"| M
    M -->|"stationary dist"| S
    T -->|"prediction"| D
    D -->|"compare with observation"| PE
    PE --> DIR
    PE --> CONF
```

**In the Prediction Engine specifically:**
- 10K agents × 5 domains (QUICK_SCALP, MEDIUM_TRADE, LONG_TREND, REVERSAL, REGIME)
- Each domain has different ergodic parameters controlling how agents explore vs stabilize
- The convergence bar smooths all domains into a single BUY/SELL/HOLD signal
- The anticipation engine learns which system-state signatures precede market moves
- The self-token lock ensures agents only form identity on real (not simulated) data

---

## Reference: Key Code Locations

| Concept | File | Key Function |
|---------|------|-------------|
| Agent definition | `conscious_agents/agent/conscious_agent.py` | `ConsciousAgent` class |
| Perception | `conscious_agents/agent/perceptual_map.py` | `perceive()` |
| Experience trie | `conscious_agents/agent/experience_trie.py` | `ExperienceTrie` |
| Meta-trie (self-model) | `conscious_agents/core/meta_trie.py` | `MetaTrie.observe_self()` |
| Self-token (I) | `conscious_agents/core/self_token.py` | `SelfTokenState.update()` |
| Strange loop | `conscious_agents/core/strange_loop.py` | `compute_self_reference_score()` |
| Combination (⊕) | `conscious_agents/combination/operator.py` | `combine()` |
| Trie merge | `conscious_agents/combination/trie_merge.py` | `merge_tries()` |
| Attractor combine | `conscious_agents/combination/attractor_combine.py` | `combine_attractors()` |
| Lexicon merge | `conscious_agents/combination/lexicon_merge.py` | `merge_lexicons()` |
| Meta-merge | `conscious_agents/combination/meta_merge.py` | `build_joint_meta_trie()` |
| Network | `conscious_agents/network/agent_network.py` | `AgentNetwork.combine_agents()` |
| Agent persistence | `conscious_agents/core/soul_persistence.py` | `save_soul()` / `load_soul()` |
| Fusion engine | `conscious_agents/fusion_engine.py` | `FusionEngine` |
| Prediction world | `conscious_agents/prediction/prediction_world.py` | `PredictionWorld` |
| Anticipation | `conscious_agents/prediction/anticipation.py` | `AnticipationEngine` |
