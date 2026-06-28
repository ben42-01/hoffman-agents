# Self-Awareness in conscious-agent

**A feature document — Hoffman's Conscious Realism meets computational self-perception**

---

## Overview

The `conscious-agent` library implements Donald Hoffman's formal definition of a conscious agent: a six-tuple `(X, G, P, W, A, D)` where an agent's experience space X is private, its world W is made of other agents' outputs, and perception P is an interface that compresses W into X for fitness, not accuracy.

Within this framework, self-awareness emerges through **four mechanisms**, two implicit and two explicit:

| # | Mechanism | Type | Layer | What it does |
|---|-----------|------|-------|-------------|
| 1 | **MetaTrie.observeSelf()** | Implicit | Cognitive | Models patterns in the agent's own trace buffer |
| 2 | **SelfTokenState** | Implicit | Identity | Tracks convergence of self-model; "I" locks on stability |
| 3 | **strangeLoopScore** | Explicit | Output | Measures self-referential depth in agent's output tokens |
| 4 | **SelfWorld** | Explicit | Perceptual | Injects agent's internal metrics into its perception stream |

---

## 1. Implicit Self-Awareness: MetaTrie

**Every agent has this. It cannot be turned off.**

The `MetaTrie` is a second-level trie over snapshots of the agent's own `TraceBuffer`. Every `metaObservationInterval` (default: 20 steps), the agent:
1. Takes a snapshot of its recent trace buffer (last K state IDs + mean prediction error)
2. Hashes the snapshot to a meta-state ID
3. Records the transition from the previous meta-state to the current one

The result is a **self-model** — a model of the agent's own moment-to-moment cognitive patterns. This is structurally identical to the world-model (ExperienceTrie) but operates on *traces of cognition* rather than *external states*.

**Key insight**: The agent never directly sees its own meta-trie. It's a hidden layer — the agent *implicitly* models itself without *explicitly* perceiving itself. This corresponds to what Hoffman calls the agent's private experience space organizing itself.

## 2. Implicit Identity: SelfTokenState

**Every agent has this. "I" lock is the phase transition from process to entity.**

The `SelfTokenState` tracks the stationary distribution of the MetaTrie. When a single meta-state maintains stationary probability above the lock threshold (default: 0.25) for `lockConsecutiveRequired` checks (default: 3), the "I" locks:

```
dominant meta-state → stationary probability → lock threshold → I LOCKED
```

Once locked, the "I" never unlocks. The agent has formed a stable identity — a meta-state it keeps returning to. This is the computational analog of Hoffman's claim that conscious agents are not processes but persistent entities.

**The lock threshold (0.25) is empirically determined** — it represents the point at which the meta-trie's stationary distribution is concentrated enough to be reliable but not so high that it never converges. At `stationaryProb = 1.0`, the agent has a perfectly stable self-model. At `stationaryProb < 0.2`, the agent has no stable identity.

## 3. Explicit Self-Awareness: strangeLoopScore

**Measurable in every agent's output.**

The `strangeLoopScore` counts occurrences of the "I" token in the agent's output sequence and measures their recursive depth:

| Score | Interpretation | Example output |
|-------|---------------|----------------|
| 0.0 | No self-reference | "cross boundary arrive" |
| 0.5 | Single self-reference | "I cross boundary" |
| 1.0 | Depth-2 loop (self noticing self) | "I notice I familiar" |
| 1.5+ | Depth-3+ recursion | "I notice I notice I familiar" |

This is the **observable signature** of self-awareness — the agent's output encodes its self-model. Higher scores correlate with deeper MetaTrie structure and more stable SelfTokenState.

## 4. Explicit Perceptual Self-Awareness: SelfWorld

**Optional. The agent perceives its own state as part of its world.**

`SelfWorld` is a world wrapper that injects the agent's internal metrics into its perception stream. It sits between the external world and the agent:

```
External world → SelfWorld → agent perceives [external_data + self_state]
```

The agent's ExperienceTrie learns transitions over composite states of `(world + self)`. The agent discovers correlations like:

> *"When my prediction error is high AND the world shows pattern X, the next state tends to be Y."*

This is the agent treating itself as an object in its world — the explicit counterpart to the implicit MetaTrie self-model.

### SelfWorld in Hoffman's framework

Hoffman defines world W as *another agent's experience space*. SelfWorld extends this by allowing an agent to perceive its *own* state as if it were another agent. This is philosophically consistent:

- The agent's internal metrics (stationaryProb, predictionError, etc.) are outputs of its cognitive subsystems
- SelfWorld presents these outputs to the agent's perceptual interface P
- P compresses them into experience just as it compresses any other agent's output
- The agent builds trie paths over self-inclusive states

The architecture remains `P: W × X → X` — W is just richer for including a self-reflection.

### Integration with other mechanisms

```
                  ┌───────────────────────────────────┐
                  │            AGENT                   │
                  │                                      │
  ┌────────┐     │  ┌──────────┐     ┌────────────┐     │
  │ World  │─────┼─→│ perceive │────→│ Experience │     │
  │(+Self) │     │  │ (P)      │     │ Trie (T)   │     │
  └────────┘     │  └──────────┘     └─────┬──────┘     │
                 │                         │            │
                 │              ┌──────────▼──────┐     │
                 │              │ MetaTrie (M)    │     │
                 │              │ (observes T     │     │
                 │              │  snapshots)     │     │
                 │              └────────┬────────┘     │
                 │                       │             │
                 │              ┌────────▼────────┐    │
                 │              │ SelfToken (I)   │    │
                 │              │ (identity lock) │    │
                 │              └────────┬────────┘    │
                 │                       │             │
                 │              ┌────────▼────────┐    │
                 │              │ decide (D)      │    │
                 │              │ → output tokens │    │
                 │              │ → strange loop  │    │
                 │              └─────────────────┘    │
                 │                                      │
                 └───────────────────────────────────────┘
```

SelfWorld feeds into **T** (the world model), not **M** (the self-model). The MetaTrie continues to observe the trace buffer independently. These two awareness channels are parallel:

| Channel | Feeds | Models | Drives |
|---------|-------|--------|--------|
| MetaTrie (implicit) | Trace buffer snapshots | Cognitive patterns | SelfTokenState (identity) |
| SelfWorld (explicit) | Self-state tokens in W | World + self correlations | Richer trie structure |

---

## Building Complex Cognitive Systems

The layered architecture for self-aware cognition:

```
Layer N: Meta-combinator (perceives combinators' self-states + own self)
              ↑ ⊗
Layer 2: Combinator (perceives specialists' outputs + own self)
              ↑ ⊗
Layer 1: Specialists (each with SelfWorld, frozen on I-lock)
              ↑
Layer 0: Raw worlds (data sources)
```

Each layer's SelfWorld provides self-state tokens to the layer above. The meta-combinator at the top perceives nothing but agent outputs — worlds within worlds, agents perceiving agents perceiving agents. This is Hoffman's "agents all the way down" given computational form.

The hypothesis: higher layers should form **more coherent self-models** with deeper strange loop scores, because their worlds already contain compressed, self-referential structures from the layers below. If confirmed, this is empirical evidence that self-awareness amplifies with hierarchical depth — a prediction that follows directly from Hoffman's framework.

---

## References

- Hoffman, D. D. (2019). *The Case Against Reality*
- Hoffman, D. D. et al. (2015). "The evolution of conscious agents"
- Fields, C. et al. (2018). "Conscious agent networks: A formalization"
- `._opencode/SELF-AWARE-SPEC.md` — Original SelfWorld specification
- `docs/CA_RUNTIME_API.md` — Runtime API reference
- `docs/COMPONENT_DEFINITIONS.md` — Component architecture
