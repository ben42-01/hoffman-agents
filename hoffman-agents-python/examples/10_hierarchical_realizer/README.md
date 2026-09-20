# Hierarchical Realizer — Emergent Pattern Discovery

## The Core Idea

This is not about training classifiers on agent features. It's about something fundamentally different:

**At any level of a hierarchy of Markovian agents, the collective dynamics of the population — their prediction errors, their mutual surprise, their response patterns — form a structured, high-dimensional geometric object. A realizing observer at the next level up can RECOGNIZE structure in that geometry, LOCK the discovered pattern, and CRYSTALLIZE it into a new instantiated agent at the higher level. That agent carries forward the discovered structure as its own experience.**

This process recurses up the hierarchy. At every level, agents produce collective dynamics; a higher-level observer detects geometric pattern; the pattern gets locked into a new agent. This is emergent intelligence in the pure sense: structure is discovered, not designed; labeled, not taught; grown, not programmed.

## The Architecture

```
Level 7: Meta-cognitive realizer — recognizes patterns across crystallized agents from Level 6
Level 6: Higher-order realizer — recognizes structure in collective dynamics of Level 5 agents
Level 5: Realizer agents — each is a crystallized cluster of Level 4 agents, augmented with a neural network
Level 4: Markovian intuition engines — the current conscious-agent library (trie, meta-trie, self-token)
Level 3-0: Raw data processing layers (not yet modeled)
```

### Key insight: preservation of individual perspective

A naive approach merges all constituents' experiences into one smooth pool (trie union, lexicon merge). This **washes away the discriminative signal**. Experiment 08 showed this empirically: identical twins merge silently, but a veteran and a novice who share a world produce a real, measurable surprise asymmetry when each is queried from the other's perspective.

The correct architecture preserves **each constituent's individual perspective** and reads **across** them. The higher-level agent doesn't merge its constituents into one — it *observes* them as a population, detects structure in their collective response, and crystallizes that structure into a new form.

## The Mechanism (as prototyped)

### Step 1 — Train a diverse population

N agents, each processing the same world but with genuinely different "sensory access" — different channels, different viewports, different signal-to-noise — develop distinct but overlapping internal models. Each agent's trie captures the transition structure of its own partial view.

### Step 2 — Observe collective response

Present all agents with the same test sequence. Record their prediction error (PE) trajectories over time. The result is an N×T response matrix R where each row is one agent's sequence of prediction errors.

### Step 3 — Build the similarity manifold

Compute pairwise similarity between agents: `S[i,j] = 1 − normalized_distance(R[i], R[j])`. Agents with similar channel coverage will respond similarly to the same input. The resulting N×N similarity matrix S encodes the latent structure of the agent population's collective understanding.

### Step 4 — Detect geometric structure

Apply spectral decomposition to the Laplacian of S. The low-frequency eigenvectors (especially the Fiedler vector — the second-smallest eigenvalue) reveal clusters in the similarity manifold. These clusters correspond to groups of agents that "see the world the same way" — not by design, but by emergent convergence of their internal models.

### Step 5 — Crystallize

Each detected cluster is combined (via `combine()`) into a single higher-level agent. This agent inherits the merged experience of the cluster's members, but more importantly, it is the **product of recognizing structure** in the lower level's collective dynamics. It carries not just knowledge, but the geometric signature of how that knowledge was discovered.

### Step 6 — Recurse

The crystallized agents themselves form a new population at Level 5. Their collective dynamics — how they respond to new inputs — can be observed, structured, and crystallized again at Level 6. Repeat.

## What the Prototype (experiment 10) Actually Demonstrated

**What worked:**
- The spectral clustering pipeline (similarity matrix → Laplacian → eigendecomposition → k-means on eigenvectors) successfully detected structure in synthetic data.
- The concept of "crystallizing" detected clusters via `combine()` is structurally sound and preserves the discovered grouping.
- The geometric approach requires no labels — structure is discovered from the population's own dynamics.

**What failed (and why):**
- The multi-channel world tokenized all channels into a single string — every agent saw the *same* token regardless of which channels they were "assigned." There was no actual difference in sensory access. The PE-response matrix was uniform (all 9s), not structured.
- Running-average PE (a windowed mean of prediction error) is too slow and too smooth. The instantaneous per-step prediction (correct/incorrect) would be a much sharper, faster signal.
- A 20-agent population over 60 test steps is too small for robust spectral geometry. Real emergent structure requires larger populations (100-1000+ agents) and longer observation windows.

**What a proper implementation needs:**

| Component | Requirement |
|---|---|
| World model | Genuinely different sensory channels per agent — not tokens. Each agent receives only its assigned channels. The world must be a high-dimensional signal that individual agents can only partially perceive. |
| Response metric | Per-step prediction success (binary: did the agent predict the next hashed state correctly?), not running-average PE. This gives a fast, sharp signal that instantly reflects which channel the agent understands. |
| Population size | 100-1000+ agents minimum. Spectral geometry on small populations (<50) produces unreliable clusters. |
| Observation window | Long enough for the population to traverse multiple "regime" transitions in the world — hundreds to thousands of steps. |
| Realizer network | A neural network that takes the spectral embedding of the similarity manifold as input and produces a "crystallization decision" — which agents to combine, and with what weight. This is the learned component that replaces the heuristic k-means step. |

## Connection to the Framework

### Hoffman's 6-tuple and the hierarchy

In Hoffman's formalism, each conscious agent is defined by (X, G, P, D, A, N): experience space, action space, and perception, decision and action kernels acting on a world W, plus a step counter. The hierarchy of agents (Level 0 → Level N) is not just a stack of increasingly complex agents; it's a **ladder of observation**, where each level's function is to *perceive the level below* and *act on the level above*.

The realizer mechanism is exactly this: a Level N agent's perception map reads the collective dynamics of Level N-1 agents. Its decision map produces crystallization actions (combine, split, ignore). Its world is the population of lower-level agents. This is a direct computational instantiation of Hoffman's formalism, applied recursively.

### Surprise as a detector

Prediction error works as a surprise signal: regions where an agent's learned model repeatedly fails mark structure the model does not capture. The hierarchical realizer extends this from a single agent observing data to a population of agents observing each other. The "holes" detected at Level 4 (structural absences in data) become the patterned substrate that Level 5 observes. The same mechanism — spectral geometry of surprise — operates at every level, just at different scales.

### The missing ingredient: refusal / exclusion (speculative)

The terms "boson-like" and "fermion-like" below are loose analogies for symmetric vs exclusive combination, not physical claims.

Experiment 07 showed that `combine()` is exactly exchange-symmetric (by construction), so it cannot produce exclusion-like behaviour. Experiment 08 showed that replacing pure union with an act of perception (observer-gated combination) can introduce principled asymmetry, but only in structured relationships (veteran/novice on the same world), not for identical or totally alien agents.

A speculative reading: exclusion would not be an incidental property of combination, but a separate, observer-dependent phenomenon that must be *recognized* and *instantiated* by a higher-level observer, not merely *discovered* in the lower level's dynamics. The hierarchical realizer is the natural architecture for this recognition-and-instantiation: a Level 5 observer detects that certain Level 4 agents' internal models are *incompatible* (high mutual surprise, cannot merge), and instantiates that incompatibility as a structural constraint — a refusal — rather than a smooth union.

## Open Questions (for the next session)

1. **What is the actual "neural network pattern" that a realizer recognizes?** The spectral embedding of the similarity manifold is a geometric object — is a neural network pattern just a geometric object that happens to have the right structure for learning? Or is the recognition of "this looks like a neural network's activation pattern" a genuinely distinct capability?

2. **How does the realizer learn to recognize structure without labels?** In our prototype, we used k-means on eigenvectors, which is unsupervised but heuristic. A proper realizer would use something like a variational autoencoder on the population's response matrix — detecting latent structure without any explicit clustering step.

3. **What stops the hierarchy from collapsing into one all-knowing agent at the top?** If every level crystallizes everything into one monolithic agent, the hierarchy flattens. Something must preserve multiplicity — some principle that says "these two clusters are genuinely different and should remain separate." This may be the same principle as fermion exclusion, just operating at the population level.

4. **Does the spectral gap of the similarity manifold itself carry meaning?** Experiment 09 showed that cyclic Markov chains have complex eigenvalues with real oscillation periods. The same family of math (spectral decomposition of transition matrices) applies to the agent population's response manifold. Does the eigengap of the similarity matrix encode something about the "temperature" or "coherence" of the population's shared understanding?

## Relationship to Other Experiments in This Repository

| Experiment | What it Found | Connection to Realizer |
|---|---|---|
| 07 exchange_symmetry | combine() is exactly exchange-symmetric (by construction) | Exclusion-like behaviour would need a separate, higher-level mechanism |
| 08 observer_gated | Perception-based asymmetry emerges from structured relationships | The "observer" component that mediates combination is the same mechanism a realizer uses to read across a population |
| 09 double_slit | Complex eigenvalues in cyclic Markov chains produce oscillatory dynamics | The spectral geometry of the similarity manifold is the same mathematical family — complex eigenvalues at the population level |
| 10 (this) | Spectral clustering can detect structure in collective agent dynamics | First directional prototype — needs proper world model and larger populations to be conclusive |

## Running the Prototype

```bash
uv run python examples/10_hierarchical_realizer/hierarchical_realizer.py
```

The prototype demonstrates the spectral clustering pipeline but is limited by the homogeneous world model (all agents see the same full token). A proper implementation requires a world where agents have genuinely different sensory channels.
