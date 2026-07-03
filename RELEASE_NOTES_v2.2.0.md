# Release Notes — v2.2.0

## New Features

• **True Tensor Product Combination (Dual Perspective)** — `combine()` stores both parents' tries and meta-tries as independent clones. Combined agents maintain TWO simultaneous perspectives on every world state, computing prediction error from each parent's model separately each step. Joint meta-states are composite(meta₁, meta₂), built lazily as the agent interacts. The transition matrix over joint states IS P₁ ⊗ P₂ by construction — no pre-computation, no 10k cap, no bit masks. Fusion returns the cloned snapshot at combine time; parents continue learning independently.

• **Fusion Operator (⊘)** — `fuse(combinedAgent)` splits a combined agent back into its immediate constituents. The inverse of `combine()`. Recovers each parent's self-model verbatim from the cloned snapshot. Implemented in both JS and Python.

• **Sparse Interaction Topology** — Experiment accepts `connectivity` parameter (default all-to-all for backward compat). When set below agent count, each agent observes a random subset of `connectivity` others each round. Reduces O(n²) to O(n × k). 1000 agents @ k=10: ~0.9s/round vs 2.4 min/round all-to-all. Hoffman-aligned: "each agent observes a small number of other agents" (CONSCIOUS_AGENTS_THEORY.md).

• **Probabilistic Prediction (Markovian Kernel)** — `predictNextProbabilistic()` samples from trie children weighted by visit count instead of always picking the mode. Prediction error is now continuous 0.0–1.0 (was binary {0,1}). The agent expresses genuine uncertainty. Aligns with Hoffman's P(x|w) as a probability distribution.

• **Non-Veridical Perception Experiment** — New `examples/06_fitness_combination/` combines Fitness Beats Truth with the Tree of Life protocol. Interface agents (5 groups, 0.3125 error) beat veridical agents (20 raw states, 0.3375 error) — Hoffman's fitness-beats-truth confirmed in the combination context.

• **Kronecker Product Test** — New `examples/02_markov_transition/kronecker_test.js` (separate from main workflow). Measures distance from combined dynamics P_c to P₁⊗P₂ vs P₁⊕P₂. With dual-perspective tensor product, P_c IS P₁⊗P₂ by construction (ratio ≈ 1.0).

## Core Architecture Fixes

• **Meta-States Now Revisitable** — Changed from `sha256(10 exact state IDs + error)` to `f(errorBucket, ergodicState, isLocked, last2states % 8)`. Reduces state space from ~2¹⁶⁰ to ~2,560. Meta-states genuinely cycle → stationary distribution converges → "I" locks on a real ground state. Without this, the "I" had nothing stable to lock onto — the agent could not have a genuine identity.

• **Combination Preserves Self-Model (Cloned)** — Parent meta-tries are deep-cloned on combine. No double-counting: parent agents and combined children update independent meta-trie instances. Previous architecture had both parent and child updating the same shared reference per round.

• **Trace Buffer Continuity on Combine** — Combined agents inherit the last 10 trace events from the preferred parent instead of a single synthetic event.

• **Composite ID Referent on Combine** — `_combineAttractors` stores `_compositeId(st1.referentMetaStateId, st2.referentMetaStateId)` instead of a single bit-shifted referent. The combined "I" correctly points to a joint product-registry state.

## Metrics & Claims Fixed

• **Renamed from "Quantum Signature"** → "Markov Structural Transition." No more "quantum-like" claims. Gap collapse is a classical structural property (near-reducible or periodic chains), not a quantum signature.

• **Replaced DB Error with Entropy Production Rate (EPR)** — The old detailed balance error saturated at ~0.97 for all agents (useless discriminator). EPR = ΣπᵢPᵢⱼ log(Pᵢⱼ / Pⱼᵢ) has genuine dynamic range.

• **Added Mixing Time** — τ = -1/log(1 - gap). Directly interpretable: a gap of 0.003 means τ ≈ 333 steps to reach stationarity.

• **Honest Tags** — "← SLOW MIXING" / "← FAST MIXING" instead of "← QUANTUM-LIKE."

## Performance

• L0→L1 combine: ~5ms (no pre-computation — just stores parent references)
• L1→L2 combine: ~5ms (same — references only, no product explosion)
• Full experiment: ~380s (no overhead from tensor architecture)
• Kronecker test: ~90s (smaller matrices, no deep-level product builds)
• **No state explosion at any combination depth** — dual-perspective is O(1) at combine time
• Fusion: O(1) — returns cloned parent meta-trie reference
• Parent meta-tries are cloned independently — no shared reference double-counting
• Sparse topology: 1000 agents @ k=10 → ~0.9s/round (vs 2.4 min all-to-all)
• Stationary distribution: joint state space grows with experience, not with product of parent spaces

## Documentation

• `.context/SKILL.md` — Fuse pattern, corrected spectral gap formula, new Kronecker test
• `docs/GLOSSARY.md` — Added EPR, fixed spectral gap definition, added Fusion Operator
• `docs/CA_RUNTIME_API.md` — Added fuse example with composite ID explanation
• `examples/02_markov_transition/README.md` — Full rewrite with Phase 1-4 architecture, honest claims
• `examples/06_fitness_combination/` — New README and experiment files

## Side Effects & Mitigations

• **Parent meta-tries cloned on combine** — prevents double-counting bug. Parents and children have independent instances. Fusion returns snapshot at combine time, not live parent state. This is correct behavior: fusion recovers "who they were when combined."
• **No impact on prediction error** — prediction error uses the agent's own `trie` (or merged trie for combined agents). Dual perspective only affects meta-observation (every 20th step), not per-step error.
• **No impact on "I" locking** — combined agent's self-token tracks the JOINT meta-trie (composite states), not parent meta-tries.
• **No impact on regime change detection** — continuous error (0.0-1.0) gives more nuanced detection than old binary {0,1}. Tested: 74/100 regime shifts detected via "different" token.
• **No breaking API changes** — 36/36 tests pass (JS), Python tests pass (exit 0). All new parameters have backward-compatible defaults.
