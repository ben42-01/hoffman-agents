# Experiment 2: Markov Structural Transition Under Combination

## What This Tests

When conscious agents interact and combine via ⊗, their meta-trie Markov chains undergo structural transitions — from simple absorbing paths to complex, irreversible dynamics. This experiment measures three signatures of this transition:

1. **Spectral gap** (1 − |λ₂|) — how fast the chain mixes. Gap ≈ 1 means near-uniform transitions (fast mixing). Gap ≈ 0 means near-reducible/cyclic structure (slow mixing).
2. **Entropy production rate** (EPR) — how irreversible the dynamics are. EPR > 0 means probability flows in a preferred direction.
3. **Mixing time** (τ = −1/log(|λ₂|)) — how many steps needed to reach stationarity.

The transition from fast-mixing classical dynamics (Phase 1) to slow-mixing, high-irreversibility dynamics (Phase 3+) reflects the emergence of structure in the agent's self-model as it integrates experiences from other agents.

## How It Works

### Phase 1: Isolated agents
- 8 base agents each walk a deterministic Markov world for 400 steps
- Meta-states computed from coarse fingerprints (error bucket + ergodic state + I-lock status + 2-state mod-8 pattern)
- Result: absorbing path chain → gap ≈ 1.0, EPR ≈ 0

### Phase 2: Interaction
- Agents observe each other's output tokens for 40 rounds
- Perturbation begins to break the simple chain structure
- Result: some agents show gap collapse, EPR rises

### Phase 3: Combination
- Every 20 rounds, ripe agents combine via ⊗
- Combination preserves both parents' meta-trie transitions (bit-shifted) and trace buffer
- Combined agents inherit rich self-models from birth
- Result: gap continues to drop, EPR rises further

### Phase 4: Fusion
- The highest-level combined agent is fused via ⊘ back into constituents
- Validates that the combination is reversible at the meta-structure level

## No Quantum Claims

This experiment does **not** claim to detect quantum signatures. The spectral gap collapse and entropy production are classical Markov chain phenomena. They measure structural change in the agent's self-model — from simple to complex — as a function of interaction and combination depth.

## Key Metrics

| Metric | What it measures | Range | Phase 1 | Phase 3+ |
|--------|-----------------|-------|---------|----------|
| Spectral gap (1−|λ₂|) | Mixing speed | [0, 1] | ~1.0 (fast) | ~0.0 (slow) |
| Mixing time (τ) | Steps to stationarity | [0, ∞) | ~0 | > 100 |
| Entropy production | Irreversibility | [0, ∞) | ~0 | > 0.5 |

## Files

| File | Purpose |
|------|---------|
| `markov_transition.js` | Main experiment (Phases 1-4) |
| `kronecker_test.js` | Standalone Kronecker product distance test (separate, not part of main workflow) |
