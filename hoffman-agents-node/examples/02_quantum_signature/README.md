# Experiment 2: Quantum Algebra Convergence

## What This Tests

Hoffman's "Fusions of Consciousness" (Entropy, 2023) predicts that conscious agents combine into more complex agents and fuse into simpler agents, with Markov chains governing the dynamics. This experiment tests whether the implemented ⊗ operator produces eigenvalue spectra that shift from classical (deterministic, reversible) to quantum-like (stochastic, irreversible) as combination depth increases.

## How It Works

1. **Phase 1**: Isolated agents — each walks a deterministic world for 400 steps. Meta-states are computed from coarse-grained trace fingerprints (error buckets + ergodic state + last-2 state IDs modulo 8), making meta-states genuinely revisitable.
2. **Phase 2**: Agents interact for 40 rounds — each observes all others' output tokens.
3. **Phase 3**: Every 20 rounds, ripe agents combine via ⊗. Transition matrices from each agent's meta-trie are analyzed for spectral gap (1 − |λ₂|) and detailed balance error.
4. **Phase 4**: The highest-level combined agent is fused back into its constituents via ⊘, demonstrating invertibility.

## Key Changes from Original

- **Revisitable meta-states**: Coarse-grained fingerprint (error bucket × ergodic state × lock status × 2-state mod-8 pattern) replaces unique hash — enables genuine communicating classes in the Markov chain
- **Real spectral gap**: Uses power iteration with Hotelling deflation instead of row-dominance heuristic — computes actual eigenvalue gap matching the Python version
- **Combination preserves dynamics**: `_buildJointMetaTrie` transfers transition structure and `_lastMetaState` — combined agent inherits parents' self-model
- **Trace buffer transfer**: Combined agent inherits last 10 events from preferred parent for short-term memory continuity
- **Fusion operator (⊘)**: New inverse of combination — splits combined agent by meta-trie bitmask, recovers constituent agents with shared world-model and fresh identity

## Expected Result

| Level | Gap | DB Error | Interpretation |
|-------|-----|----------|---------------|
| Base (L0) | ~1.0 | ~0.0 | Classical — isolated, deterministic |
| Interacting | ~1.0 | ~0.0 | Still classical — interaction alone doesn't break reversibility |
| Level 1 | mixed | > 0.5 | Transitional — some agents collapse |
| Level 2+ | < 0.05 | > 0.95 | Quantum-like — stochastic, irreversible |
| Fused | mixed | > 0.5 | Fused agents retain quantum signature from shared history |

## Why This Matters (from FOR_DR_HOFFMAN.md)

> "This result I genuinely cannot explain. The combination operator was implemented as simple path union and averaging. It was not written as a tensor product. Yet the eigenvalue spectra converge toward what a tensor product would predict. This feels significant but I lack the mathematical training to understand why."
