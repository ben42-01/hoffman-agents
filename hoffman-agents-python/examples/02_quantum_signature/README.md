# Experiment 2: Quantum Signature? — Spectral Analysis of Combination

## Summary

Earlier versions of this experiment reported a "quantum-like" collapse of the spectral gap at combination levels 1–2 and a recovery at level 3. **That result was an artifact.** With the v3 math and classical controls there is no level-dependent signature: an agent's own meta-state chain has the same mixing speed and the same irreversibility at every combination level. Everything measured here is a classical Markov chain.

## Why the old result was wrong

| Old observation | Actual cause |
|---|---|
| gap = 0 at levels 1–2 ("collapse") | A combined agent's meta-trie holds its constituents' chains as **disconnected pieces**. A chain with more than one closed class has \|λ₂\| = 1, so the gap is 0 by definition. |
| gap = 1 or 0 for single agents | The example's private matrix code turned unobserved rows into **absorbing states** (the 2.x stationary-distribution bug). |
| agents "ripe" to combine | The 2.x "I" lock fired at step 61 **in any world**. In v3 none of these agents lock, so the old example never combines. |
| "gap ≈ 0 and detailed-balance error > 0.1 = quantum-like" | This criterion holds for a **classical clock** (a 10-state cycle moving forward with p = 0.95: gap 0.009, irreversibility 1.0). Slow mixing plus one-way circulation is ordinary classical behaviour. |

## What it does now

1. **Classical controls.** Gap and irreversibility for chains that are classical by construction: a clock, a reversible random walk, an i.i.d. chain, two disconnected chains, and an independent product.
2. **Agents.** Eight seeded agents are each isolated for 400 steps, then interact for 200 rounds and combine on a fixed schedule every 20 rounds. The schedule replaces lock gating because the lock never fires here. For each agent, the analysis covers its own recurrent meta-state chain (`metaKernel`): size, transitions, period, gap = 1 − |λ₂|, and irreversibility. Irreversibility is the normalised net probability flux: 0 means detailed balance holds, 1 means every edge is one-directional. The last column counts closed classes when inherited pieces are included, which is what 2.x effectively analysed.
3. **Tensor-product prediction.** At combination time `productKernel(A, B)` gives M_A ⊗ M_B, whose \|λ₂\| is max(\|λ₂(A)\|, \|λ₂(B)\|). It is compared with the chain the combined agent actually learns afterwards.
4. **Fusion.** `fuse()` of every combined agent is checked against snapshots of its constituents' chains taken at combination time.

Gaps are excluded from the level summary when a chain has too little data (fewer than max(20, 2·states) transitions, the same rule as the "I" lock) or is periodic (gap 0 by definition).

## Result (seeds 1–8; the Python version prints identical numbers)

| Level | Agents | Gap | Irreversibility |
|---|---|---|---|
| 0 | 8 | 0.119 | 0.932 |
| 1 | 4 | 0.105 | 0.944 |
| 2 | 2 | 0.100 | 0.931 |
| 3 | 1 | excluded: period 7 | — |

- **No collapse and no recovery.** Mixing speed and circulation are the same at every level.
- **Tensor-product prediction.** The learned joint chains do **not** follow the product-kernel prediction (mean \|Δgap\| ≈ 0.09). A combined agent is a new agent with its own experience in a shared world, not two independent chains run side by side. The quote from `FOR_DR_HOFFMAN.md` below is therefore not supported.
- **Fusion.** All 7 fusions restore their constituents exactly.

> *Historical note (2.x):* "This result I genuinely cannot explain. The combination operator was implemented as simple path union and averaging. It was not written as a tensor product. Yet the eigenvalue spectra converge toward what a tensor product would predict."

## What a genuine quantum test would need

A Markov chain only ever adds non-negative probabilities, so no spectral statistic of one can show quantum behaviour. A real test needs something a classical stochastic process provably cannot do, such as interference from complex amplitudes that can cancel (see experiment 09) or a violation of a Bell/CHSH inequality.

## Run

```bash
npm run examples:qs                                                  # Node
uv run python examples/02_quantum_signature/quantum_signature.py     # Python
```
