## v3.0.0 — "Ergodic"

A release about the mathematics underneath the metaphors. The stationary distribution, the "I" lock, the output dynamics and the combination operator have all been rebuilt on explicit Markov kernels, and they are now tested against analytic answers. The full specification is in `docs/MATHEMATICAL_MODEL.md`.

**This is a breaking behavioural change.** Agents no longer lock "I" in every world at step 61. To reproduce 2.x results exactly, pass `mathVersion: 'legacy'` (Node) or `math_version="legacy"` (Python). `.soul` files saved by 2.x load as legacy automatically.

### Math fixes

**The newest meta-state was an absorbing sink.** `stationaryDistribution()` added the most recent meta-state to the chain. It had no observed outgoing transitions, so it became a self-loop and absorbed all stationary mass. For the chain 1→2→3 the result was π = (0, 0, 1). Every agent in every world, including pure noise, reported a "dominant attractor" and locked "I" at step 61, the earliest possible moment. Unobserved rows are now pruned, and the analysis runs on the recurrent class the agent actually occupies.

**Self-transitions were never recorded.** Staying in the same experiential state is exactly what an attractor looks like, and it was discarded. It now counts.

**Power iteration failed on periodic chains.** A 2-cycle returned (1/3, 2/3) instead of (½, ½). Iteration now runs on the lazy chain (P + I)/2, which has the same π and always converges, with a direct linear solve as fallback. Period, |λ₂|, relaxation time and mixing time are now computed as well.

**The "I" lock was trivial.** "max π > 0.25" holds automatically for any chain with at most 4 states, and a lock could never be undone. The v3 lock needs sufficient evidence, an aperiodic converged chain, a dominant state well above the uniform baseline, occupancy of the attractor, and a stable referent. It unlocks with hysteresis. Measured results: 0/50 false locks in structureless worlds, 20/20 locks on real attractors, and unlocks after a regime change.

**The lock flag was hashed into meta-state ids**, which split the chain into disconnected pre-lock and post-lock components. Removed.

**The output-mode chain wasn't a validated kernel.** `pStable + pLexicon + pExplore` could exceed 1. It is now an explicit, validated `MarkovKernel` D with exact diagnostics (`agent.decisionKernel`).

**Combination was order-dependent and broke under nesting.** `A ⊗ B` and `B ⊗ A` had different ids and different short-term memory. `(A ⊗ B) ⊗ C` reused the same 4-bit tags, so fusing it split the wrong states. `fuse()` gave both halves the *same* lexicon object. Combination now has commutative, associative identities, collision-free provenance at any nesting depth, and exact restoration on `fuse()`. It also exposes the Hoffman–Prakash product kernel M_A ⊗ M_B as a prior.

**Prediction error was 0/1.** It is now 1 − p(actual | previous) under a Witten–Bell estimate (graded, scale-free), with surprisal recorded on each trace event.

**Node and Python disagreed, and Python wasn't reproducible.** Python used the per-process salted `hash()` and 64-bit state ids. Both libraries now share mulberry32, FNV-1a and 32-bit state ids, and a parity test requires identical dynamics step for step.

### New API

- `MarkovKernel`, `StochasticMatrix`: validate, sample, compose, tensor, mix, stationary, diagnostics.
- `markov` utilities: stationary, communicating/closed classes, period, ergodicity, |λ₂|, mixing time, entropy, KL, TV, Kronecker product.
- `FormalConsciousAgent`: a finite (X, G, P, D, A, N) reference agent with the joint kernel Q on X × W and product combination.
- `agent.ergodicStats()` / `ergodic_stats()`, `agent.toFormal()` / `to_formal()`, `agent.decisionKernel` / `decision_kernel`.
- `metaTrie.ergodicDiagnostics()`, `productKernel(a, b)`, `experienceSpaceDistance(..., { mode: 'kernel' })`.
- New `seed` agent option (mulberry32), plus `lexiconRow`, `mathVersion`, and lock options (`minTransitions`, `lockMargin`, `unlockMargin`, `minOccupancy`, …) through the constructor and `fromConfig`.
- Lock and unlock events on `StepOutput.interrupt` and in `selfToken.lockHistory`.
- Python: `combine(a, b, weights=(w1, w2))`; `fuse` is now exported at the top level.

### Other changes

- `.soul` format version 3 stores the math version, decision parameters, meta-chain history, provenance and lock history.
- `clone()` no longer writes temporary files (Node).
- `ExperienceLexicon.bind()` replaces the signature of an existing label instead of duplicating it.
- Examples: new `12_ergodic_diagnostics`. `02_quantum_signature` is rewritten: the 2.x "quantum-like" gap collapse was an artifact (disconnected constituent chains, absorbing states, and a criterion that a classical clock satisfies). The experiment now measures each agent's own chain against classical controls and tests the tensor-product prediction directly. `05_self_ref_ablation` now ablates the v3 `lockMargin`; the Python summary also showed the wrong "Agents locked" count for the OFF condition, which is fixed.

### New experiments

- **13 Bell test through the headset.** Grant a network of conscious agents behind spacetime, and let an observer's action reach it. For this model class the best CHSH value is at most **2 + 2·δ(Qᵏ)** (proved in the example README), where δ is the Dobrushin contraction coefficient of k network steps. The optimiser attains the bound to machine precision in every tested case. Correlations reach the PR box (4) at zero network distance and fall below Tsirelson's bound once δ ≤ √2 − 1. The CHSH-optimal behaviours stay outside the quantum set until they become classical. Without an explicit no-signalling rule, 98% of random networks would signal. The formalism can produce non-local correlations but does not by itself single out the quantum ones.
- **14 Spacetime in the headset.** An agent that sees only opaque symbols recovers its hidden world's spectral dimension from experience (within 0.03). Under ⊗, dimension adds exactly (1 → 2 → 3). Interaction binds dimensions (2.8 → 1.1 as coupling grows) and creates an arrow of time (irreversibility 0 → 0.41). Geometry in the headset is inherited from agent dynamics, not generated.
- **15 Time in the traces.** Implements Hoffman's trace chains (`trace.traceChain`, `MarkovKernel.trace`). An observer restricted to a window experiences exactly the trace chain (simulation vs formula within 0.006). Each observer's clock runs at π(S) ticks per network step (Kac, to machine precision), and nested clocks compose exactly. The arrow of time is observer-dependent: with reversible dynamics every observer's uncertainty still grows, and a two-state window never perceives irreversibility.
- **16 Trace logic and decorated permutations.** Implements the decorated-permutation map of *Fusions of Consciousness* (Def. 2).
  - The map agrees with all 27 entries of the paper's Appendix B table. The table holds 14 distinct permutations for M3; the text says 17.
  - It depends only on class structure and the cyclic order of the numbering, and it discards probabilities.
  - Trace logic behaves as the papers claim: locally Boolean, globally not a lattice, and order-preserving into Lebesgue logic via π.
  - The mass ↔ entropy-rate and speed ↔ commute-time proposals pass additivity and metric checks.
- **17 Quantum agents.** Agents whose kernels are quantum channels on qubits.
  - Combined through an interaction, they reach Tsirelson's bound 2√2 exactly and never exceed it (2000 random interactions).
  - No-signalling holds automatically.
  - Markov agents embed as fully decohered quantum agents: dephasing brings CHSH down to exactly 2, and interference disappears (Sorkin I₂ → 0, I₃ = 0 throughout).
  - The quantum structure is assumed, not derived.
- **18 Relativity at infinity.** Takes Hoffman's n → ∞ conjecture to a 1+1-dimensional agent with one bit of memory (its direction), computed exactly for n up to 1280.
  - A light cone survives the limit; a memoryless agent's speed diverges.
  - The agent's clock (its count of its own changes of experience) converges to Einstein's proper time, with errors ∝ 1/n.
  - The classical dynamics keeps a preferred frame; the quantum walk tends to the Lorentz-covariant Dirac propagator.
  - This is known mathematics (Kac; Feynman's checkerboard) read as agents.
- **05 Self-reference ablation** now includes the informative control: the lock stays off in a structureless world. The old "self-reference is causally necessary" conclusion was true by construction.
- **02 Quantum signature?** has been rewritten with classical controls.

### Library additions (both languages)

- `markov.irreversibility`, `markov.dobrushin`, `markov.spectralDimension` / `spectral_dimension`, and `markov.returnProbabilities`, which can average over the stationary distribution for chains with drift.
- `analysis/decorated` (decorated permutations); `trace.qualiaKernel`, `lebesgueLeq`, `stationaryMeasure`; `markov.entropyRate`, `hittingTimes`, `commuteTimes`.
- `quantum` (Node `src/quantum/qubits.js`, Python `conscious_agent.quantum`): density matrices, Kraus channels, the Markov-kernel embedding, and the exact maximum CHSH with optimal settings (Horodecki).
- `analysis/bell`: CHSH, the Tsirelson–Landau–Masanes quantum-set test, signalling, and classification.
- Node `package.json` `exports`: `conscious-agent/worlds` and `conscious-agent/io`, which the README documented but which did not resolve, now work. `conscious-agent/bell`, `conscious-agent/markov`, `conscious-agent/trace` and `conscious-agent/quantum` are also exposed.

### Documentation and site

- Claims that did not hold up have been withdrawn and listed in `docs/Q_AND_A.md#corrections`. Examples: the "quantum algebra convergence", "self-reference is causally necessary", "lock time is exactly 60 steps", the Planck-probe lock rate, the qubit estimate, and an unverifiable "CERN / Standard Model" anecdote in example 10.
- `Q_AND_A.md` is rewritten around the current code. `SELF_AWARENESS.md` now describes self-*modelling*. The theory document is marked as the pre-implementation design document, with its six-tuple definitions corrected. The API document is marked as a design specification. The visual guide's code map now points to real files. The glossary is corrected.
- The GitHub Pages site replaces the simulated "Server Monitor" and "Agent Network" animations with **How an Agent Works**: a step-by-step, exact walk through one perceive–decide–act cycle, the joint kernel Q, the convergence of time averages to π, and combination as a Kronecker product. Its numbers match the library.

### Tests

- Node: 141 tests. Python: 103 tests. Both suites cover analytic Markov results, the ergodic theorem on formal agents, I-lock null/positive/unlock tests, combination algebra and fusion, byte-exact 2.1.2 golden fixtures, and the Node ↔ Python parity run.

### Known limitations

Transition counts never decay, so unlocking after a regime change is slow. Meta-states are coarse. `loopDepth` reflects time spent in the `core` output mode, not emergent self-reference. See `docs/MATHEMATICAL_MODEL.md` §15.
