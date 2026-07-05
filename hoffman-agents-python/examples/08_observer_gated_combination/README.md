# Observer-Gated Combination

Follow-up to [`07_exchange_symmetry`](../07_exchange_symmetry/README.md). That experiment proved `combine()` is *exactly* exchange-symmetric — a direct, unavoidable consequence of being built from pure union operations (trie merge, lexicon merge). Union has no vantage point. It cannot produce asymmetry, no matter how it's composed or how many trials you run.

This experiment asks a different question: what if, instead of merging blindly, each agent first **perceives** the other before combination proceeds?

## Why perception, specifically

`perceive()` is the one operation in this architecture that is inherently asymmetric. Prediction error is always computed *from* one agent's own history, relative to what it alone has learned to expect. It has a vantage point — an observer. Two different agents perceiving the same thing will, in general, report different surprise, because they carry different pasts. Union can never do this. Perception always does.

## The mechanism

Before merging agents A and B, each perceives the other's recent trajectory as if it were a world to predict:

```javascript
const surpriseAonB = observerSurprise(A, B);  // how surprised is A, watching B?
const surpriseBonA = observerSurprise(B, A);  // how surprised is B, watching A?
```

- **Mutual fit** (both surprises low): merge proceeds as ordinary union — boson-like.
- **A rupture detected** (either surprise high): the *less surprised* — more confident — agent anchors the merge; the other yields. This is a principled asymmetry, not an arbitrary one: it's driven by an actual epistemic relationship between the two agents, not by which one happened to be passed as the first argument.

## Three regimes tested

### Phase A — Identical twins (sanity check)
Two agents with literally identical experience (one cloned from the other). Expectation: mutual surprise should be equal, since each would predict the other exactly as well as it predicts itself — merge should stay symmetric, consistent with `07_exchange_symmetry`.

**Result: confirmed.** Asymmetry = 0.0000 across 20 trials in both languages.

### Phase B — Totally disjoint agents
Two agents trained on completely unrelated worlds (different state spaces entirely, zero overlap). Neither has ever seen anything resembling the other's experience.

**Result: symmetric, but for a different reason.** Both surprises saturate at maximum (≈1.0) — not because the agents fit each other, but because *neither can understand the other at all*. This is **symmetric confusion**, not symmetric fit, and it's not meaningfully exchange-antisymmetric: when neither party has any basis for comparison, "who understands whom better" is a degenerate question with no real answer.

### Phase C — Veteran vs. Novice (same world, unequal depth)
Two agents trained on the *same* world (same seed, same structure), but for very different lengths of time — a Veteran (2000 steps) and a Novice (40 steps). Both have real, overlapping experience, but at very different depths.

**Result: real, principled asymmetry emerged.**

| Metric | Node | Python |
|---|---|---|
| Mean surprise asymmetry | 0.1017 | 0.1086 |
| Max observed asymmetry | 0.2759 | 0.2759 |
| Dominance order-independent | 18/20 | 19/20 |
| Veteran dominates | 15/20 | 15/20 |

The Veteran — having seen far more of the shared world — predicts the Novice's narrower experience well. The Novice, having seen only a sliver of the world, is frequently surprised by patterns the Veteran has already encountered. This produces a real, nonzero surprise gap, and critically: **which agent ends up dominant is determined by the relationship between them (roughly 90-95% of the time), not by which one happens to be passed as the first function argument.** That is the signature of a principled asymmetry rather than an implementation artifact — the same distinction that separates a genuine physical effect from a coordinate-system accident.

## What this means

Three distinct regimes, cleanly separated:

1. **Identical → symmetric fit.** Nothing to distinguish, nothing distinguishes them.
2. **Totally alien → symmetric confusion.** Nothing to compare, so no meaningful asymmetry can exist.
3. **Related but unequal → real asymmetry.** A genuine epistemic gap between two agents who share a world but not a depth of understanding produces measurable, mostly order-independent dominance.

This maps loosely (as a computational analogy, not a physics claim) onto why real exchange statistics require *identical* particles to even be asked about: bosons and fermions are only meaningfully distinguished when you're comparing truly indistinguishable things. Once agents differ meaningfully in what they know, the question shifts from "is this symmetric or antisymmetric" to something else entirely — a relationship, a hierarchy, an act of observation with a genuine point of view.

### A note on what kind of effect this is

`observerGatedCombine()` never changes its own logic between the three phases above — the identical function runs every time. What changes is only the *correlation structure* between the two agents' histories (identical / totally unrelated / overlapping-but-unequal), and that alone is enough to shift the observed outcome from symmetric fit, to symmetric confusion, to real principled asymmetry. One fixed rule, different relationships, different effective statistics. See [`09_double_slit_analogy`](../09_double_slit_analogy/README.md) for why this is the same shape of explanation as decoherence in quantum mechanics — where a single consistent rule, applied to a system with different correlation/entanglement structure, produces interference or its absence, without the underlying rule ever changing.

## Caveats

1. **This is not a physics experiment.** It's an information-theoretic analogy inspired by the idea that exchange statistics might require an observer-relative (not purely structural) mechanism. Results are directional signals for further formal investigation, not physics claims.

2. **Order-independence is strong but not perfect** (18-19 out of 20, not 20/20). This is reported honestly rather than rounded up — the residual noise likely comes from the small sampling window (last 30 trace events) used to estimate surprise, which can occasionally produce near-ties that tip either direction. A larger window or more training steps would likely tighten this further; that's a natural next refinement, not a hidden problem.

3. **`observerGatedCombine()` is exploratory example code**, kept separate from the core library's `combine()`. It reuses `combine()` internally (for the actual merge mechanics) but adds the perception-gating layer on top, so it doesn't touch or risk any validated core behavior.

4. **The "hole threshold" (0.5) is a hand-picked cutoff**, not derived from first principles. Different thresholds would shift exactly where the "union" vs "exclusion" boundary falls; this parameter itself would be a reasonable target for the next round of experiments.

## Running

```bash
# Node
npm run examples:observerGatedCombination

# Python
uv run python examples/08_observer_gated_combination/observer_gated_combination.py
```
