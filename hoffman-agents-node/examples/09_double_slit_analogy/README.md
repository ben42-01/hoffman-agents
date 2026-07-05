# Double-Slit Analogy

Can the double-slit experiment's famous interference pattern be reproduced using this library's Markov-chain / trie machinery? Tested directly rather than just asserted.

**Short answer: no, not without adding an ingredient the library doesn't have — and demonstrating *why* is itself the most useful thing this experiment produces.**

## The setup

A "diamond" world: a Source state branches 50/50 into Path A (length 3) or Path B (length 5), and both reconverge at a shared Detector. This is a discrete-time stand-in for "two slits leading to one screen."

## Part 1 — Tracked vs. erased (classical result)

Two conditions:
- **Tracked**: record which path was taken alongside arrival time; histogram each path separately, then sum.
- **Erased**: record *only* arrival time, never which path; pool everything into one histogram.

**Result: identical, down to the exact count.** 2472 vs. 2472 at step 3, 2528 vs. 2528 at step 5 (Python run) — because they are mathematically the same computation performed two different ways. Whether or not you *track* which path was taken cannot change the outcome distribution, ever, in ordinary probability theory.

This is not a limitation of this library specifically — it is a hard fact about classical probability itself, and it's exactly why real double-slit interference is considered one of the deepest pieces of evidence that nature does not run on classical probability alone. If nature *did*, "erasing which-path information" could never change anything, the same way it can't here.

## Part 2 — What genuine interference would require

Using the same two path lengths, a small explicit construction: assign each path a complex amplitude with phase proportional to its length, sweep a toy "wavelength" parameter, and compare:

- **Classical**: `P = 0.5 + 0.5 = 1.0`, flat, for every value of the swept parameter. Nothing to oscillate.
- **Amplitude-based**: `P = |amplitude_A + amplitude_B|²`, which ranges from **0.000 to 2.000** across the same sweep — a genuine fringe pattern.

The entire difference between these two calculations is **one line**: whether you square before or after summing. Square-then-sum (probabilities) can never produce interference. Sum-then-square (amplitudes) always can, because the cross term `2·Re(amplitude_A · amplitude_B*)` — which only exists when you sum before squaring — is exactly the interference term, and it can be negative (destructive) or positive (constructive) depending on relative phase.

Our trie/Markov machinery only ever does the former. Every prediction, every merge, every stationary distribution in this library is built from non-negative counts and probabilities. There is no phase anywhere in it. That's not a bug — it's a structural boundary, and now it's a demonstrated one rather than an assumed one.

## Bonus — do we already have a small piece of this?

Yes, partially. The diamond world's own transition matrix (the exact kind of cyclic, branching structure `WorldBuilder` already produces) was checked directly for complex eigenvalues:

```
Transition matrix size: 10x10
Complex eigenvalues found: 8 / 10
  magnitude 0.9766, period 2.40 steps
  magnitude 0.9008, period 6.16 steps
  magnitude 0.8038, period 3.64 steps
```

Complex eigenvalues mean that repeated application of the transition matrix (`P^n`, i.e. how the chain evolves over many steps) contains genuinely oscillatory terms — `λ^n` rotates in the complex plane as `n` grows, the same mathematical family (complex exponentials) that underlies wave interference.

**This is a real, existing piece of wave-like mathematics already latent in ordinary cyclic Markov chains** — the same family of tools already touched by the spectral-gap analysis used elsewhere in this project. But it is a *different phenomenon* from double-slit interference: it describes how **one system evolves over time**, not how **two simultaneous paths interfere in space**. Related math, not the same thing.

## A sharper reframe: correlation, not a different algebra

An earlier draft of this analysis implicitly treated "classical" and "amplitude-based" as two different rulebooks that reality somehow chooses between depending on who's watching. That framing is looser than the actual physics, and worth correcting.

The precise picture (standard decoherence theory, not speculation) is: there is **one** rule throughout — amplitudes summed, then squared. What changes between "interference visible" and "interference gone" is not the rule, it's **what has become correlated with what**. When a measuring device records which path was taken, the particle becomes entangled with that device. Looking only at the particle's statistics afterward — ignoring the device — means looking at a *partial view* of a larger correlated system, and the interference cross-term washes out under that partial view, not because a different law took over.

This is a **stronger** statement than "different observers get different algebras," because it says something specific and testable: **a single, fixed rule, applied to a system that includes correlations with an observer, produces different effective/marginal statistics purely as a function of the correlation structure involved** — not because the rule changed.

That statement is exactly what [`08_observer_gated_combination`](../08_observer_gated_combination/README.md) already demonstrated computationally, without originally being framed this way. `observerGatedCombine()` never changes its logic between the twin case, the disjoint-worlds case, and the veteran/novice case — the identical function runs every time. What changes is the correlation structure between the two agents' histories (identical / totally unrelated / overlapping-but-unequal), and that alone is sufficient to produce symmetric fit, symmetric confusion, or real principled asymmetry. Same rule, different relationship, different observed statistics. That is the same shape of explanation as decoherence, applied to a completely different (and much simpler) system.

**An honest limitation this surfaces in Part 1 above**: the "erased" condition in this experiment was not a faithful analog of real quantum erasure. No correlation was ever created between the path taken and any other variable — the "tracked" and "erased" conditions differed only in whether a label was *written down*, not in whether any physical correlation existed to erase. Of course changing that made no difference; there was nothing there to change. A faithful analog would require introducing a genuine second, correlated "marker" variable — one that starts entangled with which-path information and can then actually be erased or randomized — and testing whether *that* changes the observed statistics. This experiment does not build that (yet); it's a natural, well-defined next step rather than a gap papered over.

## Honest summary

| Question | Answer |
|---|---|
| Can `combine()`/tries reproduce interference directly? | No — confirmed empirically, not assumed |
| Why not? | Everything in the library combines non-negative probabilities; interference requires signed/complex amplitudes summed before squaring |
| Is there a related piece of math already present? | Yes — cyclic Markov chains (already buildable) have complex eigenvalues, producing genuine temporal oscillation |
| Is that oscillation the same as double-slit interference? | No — same mathematical family (complex exponentials), different phenomenon (time evolution of one system vs. spatial interference of two paths) |
| Is "classical vs. quantum" really two different rulebooks? | No, more precisely: one rule, applied to systems with different correlation structures, produces different marginal statistics — see `08_observer_gated_combination` for a computational example of exactly this shape of effect |
| Does this experiment's "erased" condition model real quantum erasure? | No — no correlation was created in the first place, so there was nothing to erase. A faithful version needs a genuine correlated marker variable; that's a natural next step, not built here |

## Caveats

1. **This is not a physics experiment.** It's a deliberately constructed demonstration of a well-known mathematical fact (classical probability cannot produce interference), plus an honest inventory of what related wave-like math already exists in the codebase.
2. **The complex-amplitude construction in Part 2 is a toy add-on**, not part of the core library. It exists purely to make concrete what an "interference-capable" extension would need to look like, for comparison.
3. **No claim is made that adding complex amplitudes to `combine()`/tries would be meaningful or correct for this framework's actual purpose.** That would be a much larger design question, not something this script attempts to resolve.

## Running

```bash
# Node
npm run examples:doubleSlitAnalogy

# Python (uses numpy for the eigenvalue check)
uv run python examples/09_double_slit_analogy/double_slit_analogy.py
```
