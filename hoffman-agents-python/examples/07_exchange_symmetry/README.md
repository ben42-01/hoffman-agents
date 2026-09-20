# Exchange Symmetry Test

Tests whether the ⊗ combination operator produces **exchange-symmetric** (boson-like) or **exchange-antisymmetric** (fermion-like) results when its constituent agents are swapped.

## Background

In physics, identical particles under exchange exhibit either:
- **Symmetry** (bosons): swapping two particles leaves the system unchanged → they can occupy the same state (lasers, superfluids)
- **Antisymmetry** (fermions): swapping flips the sign → Pauli exclusion → they cannot occupy the same state (matter, atoms, quarks)

This experiment tests whether conscious agent combination produces analogous behavior in **information space** — not as a physics claim, but as an exploratory computational analogy that may surface directional hypotheses for formal investigation with frameworks like Hoffman & Prakash's Markovian kernel formalism.

## What It Measures

### Phase A — Exchange invariance

Creates identical twin agents (via `cloneAgent`), combines them as `combine(A, B)` and `combine(B, A)`, then measures whether observable quantities change under swapping:

| Observable | What it measures |
|---|---|
| **Spectral gap** (1−\|λ₂\|) | Mixing rate of meta-trie Markov chain — approaches 1 for deterministic, 0 for fully stochastic |
| **Detailed balance error** | Time-reversal symmetry breaking of meta-state transitions |
| **Loop score** | Self-referential depth in agent's output tokens |
| **Mean prediction error** | Surprise — how well the trie predicts the next state |
| **Experience space distance** | Jaccard-like divergence between AB and BA experience spaces |
| **Lock agreement** | Whether AB and BA agree on "I" lock status |

### Phase B — Occupation / redundancy

Combines an identical pair (A + its clone) vs a distinct pair (A + C trained on a different world), then measures the **redundancy ratio**:
```
redundancy = combined trie size / (parent1 trie size + parent2 trie size)
```

- **Ratio ≈ 0.5**: perfect overlap — identical twins produce no new distinguishable structure (exclusion-like)
- **Ratio ≈ 1.0**: additive — distinct experiences combine independently

## Results

| Observable | AB vs BA (Node) | AB vs BA (Python) | Interpretation |
|---|---|---|---|
| Spectral gap | \|Δ\| = 0.0000 | \|Δ\| = 0.0000 | Exchange-symmetric (exact) |
| Detailed balance error | \|Δ\| = 0.0000 | \|Δ\| = 0.0000 | Exchange-symmetric (exact) |
| Loop score | \|Δ\| = 0.0000 | \|Δ\| = 0.0000 | Exchange-symmetric (exact) |
| Mean prediction error | \|Δ\| = 0.0000 | \|Δ\| = 0.0000 | Exchange-symmetric (exact) |
| Lock status | 100% agreement | 100% agreement | Deterministic |
| Identical twin redundancy | 0.500 | 0.500 | Perfect overlap |
| Distinct agent redundancy | 0.983 | 0.983 | Near-perfect additive |

Node and Python now produce **byte-for-byte identical results**. All four physical observables show exactly zero difference between `combine(A,B)` and `combine(B,A)` across 20 independent trials — the combination operator's emergent statistics are exchange-symmetric with no measurable bias.

> **Note on methodology**: an earlier version of this experiment showed small nonzero noise (~1-3%) in spectral gap, detailed balance error, and loop score. That noise was traced to two library bugs, since fixed: (1) `combine()`'s joint meta-trie used bit-tags (`0x10000000`/`0x20000000`) that could collide with naturally-occurring high bits in the meta-state hash, corrupting a fraction of merged meta-states; (2) decision-time and lexicon-invention randomness wasn't fully threaded through the agent's configured `rng`, so `AB` and `BA` weren't always running on truly identical random streams. With both fixed, the exchange-symmetry result is now exact rather than "mostly symmetric within noise." In 3.0 the bit tags were replaced by collision-free provenance ids and `combine()` orders its arguments canonically, so `combine(A, B)` and `combine(B, A)` are identical agents by definition.

## Verdict: fermion-like hypothesis negated

This experiment set out to test two competing hypotheses about `combine()`:
- **Boson-like (symmetric)**: swapping constituents leaves all observables unchanged
- **Fermion-like (antisymmetric)**: swapping produces a structured, systematic difference — the kind of signature relevant to exclusion-type statistics (why quarks/electrons can't occupy the same state)

**Result: the fermion-like hypothesis is negated.** Exchange symmetry is exact (zero difference, not "mostly symmetric within noise") across every physical observable measured, in both language implementations. This is a clean null result, not an inconclusive one.

**Why this happens, mechanically**: `combine()` builds its joint state via set-union-style operations (trie path union, additive lexicon merge, averaged self-token parameters). Union-type operations are exchange-symmetric *by construction* — order literally cannot matter because union(A,B) = union(B,A) algebraically. Symmetric behavior isn't a discovery here; it's the necessary consequence of the specific implementation choice.

**What this implies going forward**: if a fermion-like/exclusion signature is desired (directly relevant to deriving quark-like statistics from agent networks), it will not emerge spontaneously from this style of combination operator. It would need to be deliberately designed into the combination kernel — e.g. an explicit antisymmetrization step, a sign-flip rule under exchange, or a hard occupation constraint preventing two agents from sharing a locked meta-state. This is a concrete, falsifiable next step rather than a dead end: the current operator is exchange-symmetric by construction; an antisymmetric variant is a distinct, well-defined design target. ("Boson-like" and "fermion-like" are analogies, not physical claims.)

## Caveats

1. **This is not a physics experiment** — it's an information-theoretic analysis of Markovian kernel combination operators. Results are directional signals for further formal investigation, not physics claims.

2. **The 0.500 ratio for identical twins is expected** from the trie merge algorithm — merging two identical tries produces a trie of the same size (not double). This is information-theoretic redundancy, not quantum exclusion. The question is whether and how this scales differently in larger agent networks.

3. **Binary masks** (`0x10000000` / `0x20000000`, now reserving the top 4 bits of the 32-bit meta-state hash) keep constituent identity distinguishable post-combination — fuse is invertible by design. This is an implementation choice, not a physical symmetry.

## Running

```bash
# Node
npm run examples:exchangeSymmetry

# Python
uv run python examples/07_exchange_symmetry/exchange_symmetry.py
```