# Mathematical Model (v3)
## Kernels, ergodic dynamics, the "I" lock and combination — precisely

This document specifies the mathematics implemented by `conscious-agent` 3.x in both
the Node (`hoffman-agents-node`) and Python (`hoffman-agents-python`) libraries. The two
implementations are checked against each other step for step (see
[Parity](#9-cross-language-parity)).

Every agent runs with `mathVersion: 'v3'` (Node) / `math_version="v3"` (Python) by
default. `'legacy'` reproduces 2.x exactly; [§10](#10-legacy-math-and-why-it-changed)
lists what was wrong with it.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `W` | world states (32-bit ids: first 4 bytes of SHA-256 of the observed sequences) |
| `T` | experience trie: counts `N(w → w')` of observed world transitions |
| `M` | meta-trie: counts `N(m → m')` of transitions between *meta-states* |
| `m` | meta-state: a coarse self-observation (below) |
| `D` | decision kernel over output modes `{core, lexicon, explore, idle}` |
| `π` | stationary distribution: `π = πK`, `Σπ = 1` |
| `n` | number of states in the recurrent class under analysis |

A **kernel** `K` is row-stochastic: `K(i,j) ≥ 0`, `Σ_j K(i,j) = 1`. Distributions are
row vectors, so one step is `μ' = μK`.

---

## 2. Perception: graded prediction error

On each world observation `w_t` after `w_{t-1}` the agent computes

```
p = P̂(w_t | w_{t-1})            Witten–Bell estimate from T
prediction error  e_t = 1 − p   ∈ [0, 1]
surprisal         s_t = −ln p   (recorded on the trace event)
```

Witten–Bell estimate for a source state with `N` outgoing transitions over `u`
distinct successors, and `K` possible successors (known states + 1 for "never seen"):

```
seen successor w':    c(w') / (N + u)
unseen successor:     u / (N + u) / (K − u)
source never seen:    1 / K
```

After `n` identical transitions `p = n/(n+1)`, regardless of how many other states
exist. Add-α smoothing was rejected because it dilutes a deterministic successor as the
state space grows. With 50 known states and 10 observations, Laplace smoothing still
reports error 0.82.

The transition is stored with `insertTransition(from, to)`, which counts the depth-2
node only. 2.x also incremented the depth-1 node, so visits were double-counted.

## 3. Self-observation: the meta-state chain

Every `metaObservationInterval` steps (default 20) the agent observes itself:

```
m = FNV-1a32( JSON([ last two world ids mod 8, errorBucket(mean error of last 10), outputMode ]) ) & 0x0FFFFFFF
errorBucket(e) = 0 if e<.05, 1 if e<.15, 2 if e<.35, 3 if e<.65, else 4
```

and records the transition `m_prev → m`. This includes `m_prev = m`: staying in the same
self-state is dwell time in an attractor, and it has to count. The lock flag is **not**
part of `m`. Including it (2.x) split the chain into disconnected pre-lock and post-lock
components.

`outputMode` is the decision state the agent was in. This is the strange loop: what the
agent does becomes part of what it observes about itself.

## 4. Ergodic analysis of the meta-chain

`MetaTrie.ergodicDiagnostics()` / `ergodic_diagnostics()`:

1. **Count matrix** `C(m, m') = N(m → m')` over the agent's native meta-states. States
   inherited through combination are excluded; see §7.
2. **Prune unobserved rows.** Repeatedly remove states with no outgoing count to a
   surviving state. Their kernel row is *unknown*, typically because they are the newest
   state. Making them absorbing (2.x) gives them all stationary mass and fakes an
   attractor: for the chain `1→2→3`, 2.x returns `π = (0, 0, 1)`.
3. **Communicating classes** (Tarjan SCC). A class is **closed** (recurrent) when no mass
   leaves it. Choose the closed class containing the most recent meta-state in the
   history (the attractor the agent is in). If none contains one, choose the class with
   the most transitions.
4. On that irreducible class `R` (size `n`), with `P = C_R` row-normalised:
   - `π` by power iteration on the **lazy chain** `L = (P + I)/2`. `L` has the same `π`
     and is aperiodic, so iteration converges even for periodic `P`. 2.x iterated `P`
     directly and returned `(1/3, 2/3)` for a 2-cycle; the correct answer is `(½, ½)`.
     A direct linear solve is the fallback.
   - **period** `d = gcd{ level(u) + 1 − level(v) : u→v in R }` from BFS levels.
     `aperiodic ⇔ d = 1`.
   - `|λ₂|` by power iteration on the deflated operator `x ↦ xP − (Σ xP) π`.
     **Relaxation time** is `t_rel = 1/(1 − |λ₂|)`. **Mixing time** is estimated as
     `t_mix(¼) ≤ t_rel · ln(4/π_min)`, a bound for reversible chains and an estimate
     otherwise.
   - `dominance = π_max − 1/n` (1 when `n = 1`), `H(π)`, `KL(π‖U) = ln n − H(π)`, and
     **occupancy** (the fraction of the last 20 meta-observations spent in `R`).

The dominant state is `argmax π`, taking the lowest index among values within 1e-12.
That keeps Node and Python deterministic on exact ties.

## 5. The "I" lock

"I" locks onto `m* = argmax π` when **all** of the following hold for
`lockConsecutiveRequired` consecutive self-observations:

| # | Criterion | Condition | Default |
|---|---|---|---|
| 1 | evidence | `N_R ≥ minTransitions` and `N_R ≥ minTransitionsPerState · n` | 20, 2 |
| 2 | ergodicity | `π` converged and `R` aperiodic | — |
| 3 | dominance | `π(m*) − 1/n ≥ lockMargin` (or normalised `KL(π‖U) ≥ klThreshold`, if set) | 0.15, off |
| 4 | occupancy | fraction of recent observations in `R` `≥ minOccupancy` | 0.6 |
| 5 | stability | `m*` equals the previous observation's dominant state | — |
| | persistence | consecutive observations meeting 1–5 | 3 |

**Why each criterion is there:**
- **Dominance** is measured against the uniform baseline `1/n`. A fixed threshold on
  `π_max` (2.x: `> 0.25`) is met by *any* chain with `n ≤ 4`.
- **Evidence** is measured per state, because a sparse empirical chain (about one sample
  per row) has a far-from-uniform `π` from sampling noise alone.
- **Occupancy** requires the agent to actually be *in* the attractor now, not merely to
  have visited it.

**Unlock (hysteresis).** Once locked, "I" unlocks after `unlockConsecutiveRequired` (3)
consecutive observations with `dominance < unlockMargin` (0.05) or
`occupancy < minOccupancy/2`. These checks run only while the evidence criterion holds.
Locking changes the agent's output mode, which creates new meta-states, so the lock is
not revised until the new chain has data.

Lock and unlock events are appended to `lockHistory` and emitted once via
`StepOutput.interrupt`.

**Measured behaviour** (tests `ilock.test.js` / `test_ilock.py`):

| World | 2.x | v3 |
|---|---|---|
| structureless (50-state random kernel), 3000 steps | locks at step 61, every seed | 0 of 50 seeds lock |
| dominant attractor (home w.p. 0.9) | locks at step 61 | 20 of 20 lock, around steps 460–600 |
| attractor → noise at step 1500 | never unlocks | unlocks (18 of 20 within 4500 steps) |
| constant world | locks at step 61 | locks at step 481, stays locked |

The lock can't fire before step `≈ 20 · (minTransitions + lockConsecutiveRequired)` at
default settings.

## 6. Decision kernel D

Output modes `S = (core, lexicon, explore, idle)`:

```
         core      lexicon    explore    idle
core   [ pStable,  pLexicon,  pExplore,  1 − pStable − pLexicon − pExplore ]
lexicon[ 0.70,     0.15,      0.10,      0.05 ]            (lexiconRow, configurable)
explore[ same as core row ]
idle   [ same as core row ]
```

The parameters are validated: each must be in `[0, 1]` and `pStable + pLexicon + pExplore ≤ 1`.
Invalid values throw when the agent is constructed. With the defaults, D is ergodic
with `π_D ≈ (0.789, 0.105, 0.055, 0.050)` and `|λ₂| = 0.05`. The API is
`agent.decisionKernel.diagnostics()` / `agent.decision_kernel.diagnostics()`.

Until "I" locks the agent only emits `wait`. D governs output after the lock. D is
therefore also the source of the `outputMode` component of meta-states: in a constant
world the post-lock meta-chain's stationary distribution tracks `π_D`.

`loopDepth` / `strangeLoopScore` depends on the mode by construction. The `core`
utterance `I notice I familiar` always scores 1.0, so the population loop score measures
the time spent in `core` (≈ `π_D(core)` once locked). It doesn't measure emergent
self-reference.

## 7. Combination ⊗ and fusion

For agents `A`, `B` with leaf sets `L_A`, `L_B`:

- **Identity.** `id(A ⊗ B) = "CA_" + sha256(JSON(sorted(L_A ∪ L_B)))[:12]`. Therefore
  `A ⊗ B = B ⊗ A` and `(A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)` as identities. Every other choice
  (trace interleaving, lexicon conflicts, parameter mixing) uses the canonical order
  (sorted agent ids), so the combined agent is identical whichever order the arguments
  come in.
- **Provenance, not bit tags.** Each constituent `c`'s meta-state `m` is stored under
  `fnv1a32("c:m") & 0x0FFFFFFF`, rehashed on collision, with provenance
  `id → (c, m)` and a snapshot tree of each constituent (its provenance, history, lock
  state and parameters). Nesting depth is unlimited. 2.x OR-ed 4-bit tags `0x1/0x2`
  into the ids, which collide from the second nesting level on.
- **Own meta-chain.** Inherited states are excluded from §4, so `A ⊗ B` starts with an
  empty native chain and **unlocked**. The constituents' referents live in different
  state spaces, so "both locked" (2.x) is not a lock of the combined agent.
- **Product prior** (Hoffman–Prakash). The independent combination of the meta-kernels
  is `(M_A ⊗ M_B)((a,b),(a',b')) = M_A(a,a') M_B(b,b')`. It has `π = π_A ⊗ π_B` and is
  ergodic iff both factors are. It is available as `productKernel(A, B)`; a summary is
  stored as `combinationPrior`.
- **World model and vocabulary.** The world tries are merged by summing counts, which
  gives the visit-weighted mixture of the two `P̂` kernels. For a label present in both
  lexicons, the entry with greater integration depth is kept and encounter counts are
  pooled.
- **Decision.** `D_{A⊗B} = w D_A + (1−w) D_B` (a convex mixture, which is stochastic).
  The default is `w = ½`; `combine(a, { other: b, weights: [w1, w2] })` (Node) or
  `combine(a, b, weights=(w1, w2))` (Python) sets it.
- **fuse** inverts combine into the *direct* constituents. Each one's meta-chain,
  provenance, history, lock state and parameters are restored exactly. Each part gets its
  own copy of the (still learning) world trie and lexicon. The joint meta-chain can't be
  attributed to either constituent and is dropped.

On commutativity: `CONSCIOUS_AGENTS_THEORY.md §1.3` allows order to matter in general.
That applies to *directed* joins, where one agent's actions feed the other's
perceptions. The operator implemented here is the *undirected* (independent-product)
join, which is commutative up to relabelling of the product space. The library picks a
canonical labelling, so it is commutative exactly. The exchange-symmetry experiment
(example 07) depends on this.

## 8. Formal agents and `toFormal()`

`FormalConsciousAgent(X, G, W, P, D, A)` is a finite reference implementation of
Hoffman & Prakash's `C = (X, G, P, D, A, N)`:

```
P[w] : X → Δ(X)       perception given world state w
D    : X → Δ(G)       decision
A[g] : W → Δ(W)       action g applied to the world
one cycle:  x' ~ P[w](x,·),  g ~ D(x',·),  w' ~ A[g](w,·),  N ← N+1
joint chain on X × W:  Q((x,w),(x',w')) = P[w](x,x') Σ_g D(x',g) A[g](w,w')
```

`diagnostics()` gives irreducibility, period, `π_Q`, its marginals on X and W, and
mixing. The tests check the **ergodic theorem**: empirical frequencies along one
simulated trajectory converge to `π_Q`.

`FormalConsciousAgent.combine(c1, c2)` builds `X = X₁×X₂`, `G = G₁×G₂`,
`P[w] = P₁[w] ⊗ P₂[w]`, `D = D₁ ⊗ D₂`, and `A[(g₁,g₂)] = A₁[g₁]·A₂[g₂]`. The actions are
applied in sequence because two agents writing to one world need an order or a joint
action kernel.

`ConsciousAgent.toFormal()` / `to_formal()` exports a learned agent's kernels in the same
vocabulary:
- `X`: the recurrent meta-states
- `P`: the pruned empirical world kernel from `T`
- `M`: the meta kernel
- `D`: the decision kernel
- `A`: meta-state → token distribution
- `N`: the step count

## 9. Cross-language parity

Everything that feeds back into dynamics is deterministic and specified bit-for-bit:

| Primitive | Specification |
|---|---|
| RNG | `mulberry32(seed)` (`seed` option on the agent); `choice(seq) = seq[⌊r·len⌋]` |
| hashes | FNV-1a 32 over UTF-8; meta-state JSON with no whitespace; state ids = first 4 bytes of SHA-256 |
| sampling | inverse CDF with strict `<`, one draw per sample |
| ties | lowest index within 1e-12 of the maximum |

`test/parity.test.js` and `tests/test_parity.py` run identical scenarios in both
languages: 9,000 agent steps across attractor and noise worlds, plus combine and fuse.
They require every output token, meta-state id, lock event, lexicon entry and trie size
to be identical, and floats to agree within 1e-9.

`SimpleWorld` is not part of the parity contract. Its RNG has always differed between
the Node and Python libraries.

## 10. Legacy math and why it changed

`mathVersion: 'legacy'` keeps 2.1.2 behaviour. Golden fixtures in both test suites check
it byte for byte. Files saved without `formatVersion` load as legacy.

| # | 2.x behaviour | Consequence | v3 |
|---|---|---|---|
| 1 | newest meta-state made absorbing | all stationary mass on an unobserved state; lock in any world | prune unobserved rows (§4) |
| 2 | self-transitions not recorded | real attractors under-weighted | recorded (§3) |
| 3 | power iteration on `P` | wrong `π` for periodic chains | lazy chain + solve (§4) |
| 4 | lock iff `π_max > 0.25` ×3 | trivially met for `n ≤ 4`; locks at step 61 everywhere; never unlocks | ergodic lock criteria + hysteresis (§5) |
| 5 | lock flag hashed into meta-state ids | chain splits at lock time | removed (§3) |
| 6 | implicit, unvalidated decision chain | p's could sum > 1; no analysis | explicit kernel D (§6) |
| 7 | order-dependent ids; 4-bit tags; shared lexicon on fuse | `A⊗B ≠ B⊗A`; nested collisions; aliasing | canonical ids, provenance, copies (§7) |
| 8 | 0/1 argmax prediction error | no graded surprise | Witten–Bell `1 − p` (§2) |
| 9 | Python `hash()`; 64-bit Python state ids; unseeded RNG | irreproducible runs; Node ≠ Python | FNV-1a, 32-bit ids, mulberry32 (§9) |

## 11. Traces: what an observer experiences

An observer that can only see a subset S (its window) of a network with kernel P experiences the **trace chain**

```
P_S = P_SS + P_SC (I − P_CC)⁻¹ P_CS        C = complement of S
```

It either stays in S, or leaves, spends any number of hidden steps in C, and returns. It exists and is unique whenever the chain cannot avoid S forever (Hoffman, Prakash & Chattopadhyay's Trace Chain Theorem, 2025; the construction is the standard induced or censored chain). Properties used by the library and tested in `test/trace.test.js` / `tests/test_trace.py`:

- **Transitivity.** The trace of a trace is the trace on the smaller window. This gives the partial order "A is a trace of B" behind Hoffman's trace logic.
- **Stationary distribution.** The trace's stationary distribution is π restricted to S and renormalised.
- **Proper time (Kac).** The mean number of network steps between two observer events is 1/π(S). Each observer's clock runs at π(S) ticks per network step, and clocks of nested observers compose.
- **Arrow of time.** For a stationary chain, H(Xₙ | X₀) is non-decreasing in n even when H(Xₙ) is constant: the entropic arrow comes from conditioning on an observation. The trace of a reversible chain is reversible. Every two-state chain satisfies detailed balance, so an observer whose window has two states never perceives irreversibility.

API: `trace.traceChain(P, S)`, `isTraceOf`, `clockRate`, `meanReturnTime`, `conditionalEntropyProfile`, `MarkovKernel.trace(window)` (Python: snake_case). Demonstrated in experiment 15.

Hoffman conjectures that Minkowski spacetime (Lorentz time dilation) emerges from traces of n-cycle chains as n → ∞. §14 establishes the time-dilation part in a 1+1-dimensional model.

## 12. Decorated permutations and trace logic

**Decorated permutations** (Hoffman, Prakash & Prentner, *Fusions of Consciousness*, 2023, Def. 2). A kernel on states 1..n, in a fixed order, maps to σ: {1..n} → {1..2n}:

- a transient state a goes to a;
- an absorbing state a goes to a + n;
- any other recurrent state a goes to the first b > a such that the cyclic interval (a, …, b), read mod n, covers a's communicating class.

Decorated permutations index cells of the positive Grassmannian. Facts established in experiment 16 and its tests:

- **Checked against the paper.** σ agrees with all 27 entries of the paper's Appendix B table (the vertices of M3). The table holds **14** distinct decorated permutations, and exhaustive enumeration of 3-state transition patterns also gives 14. The paper's text says 17.
- **What σ depends on.** σ depends only on the class partition and the cyclic order of the numbering. Transition probabilities are discarded, and for an irreducible chain σ is the same for every kernel.
- **Covariance.** Under relabelling, σ transforms by conjugation exactly when the relabelling preserves the cyclic order within every recurrent class of ≥ 3 states. This held for all 720 relabellings of the 6-state test chains.

**Trace logic** (Hoffman & Prakash, *Traces of Consciousness*). The **qualia kernel** of an agent is Q = D·A·P (X → G → W → X). Kernels on labelled state sets are ordered by K_A ≤ₜ K_B iff A's states are a subset of B's and K_A is the trace chain (§11) of K_B on them. "A observes B" means Q_A ≤ₜ Q_B.

- **Local structure.** The traces of one fixed kernel form the Boolean algebra of its windows: join is the trace on the union, meet the trace on the intersection, complement the trace on the complement.
- **Global structure.** Globally the order is not a lattice. Two different kernels on {a, b} have two maximal common lower bounds (the traces on {a} and on {b}), and there is no top element.
- **Lebesgue logic.** K ↦ π_K is order-preserving into the Lebesgue order on measures (ν ≤_L μ iff supp ν ⊆ supp μ and μ restricted to supp ν is proportional to ν; Bennett, Hoffman & Murthy 1993). The map is not injective (P and P² have the same π), and the converse fails.

**Proposed physical readings** (proposals, not theorems):

- **Mass ↔ entropy rate** h = Σᵢ πᵢ H(P_i·). It is additive under ⊗, and it decreases when agents are coupled.
- **Speed ↔ commute time** κ(i, j) = E_i T_j + E_j T_i. It is a metric on the states of any irreducible chain.

API: `decorated.decoratedPermutation`, `relabel`, `isCovariant`; `trace.qualiaKernel`, `lebesgueLeq`, `stationaryMeasure`; `markov.entropyRate`, `hittingTimes`, `commuteTimes` (Python: snake_case).

## 13. Quantum agents

A quantum agent replaces each part of the six-tuple with its quantum counterpart:

| Markov agent | Quantum agent |
|---|---|
| probability vectors | density matrices ρ |
| Markov kernels | channels ρ ↦ Σ_k K_k ρ K_k† with Σ K_k†K_k = I |
| decisions | measurements (Born rule) |

Combination is the tensor product; interaction is a unitary on the joint space.

- **Embedding.** A Markov kernel P is the channel with Kraus operators K_ij = √P_ij |j⟩⟨i|. It acts as P on diagonal states and erases all off-diagonal terms. Markov agents are therefore exactly the fully decohered quantum agents.
- **Bell correlations.** For a two-qubit state with correlation matrix T_ij = Tr(ρ σ_i ⊗ σ_j), the best CHSH value over spin measurements is 2√(m₁ + m₂), where m₁ ≥ m₂ are the two largest eigenvalues of TᵀT (Horodecki 1995). This is at most 2√2 (Tsirelson).
  - Product states give ≤ 2.
  - The interaction exp(−it X⊗X) applied to |00⟩ gives 2√(1 + sin²2t).
  - Local measurements on a tensor product cannot signal.
- **Decoherence.** Depolarising noise of strength p on both agents multiplies T by (1 − p)². Dephasing multiplies its X/Y part by (1 − 2p)². Full dephasing leaves a classical state with CHSH ≤ 2.
- **Interference.** With routes combined by adding amplitudes, Sorkin's I₂ is generally non-zero but I₃ = 0 (Born rule). For Markov agents both vanish.

The Tsirelson cap and no-signalling come from the Hilbert-space structure and the tensor product. They are assumed here, not derived from the agent formalism (experiment 17).

API: `quantum.maxChsh`, `optimalSettings`, `behaviour`, `correlationMatrix`, `applyChannel`, `onQubit`, `markovChannel`, `depolarizing`, `dephasing`, `pauliRotation`, `randomUnitary2`, `krausDeviation` (Python: `conscious_agent.quantum`, snake_case).

## 14. Relativity at infinity

**The agent.** An agent's experience is (position on a ring with n sites per unit length, direction bit). Each step it moves one site in its current direction, then reverses with probability a/n. The quantum version reverses with amplitude i·sin(m/n) and continues with cos(m/n), a unitary discrete-time quantum walk. The clock observer counts reversals. Take n·t steps with x·n net displacement, and let τ = √(t² − x²) (c = 1). For walks that start heading right and end heading left, the n → ∞ limits are:

| quantity | limit |
|---|---|
| probability density | a·e^(−at)·I₀(aτ) |
| clock reading (mean reversals) | 1 + aτ·I₁(aτ)/I₀(aτ) |
| quantum amplitude density | i·m·J₀(mτ) |

These follow by summing over paths with 2j − 1 reversals: the path volumes are (t_R t_L)^(j−1)/((j−1)!)², with t_R,L = (t ± x)/2, so 2√(t_R t_L) = τ. They are the telegraph process (Goldstein 1951; Kac 1974) and Feynman's checkerboard; see Gaveau, Jacobson, Kac & Schulman (1984).

- **The clock and the quantum amplitude depend only on proper time.** This is time dilation. Exact dynamic programming shows convergence ∝ 1/n.
- **The classical density keeps e^(−at).** The network's rest frame is therefore preferred.
- **The light cone needs memory.** The top speed is c at every n. A memoryless walk with the same diffusion constant has top speed √(n/a) → ∞.

Demonstrated in experiment 18. It is 1+1-dimensional, and it is not Hoffman's exact construction.

## 15. Known limitations

- **No forgetting.** Transition counts accumulate forever. After a regime change, unlock
  takes on the order of the time already spent in the old regime (about 3,000 steps
  after 1,500 steps of attractor). A discounted estimator is the natural next step.
- **Coarse meta-states.** `id mod 8` and five error buckets are a lossy abstraction.
  Distinct experiences can share a meta-state, and FNV-1a over 28 bits can collide
  (in the native chain, collisions are only resolved against inherited ids).
- **Estimates, not truths.** Every quantity in §4 is computed from the empirical chain.
  The evidence criterion bounds the sampling noise but doesn't remove it.
- **Mixing bounds** are exact only for reversible chains; learned chains usually aren't
  reversible.
