# Experiment 16: Trace Logic and Decorated Permutations

## Question

Markov chains are the simplest choice of dynamics for conscious agents. Hoffman's program offers two ways to go beyond the chain itself toward physics:

- **Decorated permutations** (Hoffman, Prakash & Prentner, *Fusions of Consciousness*, 2023, Definition 2). Every Markov chain maps to a decorated permutation, the combinatorial object that labels cells of the positive Grassmannian. That geometry underlies the amplituhedron and scattering amplitudes in particle physics.
- **Trace logic** (Hoffman & Prakash, *Traces of Consciousness*). Agents are ordered by "A is a trace of B". "A observes B" means A's qualia kernel Q = D·A·P is a trace of B's dynamics. The paper claims this logic is locally Boolean, globally not even a lattice, and maps homomorphically onto the Lebesgue logic of belief (Bennett, Hoffman & Murthy 1993) via the stationary measure.

**What exactly do these constructions keep, what do they depend on, and do their claims hold?** A third part stress-tests two physics proposals from the trace-logic papers: **mass ↔ entropy rate** and **speed ↔ commute time**.

## Results (Node and Python print identical output)

### A. Decorated permutations

The map works as follows:
- A transient state a goes to a.
- An absorbing state a goes to a + n.
- Any other recurrent state a goes to the first b > a whose cyclic interval (a … b) covers a's communicating class.

**Checked against the paper.** The map reproduces the paper's worked example, σ = [8, 11, 4, 12, 10, 15, 9, 14, 16]. It also agrees with **all 27 entries of the paper's Appendix B table** (the vertices of the Markov polytope M3). That table contains **14** distinct decorated permutations, and all 343 transition patterns of 3-state chains also give 14. The paper's text says 17; the text miscounts, and the table is correct.

**What σ keeps.**
- **Probabilities are discarded.** An agent that almost never changes experience ([[0.99, 0.01], [0.01, 0.99]]) and one that almost always does ([[0.01, 0.99], [0.99, 0.01]]) both map to [2, 3].
- **The dynamics inside a class is invisible.** For an irreducible agent, σ is the same list for every chain and every numbering. It records the class, not what the agent does.
- **σ depends on how experiences are numbered.** Across all 720 relabellings of 6-state chains, σ renames consistently exactly when the relabelling keeps the cyclic order of every recurrent class of ≥ 3 states. This rule holds for every relabelling tested.

| 6-state chain | σ | renames consistently | identical list |
|---|---|---|---|
| classes (1 3 5)(2 6)(4) | [5, 6, 7, 10, 9, 8] | 360/720 | 12/720 |
| one 6-cycle | [6, 7, 8, 9, 10, 11] | 6/720 | 720/720 |
| classes (1 2)(3 4)(5 6) | [2, 7, 4, 9, 6, 11] | 720/720 | 48/720 |
| random irreducible agent | [6, 7, 8, 9, 10, 11] | 6/720 | 720/720 |

This is not a bug in the map. Scattering amplitudes also come with a cyclic order ("colour ordering"). It is a requirement: to reach physics this way, the program needs a principle that puts experiences in a cyclic order, and it needs dynamics that σ can see.

### B. Trace logic

- **Observation.** The trace of a network on an observer's window satisfies "A observes N". A random kernel on the same window does not. Qualia kernels Q = D·A·P are stochastic, as required.
- **Locally Boolean (confirmed).** For a fixed 4-experience agent, the trace order matches window inclusion in 225 of 225 pairs. Its 15 traces, plus a zero, form the Boolean algebra of its windows: join is the trace on the union, meet the trace on the intersection, complement the trace on the complementary window.
- **Globally not a lattice (confirmed).** Two different agents K1 and K2 on {a, b} both lie above "trace on {a}" and "trace on {b}", and all four are pairwise incomparable. So K1 and K2 have no meet and {a}, {b} have no join. There is no top element either.
- **Homomorphism to Lebesgue logic (confirmed).** In 199 of 199 random (agent, window) pairs, A ≤ₜ B implied π_A ≤_L π_B. The map is **not injective**: N and N² have the same stationary measure. **The converse fails**: a kernel with the observer's stationary measure but different dynamics satisfies ≤_L without being a trace.

This logic is **not quantum logic**. Quantum logic is an orthomodular lattice that fails only distributivity. Trace logic fails the lattice axioms themselves.

### C. Physics proposals, stress-tested

These are proposals from the papers, not theorems. The checks below could have failed.

- **Mass ↔ entropy rate.**
  - Mass of independent systems adds, and so does entropy rate under ⊗: h(P1 ⊗ P2) = h(P1) + h(P2) to machine precision. This is also a theorem.
  - Bound systems weigh less than their parts (mass defect). Coupling the three ring agents of experiment 14 lowers the entropy rate: 3.119 → 3.064 → 2.909 → 2.640 as coupling grows from 0 to 0.9. This is qualitatively consistent.
- **Speed ↔ commute time.** For speed to be distance over time, commute time must be a distance. It is: it is symmetric, and the triangle inequality holds in all 6480 triples tested. This is also a theorem, for any irreducible chain.

Passing these checks is necessary, not sufficient. Nothing here derives a mass spectrum or a speed limit.

## What this does and does not show

- **Established.** Both constructions are implemented exactly as defined and checked against the paper's own data. Trace logic has the structure the papers claim.
- **Clarified.** Decorated permutations keep only which experiences communicate and in what cyclic order. Any route from them to particle physics has to supply that order and cannot depend on transition probabilities.
- **Corrected.** Fusions of Consciousness says M3 yields 17 decorated permutations; its own table, and exhaustive enumeration, give 14.
- **Open.** Whether the physics proposals hold quantitatively, and whether any principle picks out the cyclic order.

## Library

- `decorated.decoratedPermutation`, `relabel`, `isCovariant` (Python: `decorated.decorated_permutation` and so on).
- `trace.qualiaKernel`, `lebesgueLeq`, `stationaryMeasure`.
- `markov.entropyRate`, `hittingTimes`, `commuteTimes`.

The Appendix B table is stored in `test/fixtures/fusions-appendix-b.json` (Python: `tests/fixtures/`). It is from the paper, which is CC BY 4.0.

## Run

```bash
npm run examples:traceLogic                                                        # Node
python examples/16_trace_logic_and_decorated_permutations/trace_logic.py           # Python
```
