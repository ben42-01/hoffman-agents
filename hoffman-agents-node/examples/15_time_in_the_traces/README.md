# Experiment 15: Time in the Traces

## Question

Hoffman's recent work centres on **traces**. An observer who can only see part of a network of conscious agents (its *window* S) does not experience the network's dynamics. It experiences the **trace chain** on S:

```
P_S = P_SS + P_SC (I − P_CC)⁻¹ P_CS        (C = everything outside the window)
```

The observer either stays in its window, or the network leaves it, wanders through states the observer cannot see for any number of steps, and comes back. Hoffman, Prakash & Chattopadhyay's Trace Chain Theorem (2025) states that this effective dynamics exists and is unique, and "A is a trace of B" is the ordering behind their *trace logic*.

**What does time look like from inside a trace?**

## Results (Node and Python print identical output)

**1. Every observer experiences its own Markov chain.** Two combined conscious agents act on a shared world (8 joint states). Three observers watch different windows. Simulated observations match the trace formula to within 0.006 after 200,000 steps. Nested observers are consistent: a trace of a trace equals the direct trace, to machine precision.

**2. Proper time is per-observer.** An observer's clock ticks only when the network is inside its window, so each observer ages at **π(S) ticks per network step**. By Kac's lemma, the mean number of hidden network steps between two of its moments is 1/π(S); this is confirmed to machine precision.

| Observer | ticks per network step | mean gap | 1/π(S) |
|---|---|---|---|
| sees only world = L | 0.502 | 1.99 | 2.00 |
| sees only agent 1 alert | 0.529 | 1.89 | 1.89 |
| both conditions | 0.266 | 3.76 | 3.78 |

Nested clocks compose exactly: the inner observer's rate, measured in the outer observer's own time, equals π(inner)/π(outer). The gaps between an observer's moments vary (standard deviation 1.4–3.6 network steps), but the observer has no access to that. It experiences one step per event.

**3. The arrow of time is observer-dependent.**
- **Reversible network.** There is no arrow in the dynamics (irreversibility 0 for the network and for the observer). Yet each observer's uncertainty about its future, H(Xₙ | X₀), grows monotonically while H(Xₙ) stays constant. The entropic arrow comes from observing, as Hoffman, Prakash & Prentner (2023) prove, not from the dynamics.
- **Irreversible network** (irreversibility 0.143). The observers see 0.012, 0.071 and 0.000. **A window of two states never perceives an arrow**, because every two-state chain satisfies detailed balance: even a strictly one-way 3-cycle looks time-symmetric through a two-state window. Perceiving an arrow of time needs a window of at least three states.

## What this does and does not show

- **Established results.** Parts 1 and 2 are standard mathematics about trace chains (Kac, transitivity), shown operating in a network of conscious agents. They are not new theorems. They make Hoffman's picture of observer-relative time concrete and checkable.
- **What is specific to this experiment.** The two-state result and the observer-dependence of the arrow are simple, but they are exactly the kind of statement a trace-based theory of time has to account for.
- **What remains open.** Whether these per-observer clocks obey **Einstein's time dilation**. Hoffman conjectures that Minkowski spacetime emerges from traces of n-cycle chains as n → ∞. [Experiment 18](../18_relativity_at_infinity/) tests the time-dilation part in a 1+1-dimensional model. As n → ∞, an agent's clock converges to Einstein's proper time, and only in the limit. Hoffman's exact construction from the 2025–26 trace-logic papers remains untested.

## Library

`trace.traceChain`, `isTraceOf`, `clockRate`, `meanReturnTime` and `conditionalEntropyProfile` (Python: `trace.trace_chain` and so on), plus `MarkovKernel.trace(window)` and `hasTrace` / `has_trace`.

## Run

```bash
npm run examples:timeInTheTraces                                  # Node
python examples/15_time_in_the_traces/time_in_the_traces.py       # Python
```
