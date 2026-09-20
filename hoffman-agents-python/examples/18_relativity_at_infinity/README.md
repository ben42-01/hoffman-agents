# Experiment 18: Relativity at Infinity

## Question

Hoffman conjectures that Minkowski spacetime, with Einstein's time dilation, emerges from the traces of n-cycle Markov chains **as n → ∞**. No finite network has relativity. The question is what appears in the limit.

**The agent.** Its experience is a position on a ring with n sites per unit length, plus **one bit of memory**: the direction it is heading. Each step it moves one site, and with probability a/n it reverses. In the quantum version, a reversal has amplitude i·sin(m/n), which makes it a discrete-time quantum walk.

**Two observers.**
- **The lab** sees the agent's position after t·n steps.
- **The agent's clock** registers only the agent's changes of experience, its reversals. This is the observer-restricted time of experiment 15.

Everything is computed exactly by dynamic programming, with no sampling, for n = 20, 80, 320 and 1280.

## Results (Node and Python print identical output)

### A. A light cone needs memory

After one unit of lab time, the agent with a direction bit is never beyond distance 1: its top speed c = 1 holds at every n. A memoryless control has the same long-run spread, but to achieve it its steps must get faster as n grows. Its top speed √(n/a) diverges, and the probability of landing beyond distance 1 tends to the diffusion value.

| n | 20 | 80 | 320 | 1280 | limit |
|---|---|---|---|---|---|
| memoryless: top speed | 2 | 4 | 8 | 16 | ∞ |
| memoryless: mass beyond distance 1 | 0.0118 | 0.0183 | 0.0218 | 0.0235 | 0.0253 |
| with direction bit: mass beyond distance 1 | 0 | 0 | 0 | 0 | 0 |

### B. Time dilation appears in the limit

The clock reading is averaged over walks that start heading right and end heading left at displacement x after lab time t. In the continuum the reading is **1 + aτ·I₁(aτ)/I₀(aτ), a function of proper time τ = √(t² − x²) alone**. At n = 1280, inverting that formula gives the proper time each clock has experienced:

| speed v | clock reading | proper time read off the clock | Einstein √(1 − v²) |
|---|---|---|---|
| 0 | 5.4671 | 1.0000 | 1.0000 |
| 0.6 | 4.4534 | 0.7999 | 0.8000 |
| 0.8 | 3.4279 | 0.5996 | 0.6000 |

Moving clocks run slow by the Lorentz factor. Three agents with the same proper time τ = 0.6, but different speeds, show how this emerges:

| n | clock readings (v = 0, 0.6, 0.8) | spread |
|---|---|---|
| 20 | 3.4436, 3.4299, 3.2103 | 0.2333 |
| 80 | 3.4355, 3.4360, 3.3922 | 0.0438 |
| 320 | 3.4314, 3.4317, 3.4214 | 0.0103 |
| 1280 | 3.4303, 3.4304, 3.4279 | 0.0025 |

The spread shrinks like 1/n. Every finite network has a preferred frame, the lattice. The relativistic clock exists only in the limit.

### C. Full Lorentz invariance needs complex amplitudes

- **Classical agent.** For the same three agents with τ = 0.6, the probability density tends to a·e^(−at)·I₀(aτ): 1.215, 0.574 and 0.164. The factor e^(−at) depends on lab time, so the network's rest frame stays special. The classical clock is relativistic, but the classical dynamics as a whole is not.
- **Quantum agent.** The amplitude tends to i·m·J₀(mτ), a function of τ alone. The spread across the three agents falls from 0.43 to 0.009 as n goes from 20 to 1280. This is one component of the 1+1-dimensional **Dirac propagator** (Feynman's checkerboard), which is Lorentz covariant. The reversal rate m plays the role of **mass**. The evolution is exactly unitary.

## What this does and does not show

- **Established, in this model.** A maximal speed, time dilation, and (with complex amplitudes) a Lorentz-covariant propagator all emerge from a conscious agent with one bit of memory. They emerge **only as n → ∞**: every finite network has a preferred frame, with corrections of order 1/n. This is the time-dilation part of Hoffman's conjecture, holding in a precise sense.
- **Not new mathematics.** The results are the telegraph process (Goldstein 1951; Kac 1974), Feynman's checkerboard (Feynman & Hibbs 1965), and the relativistic analogy of Gaveau, Jacobson, Kac & Schulman (1984). What is new is reading them as agents and observer clocks, and checking the convergence exactly.
- **Limits.** The model is 1+1-dimensional, and it is our construction, not the exact one in Hoffman's trace papers. As in experiment 17, full Lorentz covariance uses complex amplitudes, which are assumed rather than derived.

## Closing note for version 3

Experiments 13 to 18 point to a consistent picture. Finite agent networks reproduce the *structure* of physics only partly: correlations without the quantum cap, geometry that is inherited rather than generated, and clocks with a preferred frame. The pieces that are missing appear in the limit, or with complex amplitudes. Any finite model sees a finite shadow of whatever lies beyond spacetime. This library is a tool for making those shadows exact.

## Run

```bash
npm run examples:relativityAtInfinity                                    # Node
python examples/18_relativity_at_infinity/relativity_at_infinity.py      # Python
```
