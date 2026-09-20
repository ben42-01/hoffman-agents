# Experiment 17: Quantum Agents

## Question

Experiment 13 showed that classical conscious-agent networks behind a headset give CHSH = 2 + 2δ. That ranges anywhere from 2 up to the PR box's 4, and without an extra rule 98% of the networks signal. Nature stops at **Tsirelson's bound 2√2 ≈ 2.83** and never signals.

Markov kernels are the simplest choice of dynamics, not the only one. Here the agents' kernels carry **complex amplitudes**:

| Six-tuple part | Markov agent | Quantum agent |
|---|---|---|
| experiences X | probability distribution | density matrix (qubit) |
| perception P, action A | Markov kernel | quantum channel (Kraus operators) |
| decision D | Markov kernel | measurement (Born rule) |
| combination | product of kernels + interaction | tensor product + interaction unitary |

**Does combining quantum agents reproduce the quantum limit 2√2 exactly? If so, what does the work?**

## Results (Node and Python print identical output)

**1. Markov agents are quantum agents that have fully decohered.** Each Markov kernel P becomes a channel with Kraus operators √P_ij |j⟩⟨i|. On classical (diagonal) states it acts exactly as P, and it erases all coherence. We checked 50 random kernels; each property holds to machine precision.

**2. Combination without interaction gives no Bell correlations.** Across 1000 random product agents (pure and noisy), the best CHSH over all measurements is 2.000000, the local bound.

**3. Combination with interaction reaches the quantum limit exactly.** Two agents start in |00⟩ and interact through U(t) = exp(−i t X⊗X):

| t | best CHSH | formula 2√(1 + sin²2t) | measured | classified |
|---|---|---|---|---|
| 0 | 2.000000 | 2.000000 | 2.000000 | local |
| π/8 | 2.449490 | 2.449490 | 2.449490 | quantum |
| π/4 | **2.828427** | 2.828427 | 2.828427 | quantum |

- **At t = π/4, CHSH equals 2√2** to machine precision.
- **Random interactions never exceed it.** Over 2000 random interactions (random two-qubit unitaries on random product agents, a quarter with noise), the best value is 2.8207. None exceeds 2√2; 80.7% exceed 2.
- **The measurements confirm it.** Measuring with the optimal settings gives statistics that match these values to machine precision. Experiment 13's Bell classifier labels them "local" or "quantum", never "post-quantum" or "signalling".
- **No-signalling holds automatically.** The largest signalling is at machine precision.

**4. Network distance and decoherence.** After the interaction, each agent's experience passes through k steps of a local channel:

| k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| noise (depolarising p = 0.1): 2√2·0.9^(2k) | 2.828 | 2.291 | 1.856 | 1.503 | 1.218 | 0.986 |
| decoherence (dephasing p = 0.2): 2√(1 + 0.6^(4k)) | 2.828 | 2.126 | 2.017 | 2.002 | 2.000 | 2.000 |

- Both closed forms hold to machine precision.
- **Full dephasing turns both agents into Markov agents, and CHSH falls to exactly 2**, the classical bound.
- Correlations decay with network distance as in experiment 13, but they start from 2√2, not from 4.

**5. Interference.** An agent reaches an experience by three routes, and any subset of routes can be blocked. Sorkin's (1994) interference terms measure two-route (I2) and three-route (I3) interference. Across 200 random agents:

| | largest \|I2\| | largest \|I3\| |
|---|---|---|
| quantum agents | 0.222 | machine precision |
| half decohered | 0.111 | machine precision |
| fully decohered | machine precision | machine precision |
| Markov agents | machine precision | machine precision |

Quantum agents interfere in pairs but never three at a time. That is the signature of the Born rule, which photon experiments confirm (Sinha et al. 2010). Decoherence scales the interference down to zero.

## What this does and does not show

- **The answer is yes.** Combined quantum agents reproduce the quantum limit exactly and never exceed it. They also never signal, without any extra rule.
- **But the quantum structure does the work, not the agent formalism.** Tsirelson's theorem says any complex Hilbert space with the Born rule caps CHSH at 2√2. The tensor-product combination with local measurements guarantees no-signalling. Classical kernels supply neither (experiment 13). So quantum agents answer experiment 13 **by assumption**: the result is a consistency check, not a derivation of quantum mechanics from consciousness.
- **What is established.** The classical world sits inside this model as its decohered limit. Markov agents are exactly the quantum agents without coherence, and removing coherence moves Bell correlations and interference continuously to their classical values.
- **What is open.** Hoffman's program needs a reason for the Hilbert-space structure. Either it is derived from agent dynamics (for example through the decorated permutations and positive Grassmannian of experiment 16, where the amplituhedron lives), or quantum agents are taken as the starting point.

## Library

`quantum` (Python: `conscious_agent.quantum`) contains:
- channels: `applyChannel`, `onQubit`, `depolarizing`, `dephasing`, `markovChannel`, `krausDeviation`;
- Bell analysis: `maxChsh` (Horodecki criterion), `optimalSettings`, `behaviour`, `correlationMatrix`;
- helpers: `pauliRotation`, `randomUnitary2`, `densityFromState`.

It is dependency-free in Node and uses numpy in Python.

## Run

```bash
npm run examples:quantumAgents                              # Node
python examples/17_quantum_agents/quantum_agents.py         # Python
```
