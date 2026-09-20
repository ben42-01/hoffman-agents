# Glossary of Terms

For a non-specialist joining the project.

---

### Agent (Conscious Agent / CA)
A computational object implementing Hoffman & Prakash's six-tuple `(X, G, P, D, A, N)` of Markov kernels acting on a world W. `FormalConsciousAgent` is the exact tuple; `ConsciousAgent` learns its kernels from experience and forms an "I" when its self-model has a stable attractor.

### Experience Space (X)
The set of all possible experiences an agent can have. Private to that agent. Represented computationally as a tuple of four structures: T (world model), M (self-model), I (identity attractor), L (lexicon).

### Action Space (G)
The set of possible outputs an agent can produce. What it broadcasts to other agents.

### World Space (W)
What the agent perceives as "the world." In Hoffman's framework W is ultimately other agents' experience; in the library it can be other agents' outputs or a data source.

### Perception Kernel (P)
`P: W × X → Δ(X)`: given the world state and the current experience, a probability distribution over the next experience.

### Decision Kernel (D)
`D: X → Δ(G)`: given the current experience, a probability distribution over actions. In `ConsciousAgent`, D is the 4-state kernel over output modes (core, lexicon, explore, idle).

### Action Kernel (A)
`A: G × W → Δ(W)`: how an action changes the world.

### Counter (N)
The number of completed perceive–decide–act cycles: the agent's own clock.

### Combination Operator (⊗)
The operation that merges two agents into a higher-order agent whose experience space is the product of theirs. The implemented operator is the independent-product join: associative, commutative, with the trivial agent CA0 as identity.

### Fusion Operator
The inverse of combination: `fuse()` splits a combined agent into its direct constituents and restores each one's self-model, lock state and parameters exactly (from provenance recorded at combination time). The shared world model and lexicon are copied into each part.

### Experience Trie (T)
A compressed prefix tree over state transition sequences. The agent's long-term memory and world-model.

### Meta-Trie (M)
A second-level trie over snapshots of the agent's own trace buffer. The agent's self-model — it observes itself the way it observes the world.

### Self-Token / "I" Attractor (I)
The dominant state of the agent's self-observation chain (meta-trie). "I" locks when that chain has a clear, stable attractor: enough data, aperiodic, one state well above the uniform baseline, currently occupied. It unlocks if the attractor dissolves.

### Strange Loop (SL)
A count of self-reference in the agent's output sequence. SL=0 means no "I"; SL=1.0 means a depth-2 pattern ("I notice I familiar"). Because the agent's `core` utterance is fixed, a population's SL mostly reflects time spent in that output mode.

### Experience Lexicon (L)
The agent's vocabulary — a mapping from trace signatures (patterns of experience) to words. Each word points to a lived experience, not a dictionary definition.

### Attractor Lock
The moment the "I" locks (see Self-Token). In v3 it requires the lock criteria to hold for 3 consecutive self-observations. With `mathVersion: 'legacy'` the 2.x rule (max stationary probability > 0.25) applies.

### Trace Buffer
Short-term sliding window of the agent's last ~50 observations. The source material for meta-trie construction.

### Private Experience Space
Each agent has unique internal state IDs. Agent A's internal "state 42" has no relationship to Agent B's "state 42." This is faithful to Hoffman — experiences are private.

### Ablation
Running an experiment with a key mechanism disabled (e.g., self-reference turned off) to measure its causal contribution.

### Edge of Chaos
The regime between perfect order and perfect randomness. Often proposed as where complex dynamics are richest; not systematically measured in this library.

### Interface
A projection that compresses raw observations into fitness-relevant symbols. The CA's lexicon is an interface. Hoffman's claim: perception is an interface tuned for fitness, not accuracy.

### Markov World
A finite set of states with defined transition probabilities. Every CA needs a Markov world to inhabit. Any sequential data can be normalized into one.

### Ergodicity
A chain is ergodic when it is irreducible (every state reachable from every other) and aperiodic. Then long-run frequencies converge to a unique stationary distribution. The "I" lock requires the agent's self-chain to be ergodic on its recurrent class.

### Spectral Gap
1 − |λ₂|, where λ₂ is the second-largest eigenvalue (in modulus) of a transition matrix. It measures mixing speed: a small gap means slow mixing (long memory). It is 0 for chains with more than one closed class and for periodic chains. It is a classical property: a small gap is not a quantum signature (see example 02).

### Detailed Balance
π_i P_ij = π_j P_ji for all i, j: in the stationary regime every transition is balanced by its reverse. Chains that satisfy it are reversible (no arrow of time). `markov.irreversibility` measures the violation (0 = reversible, 1 = one-way). Violation is common in classical systems, such as any cycle, and is not a quantum signature.

### Ground Truth / Truth CA
A control condition where the agent sees the full state space (no compression). Used to compare against the Interface CA (which sees only a projection).

### Planck Probe / Planck Threshold
A hypothesis from the design document: a minimum network size below which geometric structure cannot form. An earlier "Planck probe" reported 100% lock at all sizes, but that was an artifact of the 2.x lock rule, which fired in any world. The hypothesis is untested with v3.

### Spectral Dimension
The exponent d_s in R(t) ~ t^(−d_s/2), where R(t) is a random walk's return probability. It reads 1, 2, 3 on lattices of that dimension, has no plateau on networks without geometry, and adds under independent combination. Used in experiment 14 (`markov.spectralDimension`).

### CHSH / Tsirelson Bound
The CHSH value of a Bell test is at most 2 for classical local systems, 2√2 (Tsirelson's bound) for quantum mechanics, and 4 for the PR box, the maximum without signalling. Used in experiment 13 (`analysis/bell`).

### Dobrushin Coefficient
δ(P) = max TV distance between two rows of P: how much information about the starting state survives one step. Experiment 13 proves CHSH ≤ 2 + 2δ(Qᵏ) for correlations passed through a hidden network.

### Trace Chain
What an observer that can only see a subset S of a network experiences: P_S = P_SS + P_SC (I − P_CC)⁻¹ P_CS. Traces of traces are traces, which gives the partial order behind Hoffman's trace logic. An observer's clock runs at π(S) ticks per network step. Used in experiment 15 (`trace.traceChain`).

### Decorated Permutation
A map σ: {1..n} → {1..2n} with a ≤ σ(a) ≤ a + n whose values mod n form a permutation. Decorated permutations index cells of the positive Grassmannian. Hoffman, Prakash & Prentner (2023) map each Markov chain to one, using only which states communicate and the cyclic order of their numbering. Used in experiment 16 (`decorated.decoratedPermutation`).

### Qualia Kernel / Trace Logic
The qualia kernel Q = D·A·P is an agent's experience-to-experience dynamics. In trace logic, agents are ordered by "Q_A is a trace of Q_B" ("A observes B"). The order is Boolean for the traces of one agent, is not a lattice globally, and maps into the Lebesgue logic of probability measures via the stationary distribution. Used in experiment 16.

### Entropy Rate / Commute Time
Entropy rate h = Σᵢ πᵢ H(Pᵢ) is the new information a chain produces per step. It adds under ⊗. Commute time κ(i, j) = E_i T_j + E_j T_i is a metric on states. Both are proposed readings of mass and of distance for speed, and are tested in experiment 16.

### Quantum Channel / Quantum Agent
A quantum channel maps density matrices by ρ ↦ Σ K ρ K† with Σ K†K = I. A quantum agent uses channels for P and A and measurements for D. A Markov kernel is the channel that erases all coherence. Used in experiment 17 (`quantum`).

### Sorkin Parameter (I₂, I₃)
Interference terms with routes blocked. I₂ ≠ 0 is two-route interference, which quantum amplitudes show and classical probabilities do not. I₃ = 0 for any theory with the Born rule. Used in experiment 17.

### Decoherence
Loss of off-diagonal (phase) terms of a density matrix, for example by dephasing. A fully decohered quantum agent is a Markov agent. Its Bell correlations fall to the classical bound and its interference vanishes.

### Proper Time / Time Dilation
The time a moving clock experiences, τ = √(t² − x²/c²). A clock moving at speed v runs slow by the factor √(1 − v²/c²). In experiment 18, the clock that counts an agent's own changes of experience converges to proper time as n → ∞.

### Telegraph Process / Feynman Checkerboard
A walker that moves at speed ±c and reverses at random (telegraph process) or with amplitude i·m·dt (Feynman's checkerboard, a quantum walk). Their continuum limits involve I₀(aτ) and J₀(mτ) of the proper time τ; the checkerboard gives the 1+1-dimensional Dirac propagator. Used in experiment 18.
