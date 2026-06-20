# Glossary of Terms

For a non-specialist joining the project.

---

### Agent (Conscious Agent / CA)
A computational object implementing Hoffman's six-tuple `(X, G, P, W, A, D)`. It has private experiences, a world made of other agents' outputs, and a stable identity ("I").

### Experience Space (X)
The set of all possible experiences an agent can have. Private to that agent. Represented computationally as a tuple of four structures: T (world model), M (self-model), I (identity attractor), L (lexicon).

### Action Space (G)
The set of possible outputs an agent can produce. What it broadcasts to other agents.

### World Space (W)
What the agent perceives as "the world." In this framework, W is always the output of other agents — never a fixed external environment.

### Perceptual Map (P)
The function `P: W × X → X` — how the agent's experience changes when it observes the world.

### Decision Map (D)
The function `D: X → G` — how the agent's experience produces its output.

### Combination Operator (⊗)
The operation that merges two agents into a higher-order agent. Satisfies associativity, non-commutativity, and identity properties.

### Experience Trie (T)
A compressed prefix tree over state transition sequences. The agent's long-term memory and world-model.

### Meta-Trie (M)
A second-level trie over snapshots of the agent's own trace buffer. The agent's self-model — it observes itself the way it observes the world.

### Self-Token / "I" Attractor (I)
The dominant meta-state in the agent's meta-trie stationary distribution. When this stabilizes above a threshold, the agent has a locked identity.

### Strange Loop (SL)
A measurement of self-referential depth in the agent's output sequence. SL=0 means no self-reference. SL=1.0 means a depth-2 loop ("I notice I familiar"). SL>1.0 means deeper recursion.

### Experience Lexicon (L)
The agent's vocabulary — a mapping from trace signatures (patterns of experience) to words. Each word points to a lived experience, not a dictionary definition.

### Attractor Lock
The phase transition when an agent's "I" identity stabilizes. Occurs when a single meta-state maintains stationary probability > threshold for 3 consecutive checks.

### Trace Buffer
Short-term sliding window of the agent's last ~50 observations. The source material for meta-trie construction.

### Private Experience Space
Each agent has unique internal state IDs. Agent A's internal "state 42" has no relationship to Agent B's "state 42." This is faithful to Hoffman — experiences are private.

### Ablation
Running an experiment with a key mechanism disabled (e.g., self-reference turned off) to measure its causal contribution.

### Edge of Chaos
The regime between perfect order and perfect randomness where structure is maximal. In this framework, I-lock peaks at the edge of chaos.

### Interface
A projection that compresses raw observations into fitness-relevant symbols. The CA's lexicon is an interface. Hoffman's claim: perception is an interface tuned for fitness, not accuracy.

### Markov World
A finite set of states with defined transition probabilities. Every CA needs a Markov world to inhabit. Any sequential data can be normalized into one.

### Ergodicity
Every state reachable from every other state. Required for the agent to build a meaningful world-model.

### Spectral Gap
The difference between the largest (1.0) and second-largest eigenvalue of a transition matrix. Small spectral gap = near-degenerate eigenvalues, associated with quantum-like dynamics.

### Detailed Balance
A property of reversible Markov chains: the probability of transitioning from i to j equals that of j to i. Breaks when combination produces irreversible dynamics.

### Ground Truth / Truth CA
A control condition where the agent sees the full state space (no compression). Used to compare against the Interface CA (which sees only a projection).

### Planck Probe / Planck Threshold
An experiment testing whether there is a minimum network size below which structure cannot form. The Planck probe found no threshold up to 1024 agents — lock rate is 100% at all sizes.
