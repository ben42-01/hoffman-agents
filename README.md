# conscious-agent

**An executable model of Donald Hoffman's conscious-agent formalism, in Node.js and Python.**

Hoffman & Prakash define a conscious agent as a six-tuple (X, G, P, D, A, N) of Markov kernels: perception, decision and action acting on a world made, ultimately, of other agents. This library implements that formalism exactly, and adds learning agents that build their kernels from experience. With it you can compute what the theory implies and test it against controls.

- **`FormalConsciousAgent`**: the six-tuple as explicit kernels, with the joint dynamics on experience × world, stationary distributions, ergodicity, mixing, and combination (⊗).
- **`ConsciousAgent`**: an agent that learns a world model (experience trie) and a model of its own dynamics (meta-trie), and forms an "I" when that self-model has a stable attractor.
- **Analysis**: ergodic diagnostics, trace chains and trace logic (what an observer with a partial view experiences), decorated permutations, spectral dimension, irreversibility (arrow of time), Bell/CHSH classification.
- **Quantum agents**: kernels as quantum channels on qubits, with exact CHSH (Horodecki), decoherence and a Markov-kernel embedding.
- **Identical in both languages**: seeded runs produce the same numbers in Node and Python, and the test suites check this step by step.

## What it has shown

Each result below has a control and is covered by the tests.

| Experiment | Finding |
|---|---|
| [13 Bell test through the headset](hoffman-agents-node/examples/13_bell_through_the_headset/) | Grant a network of agents behind spacetime. If an observer's action can reach it, the best Bell (CHSH) value is **exactly 2 + 2·δ(Qᵏ)**, where δ is the network's contraction coefficient after k steps (proved; attained numerically). Correlations reach the PR box (4) at zero network distance and become classical as the network mixes. The kernels do not single out quantum correlations, and without an extra no-signalling rule 98% of random networks would signal. |
| [14 Spacetime in the headset](hoffman-agents-node/examples/14_spacetime_in_the_headset/) | An agent that sees only opaque symbols recovers its hidden world's dimension from experience. Under ⊗, dimension adds exactly (three combined 1-D agents form a 3-D space). Interaction binds dimensions (3 → 1) and creates an arrow of time. Geometry in the headset is inherited from agent dynamics, not generated. |
| [15 Time in the traces](hoffman-agents-node/examples/15_time_in_the_traces/) | Following Hoffman's trace logic: an observer who sees only part of the network experiences exactly the trace chain of its window. Each observer's clock runs at its own rate, and nested clocks compose exactly. The arrow of time is observer-dependent: it appears from observing even when the dynamics is reversible, and a two-state window never perceives one. Experiment 18 takes these clocks to the n → ∞ limit. |
| [16 Trace logic and decorated permutations](hoffman-agents-node/examples/16_trace_logic_and_decorated_permutations/) | Hoffman's map from Markov chains to decorated permutations (the combinatorics behind the amplituhedron) agrees with all 27 entries of the paper's table: 14 permutations for 3-state agents, not the 17 its text states. It discards all probabilities and depends on how experiences are numbered, so a route to particle physics needs a principle that orders experiences. Trace logic is locally Boolean, globally not a lattice, and maps into Lebesgue logic, as claimed. |
| [17 Quantum agents](hoffman-agents-node/examples/17_quantum_agents/) | With complex amplitudes (quantum channels), combined agents reach the quantum limit **exactly 2√2** and never exceed it, and they cannot signal. The Hilbert-space structure does this work, not the agent formalism. Markov agents are the fully decohered quantum agents: decoherence brings CHSH down to exactly 2 and removes interference. |
| [18 Relativity at infinity](hoffman-agents-node/examples/18_relativity_at_infinity/) | Hoffman conjectures that relativity emerges from agent chains as n → ∞. For an agent with one bit of memory (its direction), in the limit: a light cone appears, and the clock that counts its own changes of experience runs at **Einstein's proper time** (speed 0.6 → 0.800, speed 0.8 → 0.600). Every finite network keeps a preferred frame (errors ∝ 1/n). Full Lorentz covariance (the Dirac propagator) needs complex amplitudes. The mathematics is known (Kac, Feynman's checkerboard); the model is 1+1-dimensional. |
| [12 Ergodic diagnostics](hoffman-agents-node/examples/12_ergodic_diagnostics/), [05 ablation](hoffman-agents-node/examples/05_self_ref_ablation/) | The "I" lock fires where experience has a stable attractor and never in structureless worlds (0 of 50). |
| [02 Quantum signature?](hoffman-agents-node/examples/02_quantum_signature/) | An earlier "quantum-like" spectral signature was an artifact; a classical clock meets the same criterion. |
| [07](hoffman-agents-node/examples/07_exchange_symmetry/), [08](hoffman-agents-node/examples/08_observer_gated_combination/), [09](hoffman-agents-node/examples/09_double_slit_analogy/) | Combination is exchange-symmetric by construction; observer-gated combination produces order-dependence; classical probability cannot produce interference. |

**What it has not shown.** Nothing here is evidence that the agents are conscious, that nature is made of conscious agents, or that quantum mechanics or spacetime emerges from them (the quantum agents of experiment 17 have quantum mechanics built in). The experiments turn parts of the theory into precise, testable statements, and in several cases identify what the theory would additionally need. Claims made by earlier versions of this project that did not hold up are listed under [Corrections](docs/Q_AND_A.md#corrections).

## Packages

| Package | Install | Docs |
|---|---|---|
| Node.js (≥ 18, no dependencies) | `npm install github:ben42-01/hoffman-agents` | [hoffman-agents-node/README.md](hoffman-agents-node/README.md) |
| Python (≥ 3.10, numpy, scipy) | `pip install "git+https://github.com/ben42-01/hoffman-agents.git@main#subdirectory=hoffman-agents-python"` | [hoffman-agents-python/README.md](hoffman-agents-python/README.md) |

The React Native package (`hoffman-agents-react-native`) still implements the 2.x math.

## Quick start

```javascript
const { ConsciousAgent, FormalConsciousAgent } = require('conscious-agent');
const { CoinTossWorld } = require('conscious-agent/worlds');

const agent = new ConsciousAgent({ agentId: 'a', seed: 1, world: new CoinTossWorld(4) });
agent.run(2000);
console.log(agent.ergodicStats().lock);           // lock state, criteria, lock/unlock history

const c = new FormalConsciousAgent({
  X: ['calm', 'alert'], G: ['stay', 'move'], W: ['left', 'right'],
  P: { left: [[0.9, 0.1], [0.6, 0.4]], right: [[0.3, 0.7], [0.1, 0.9]] },
  D: [[0.8, 0.2], [0.2, 0.8]],
  A: { stay: [[1, 0], [0, 1]], move: [[0, 1], [1, 0]] },
});
console.log(c.diagnostics().stationary);          // long-run distribution on X × W
```

```python
from conscious_agent import ConsciousAgent
from conscious_agent.worlds import CoinTossWorld

agent = ConsciousAgent(agent_id="a", seed=1, world=CoinTossWorld(n_coins=4))
agent.run(2000)
print(agent.ergodic_stats()["lock"])
```

## Documentation

- [MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md): the precise mathematics (kernels, ergodic analysis, the "I" lock, combination, parity)
- [Q_AND_A.md](docs/Q_AND_A.md): results, limitations and corrections
- [COMPONENT_DEFINITIONS.md](docs/COMPONENT_DEFINITIONS.md): components and data flow
- [CONSCIOUS_AGENTS_VISUAL_GUIDE.md](docs/CONSCIOUS_AGENTS_VISUAL_GUIDE.md): diagrams mapping the formalism to code
- [SELF_AWARENESS.md](docs/SELF_AWARENESS.md): the self-modelling mechanisms and what they do and do not show
- [GLOSSARY.md](docs/GLOSSARY.md)
- [CONSCIOUS_AGENTS_THEORY.md](docs/CONSCIOUS_AGENTS_THEORY.md): the founding design document (hypotheses, pre-implementation)
- [CA_RUNTIME_API.md](docs/CA_RUNTIME_API.md): the original API design specification (partly unimplemented)

## References

- Hoffman, D. D. & Prakash, C. (2014). Objects of consciousness. *Frontiers in Psychology*, 5, 577.
- Hoffman, D. D., Singh, M. & Prakash, C. (2015). The interface theory of perception. *Psychonomic Bulletin & Review*, 22, 1480–1506.
- Fields, C., Hoffman, D. D., Prakash, C. & Singh, M. (2018). Conscious agent networks: Formal analysis and application to cognition. *Cognitive Systems Research*, 47, 186–213.
- Hoffman, D. D., Prakash, C. & Prentner, R. (2023). Fusions of consciousness. *Entropy*, 25(1), 129.
- Bennett, B. M., Hoffman, D. D. & Murthy, P. (1993). Lebesgue logic for probabilistic reasoning and some applications to perception. *Journal of Mathematical Psychology*, 37(1), 63–103.
- Horodecki, R., Horodecki, P. & Horodecki, M. (1995). Violating Bell inequality by mixed spin-½ states: necessary and sufficient condition. *Physics Letters A*, 200, 340–344.
- Sorkin, R. D. (1994). Quantum mechanics as quantum measure theory. *Modern Physics Letters A*, 9, 3119–3127.

## License

MIT
