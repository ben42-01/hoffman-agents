# Experiment 14: Does Spacetime Emerge in the Headset?

## Question

Grant Hoffman's premise that space and time are an interface built from conscious-agent dynamics. **When does an observer's experience carry the structure of space (a dimension) and of time (an arrow), and where does that structure come from?**

## Instruments

- **Space: spectral dimension d_s.** A random walk returns to its starting point with probability R(t) ~ t^(−d_s/2). d_s is 1, 2 and 3 on lattices of that dimension, and it has no plateau on networks without geometry (expanders). It is the standard probe of emergent dimension in quantum-gravity models such as causal dynamical triangulations. The library API is `markov.spectralDimension` / `markov.spectral_dimension`.
- **Time: irreversibility.** This is the normalised net probability flux of the stationary chain. 0 means detailed balance holds (no arrow of time); 1 means every transition is one-way. API: `markov.irreversibility`.

## Parts and results (Node and Python print identical output)

**1. Calibration.** Ring → 0.98, 2-D torus → 2.02, 3-D torus → 3.12. On a random 4-regular graph the local dimension keeps rising (1.9 → 2.5 → 3.4), which means no geometry.

**2. Reconstruction through the headset.** A `ConsciousAgent` lives in a hidden lattice world but receives only opaque tokens: no coordinates and no neighbourhood information.

| Hidden world | d_s (true) | d_s (learned from experience) |
|---|---|---|
| 1-D ring, 64 sites | 0.95 | 0.95 |
| 2-D torus, 12×12 | 2.00 | 2.03 |
| 3-D torus, 8×8×8 | 2.70 | 2.67 |
| 2-D torus, 30% of sites aliased | 2.00 | 2.32 |

The kernel the agent learns from its experience trie carries the world's dimension. Small tori read below their true dimension because of finite size, so the comparison that matters is learned vs true. A coarser headset, where sites share tokens, distorts perceived space.

**3. Combination adds dimensions.** Ring-shaped conscious agents combined with ⊗ give d_s = 0.94, 1.99 and 3.10 for 1, 2 and 3 agents. This is exact for independent combination: the return probability of a product chain is the product of return probabilities, so spectral dimensions add. Three combined 1-D agents *are* a 3-D space. Combining with a geometry-free agent gives no plateau.

**4. Interaction binds dimensions and creates an arrow of time.** Three ring agents whose worlds are each other: agent i perceives agent i+1 and steps toward it with probability (1 + c)/2.

| coupling c | d_s | irreversibility |
|---|---|---|
| 0.0 | 2.82 | 0.000 |
| 0.3 | 2.55 | 0.170 |
| 0.6 | 1.74 | 0.293 |
| 0.9 | 1.06 | 0.410 |

Strongly coupled agents move as one, and only their shared centre diffuses: a 1-D space. The cyclic chase (1 → 2 → 3 → 1) produces net circulation, which is an arrow of time. These returns are measured from the stationary distribution, because coupled agents drift toward each other, and returns from transient states would measure that drift rather than geometry.

## Interpretation

In this model, **space in the headset is inherited, not created**:
- An observer recovers whatever geometry the agent dynamics has.
- ⊗ adds dimensions exactly.
- Interaction binds them.
- Networks without geometric structure give none.

Getting 3+1 dimensions requires specific agent dynamics: three independent one-dimensional directions, weakly coupled, plus a circulation that gives a time arrow. Nothing in the formalism selects that structure. That is the concrete open question for the "spacetime is a headset" view: *which principle, perhaps fitness, as in Hoffman's interface theory, would select it?*

## Run

```bash
npm run examples:spacetimeInTheHeadset                                          # Node, ~20 s
python examples/14_spacetime_in_the_headset/spacetime_in_the_headset.py         # Python, ~45 s
```
