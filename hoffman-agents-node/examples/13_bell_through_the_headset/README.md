# Experiment 13: Bell Test Through the Headset

## Question

Grant Hoffman's premise: spacetime is an interface (a "headset"), and behind it is a network of conscious agents that spacetime does not constrain. Two observers at the interface run a Bell (CHSH) test. **Which correlations can such a network produce, and does it single out the ones nature shows?**

The reference points:

| Behaviour | CHSH | Status in nature |
|---|---|---|
| Classical, local in spacetime | ≤ 2 | Bell bound, violated by experiment |
| Quantum mechanics | ≤ 2√2 ≈ 2.83 | Tsirelson's bound, observed |
| PR box | 4 | Never observed, though it doesn't allow signalling |
| Signalling | any | Forbidden (no faster-than-light messages) |

## Model

Each round follows the perceive → decide → act cycle of the (X, G, P, D, A, N) agent:

1. The hidden network, the joint kernel **Q** of two combined `FormalConsciousAgent`s, is in its stationary regime: `h ~ π_Q`.
2. Alice picks a setting `a` and perceives `x̃ ~ P_A[a](h, ·)`.
3. Alice acts, and her action enters the network: `h₁ ~ A[a, x̃](h, ·)`.
4. The network runs `k` steps of its own dynamics: `h₂ ~ Qᵏ(h₁, ·)`.
5. Bob picks `b` and perceives `ỹ ~ P_B[b](h₂, ·)`.

**Headset rule.** Both reported outcomes are XOR-ed with a shared hidden bit `r` that no action touches: `x = x̃ ⊕ r`, `y = ỹ ⊕ r`. Each party's outcome is then a fair coin whatever the other does, so nothing can be signalled through the headset, while correlations survive.

`k` is the network distance between Alice's action and Bob's perception. It is measured in steps of the network's own dynamics, not in spacetime.

Strategies are optimised exactly. CHSH is linear in each party's strategy given the other's, so alternating best responses over deterministic strategies climbs monotonically. The experiment uses 48 seeded restarts.

## Result: the network-distance bound

For this model class:

**CHSH ≤ 2 + 2·δ(Qᵏ)**

Here δ(M) = maxᵢ,ⱼ TV(Mᵢ., Mⱼ.) is the Dobrushin contraction coefficient: how much a network state can still be told apart after k steps.

**Proof.** Let `g_b(h') = E[(−1)^ỹ | b, h'] ∈ [−1, 1]` be Bob's strategy. For a hidden state `h`, write `μ_a` for the distribution of `h₂` after Alice's action for setting `a` (a mixture of rows of Qᵏ), and `s_a = ±1` for her outcome sign. By convexity, deterministic outcome signs suffice. The contribution of `h` to E₀₀ + E₀₁ + E₁₀ − E₁₁ is

  s₀ μ₀·(g₀ + g₁) + s₁ μ₁·(g₀ − g₁).

Split μ₀ = m + α and μ₁ = m + β, where m = min(μ₀, μ₁) has mass 1 − τ, and α, β are non-negative with mass τ = TV(μ₀, μ₁) each. Pointwise, |s₀(g₀+g₁) + s₁(g₀−g₁)| ≤ |g₀+g₁| + |g₀−g₁| = 2·max(|g₀|, |g₁|) ≤ 2, and |g₀ ± g₁| ≤ 2. So the contribution is at most 2(1 − τ) + 2τ + 2τ = 2 + 2τ.

The total variation between mixtures of rows is at most the largest total variation between rows, so τ ≤ δ(Qᵏ). Averaging over h ~ π gives CHSH ≤ 2 + 2δ(Qᵏ). The other CHSH variants follow by relabelling. ∎

In every network and every k tested, the optimiser **attains** the bound to machine precision. The unit tests (`test/analysis.test.js`, `tests/test_analysis.py`) check both the bound and its attainment.

Since δ(Qᵏ) → 0 at the network's mixing rate (≈ |λ₂|ᵏ), non-local correlations fade with network distance. **CHSH ≤ 2√2 exactly when δ(Qᵏ) ≤ √2 − 1.**

## Measured (seeds fixed; Node and Python print identical output)

| Hidden network | \|λ₂\| | CHSH at k = 0 | CHSH ≤ 2√2 from | classical (≤ 2) from |
|---|---|---|---|---|
| stickiness 0 | 0.256 | 4.000 | k = 2 | k = 8 |
| stickiness 0.5 | 0.706 | 4.000 | k = 4 | k > 12 |
| stickiness 0.8 | 0.827 | 4.000 | k = 6 | k > 12 |
| stickiness 0.95 | 0.967 | 4.000 | k > 12 | k > 12 |

- **Control.** When Alice's action cannot reach the network (spacetime-local), the best CHSH is exactly 2.000000. Bell's theorem holds, as it must.
- **Optimal behaviours.** They have perfect ±1 correlators, which places them *outside the quantum set* (Tsirelson–Landau–Masanes test) even when CHSH < 2√2. They are post-quantum until they become classical.
- **Random strategies.** Random, unoptimised strategies are essentially never non-local (0.0%): non-locality has to be tuned.
- **Signalling.** Without the headset rule, 98.2% of random networks let Alice signal to Bob.

## Interpretation

Granting a network behind spacetime, conscious-agent kernels can produce **every no-signalling correlation up to the PR box**, quantum ones included. But nothing in the kernels singles out the quantum set. To match nature the theory needs two further principles:

1. **No signalling through the headset.** Here it was imposed as the XOR rule. Without it, signalling is generic.
2. **A cap at Tsirelson's bound.** In this model only network distance provides one, and it gives the wrong *shape*: nature's correlations are quantum at all distances, not post-quantum up close and classical far away.

This is a constraint on the theory, not a refutation. It says which extra ingredients a conscious-agent account of quantum correlations would need.

**Context.** It is known in quantum foundations that hidden classical communication can reproduce any no-signalling correlation (Popescu & Rohrlich 1994; Toner & Bacon 2003 for the quantum singlet). What this experiment adds is the explicit bound in terms of the agent network's mixing, and the observation that the conscious-agent formalism falls into this class.

## Run

```bash
npm run examples:bellThroughTheHeadset                                          # Node
python examples/13_bell_through_the_headset/bell_through_the_headset.py         # Python
```
