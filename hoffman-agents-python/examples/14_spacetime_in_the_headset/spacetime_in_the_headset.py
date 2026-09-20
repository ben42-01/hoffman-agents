"""
Does Spacetime Emerge in the Headset?

Granting Hoffman's premise that space and time are an interface built from
conscious-agent dynamics, this experiment asks when an observer's experience
carries the structure of space (a dimension) and of time (an arrow).

Ruler: the spectral dimension d_s. A random walk returns to its start with
probability R(t) ~ t^(-d_s/2); d_s = 1, 2, 3 on lattices of that dimension, it
has no plateau on networks without geometry (expanders), and it is the probe
used for emergent dimension in quantum-gravity models (causal dynamical
triangulations). Arrow of time: irreversibility, the normalised net probability
flux (0 = detailed balance, no arrow).

  1. Calibration: the ruler on lattices of known dimension and on an expander.
  2. Reconstruction: an agent living in a hidden lattice world sees only opaque
     symbols. Does the kernel it learns from experience carry the dimension?
  3. Combination: conscious agents whose experience is a 1-D ring are combined
     with (x). How does dimension behave under combination?
  4. Interaction: three ring agents whose worlds are each other (agent i
     perceives agent i+1 and drifts toward it). Does interaction keep, create or
     destroy dimensions, and does an arrow of time appear?

Port of the Node example.
"""
import time

import numpy as np

from conscious_agent import ConsciousAgent, FormalConsciousAgent, MarkovKernel, WorldState, markov, mulberry32


# ────────────── kernels ──────────────

def torus(L, d):
    """Simple random walk on a d-dimensional torus of side L (mixed-radix index)."""
    n = L ** d
    P = np.zeros((n, n))
    for i in range(n):
        stride = 1
        for _ in range(d):
            c = (i // stride) % L
            for s in (1, -1):
                P[i, i + (((c + s) % L) - c) * stride] += 1 / (2 * d)
            stride *= L
    return P


def expander(n, seed):
    """Random 4-regular multigraph (two random perfect matchings): no geometry."""
    r = mulberry32(seed)
    P = np.zeros((n, n))
    for _ in range(2):
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(r.random() * (i + 1))
            p[i], p[j] = p[j], p[i]
        for i in range(n):
            P[i, p[i]] += 0.25
            P[p[i], i] += 0.25
    return P


def ring_agent(m):
    """A conscious agent whose experience space X is a ring of m states."""
    ring = torus(m, 1).tolist()
    return FormalConsciousAgent([str(i) for i in range(m)], ["·"], ["·"], {"·": ring}, [[1.0] for _ in ring],
                                {"·": [[1.0]]})


def coupled_rings(m, c):
    """Three ring agents, each perceiving the next: agent i steps toward agent
    i+1 with probability (1 + c)/2 (lazy: it stays put half the time)."""
    n = m ** 3
    Q = np.zeros((n, n))

    def step(x, w):
        d = (w - x) % m
        toward = 0 if d == 0 else (1 if d <= m / 2 else -1)
        return [(x, 0.5), ((x + 1) % m, 0.25 * (1 + c * toward)), ((x - 1) % m, 0.25 * (1 - c * toward))]

    for i in range(n):
        x = [i % m, (i // m) % m, i // (m * m)]
        moves = [step(x[k], x[(k + 1) % 3]) for k in range(3)]
        for a, pa in moves[0]:
            for b, pb in moves[1]:
                for d, pd in moves[2]:
                    Q[i, a + m * b + m * m * d] += pa * pb * pd
    return Q


# ────────────── measurements ──────────────

def f(x, d=2):
    return f"{x:.{d}f}" if np.isfinite(x) else str(x)


def measure(P, window, **options):
    s = markov.spectral_dimension(P, t_min=window[0], t_max=window[1], **options)
    return s


def local_str(loc):
    return " ".join(f"{l['t']}:{f(l['d'])}" for l in loc)


def learned_kernel(L, d, steps, seed, aliasing=0.0):
    """The agent sees each lattice site only as an opaque token; with aliasing > 0
    some pairs of sites share a token."""
    P = torus(L, d)
    n = len(P)
    r = mulberry32(seed)
    token = [f"q{i * 7919 % 100003}" for i in range(n)]
    if aliasing > 0:
        order = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(r.random() * (i + 1))
            order[i], order[j] = order[j], order[i]
        for k in range(int(aliasing * n / 2)):
            token[order[2 * k + 1]] = token[order[2 * k]]

    class HiddenLattice:
        site = 0

        def step(self):
            u, acc = r.random(), 0.0
            for j in range(n):
                acc += P[self.site, j]
                if u < acc:
                    self.site = j
                    break
            return WorldState.from_sequence("world", [token[self.site]])

    agent = ConsciousAgent(agent_id=f"observer_{d}d", seed=seed, world=HiddenLattice())
    agent.run(steps)
    K = agent.to_formal()["P"]
    return K, P


def main():
    t0 = time.time()
    print("=" * 78)
    print("Does spacetime emerge in the headset?")
    print("=" * 78)

    print("\n  1. Calibrating the ruler (spectral dimension; local values at t = 2, 4, 8, ...)")
    for name, P, w in (("ring, 200 sites (d = 1)", torus(200, 1), (4, 64)),
                       ("torus 30×30 (d = 2)", torus(30, 2), (4, 64)),
                       ("torus 16×16×16 (d = 3)", torus(16, 3), (4, 32)),
                       ("random 4-regular graph, 400 nodes", expander(400, 1), (2, 8))):
        m = measure(P, w)
        print(f"    {name:<36} d_s = {f(m['dimension'])}   local {local_str(m['local'])}")
    print("    → lattices read 1, 2, 3; the expander has no plateau (its local dimension keeps rising).")

    print("\n  2. Reconstruction: an agent sees only opaque tokens from a hidden lattice")
    print("    (small worlds read below their dimension; what matters is learned vs true)")
    print(f"    {'hidden world':<32} {'steps':<7} {'states learned':<15} d_s(true)  d_s(learned)")
    recon = []
    for L, d, w, aliasing in ((64, 1, (2, 16), 0), (12, 2, (2, 12), 0), (8, 3, (2, 8), 0), (12, 2, (2, 12), 0.3)):
        steps = 60 * L ** d
        K, truth = learned_kernel(L, d, steps, seed=10 + d, aliasing=aliasing)
        dt, dl = measure(truth, w)["dimension"], measure(K.matrix, w)["dimension"]
        recon.append({"d": d, "aliasing": aliasing, "dt": dt, "dl": dl})
        label = f"{d}-D torus, side {L}" + (f", {aliasing * 100:g}% aliased" if aliasing else "")
        print(f"    {label:<32} {steps:<7} {len(K.states):<15} {f(dt):<10} {f(dl)}")

    print("\n  3. Combination: ring agents (X = ring of 10 experiences) combined with ⊗")
    comb = []
    agent = ring_agent(10)
    for count in (1, 2, 3):
        if count > 1:
            agent = FormalConsciousAgent.combine(agent, ring_agent(10))
        Q = agent.joint_kernel().matrix
        m = measure(Q, (2, 10))
        comb.append({"count": count, "d": m["dimension"]})
        label = f"{count} agent{'s' if count > 1 else ''} combined"
        print(f"    {label:<22} {len(Q):<6} experiences   d_s = {f(m['dimension'])}   local {local_str(m['local'])}")
    mixed = MarkovKernel([str(i) for i in range(10)], torus(10, 1)).tensor(
        MarkovKernel([str(i) for i in range(100)], expander(100, 3)))
    mm = measure(mixed.matrix, (2, 10))
    print(f"    {'ring ⊗ geometry-free':<22} {len(mixed.states):<6} experiences   d_s = {f(mm['dimension'])}   local {local_str(mm['local'])}")

    print("\n  4. Interaction: three ring agents, each perceiving the next (coupling c)")
    print("    (returns measured from the stationary distribution: coupled agents drift toward each other,")
    print("     and returns from transient states would measure that drift rather than geometry)")
    print("    c      d_s     local                              irreversibility (arrow of time)")
    inter = []
    for c in (0, 0.3, 0.6, 0.9):
        Q = coupled_rings(10, c)
        pi = markov.stationary(Q)["pi"]
        m = measure(Q, (2, 10), start_distribution="stationary", pi=pi, max_starts=128, lazy=False)
        irr = markov.irreversibility(Q, pi)
        inter.append({"c": c, "d": m["dimension"], "irr": irr})
        print(f"    {f(c, 1):<6} {f(m['dimension']):<7} {local_str(m['local']):<34} {f(irr, 3)}")

    print("\n" + "─" * 78)
    print("Findings")
    print("─" * 78)
    worst = max(abs(r["dt"] - r["dl"]) for r in recon if not r["aliasing"])
    aliased = next(r for r in recon if r["aliasing"])
    print("  1. From opaque symbols alone, the agent's learned kernel carries its world's dimension")
    print(f"     (largest |d_s(learned) − d_s(true)| = {f(worst)}). Aliasing {aliased['aliasing'] * 100:g}% of the sites shifts it")
    print(f"     from {f(aliased['dt'])} to {f(aliased['dl'])}: a coarser headset distorts perceived space.")
    adds = ", ".join(f"{c['count']} → {f(c['d'])}" for c in comb)
    print(f"  2. Under ⊗, dimension adds: {adds}.")
    print("     This is exact for independent combination (return probabilities multiply), so three")
    print("     combined 1-D agents are a 3-D space. Combining with a geometry-free agent gives no plateau.")
    c0, c_max = inter[0], inter[-1]
    binds = "binds dimensions" if c_max["d"] < c0["d"] - 0.3 else "changes dimension little"
    chain = " → ".join(f"{f(x['d'])} (c = {x['c']:g})" for x in inter)
    print(f"  3. Interaction {binds}: d_s {chain}.")
    print("     Strongly coupled agents move as one: only their shared centre diffuses, a 1-D space.")
    print(f"     An arrow of time appears with the cyclic coupling: irreversibility {f(c0['irr'], 3)} → {f(c_max['irr'], 3)}.")
    print("\n  Interpretation: in this model, space in the headset is inherited, not created. The observer")
    print("  recovers whatever geometry the agent dynamics has; ⊗ adds dimensions exactly; interaction")
    print("  binds them; networks without geometric structure give none. Getting 3+1 dimensions therefore")
    print("  requires specific agent dynamics (three independent 1-D directions, weak coupling). Nothing")
    print('  here selects that structure, which is the open question for the "spacetime is a headset" view.')
    print(f"\n  Done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
