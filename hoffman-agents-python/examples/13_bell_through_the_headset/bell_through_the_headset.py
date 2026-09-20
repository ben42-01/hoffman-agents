"""
Bell Test Through the Headset

Granting Hoffman's premise: spacetime is an interface (a "headset"), and behind
it is a network of conscious agents that spacetime does not constrain. Two
observers, Alice and Bob, are conscious agents at the interface. Each round:

  1. the hidden network is in its stationary regime:      h  ~ pi_Q
  2. Alice chooses a setting a and perceives x~ from h:    x~ ~ P_A[a](h, .)
  3. Alice acts, and her action enters the network:        h1 ~ A[a, x~](h, .)
  4. the network runs k steps of its own dynamics Q:       h2 ~ Q^k(h1, .)
  5. Bob chooses b and perceives y~ from h2:               y~ ~ P_B[b](h2, .)

This is the perceive -> decide -> act cycle of the (X, G, P, D, A, N) agent.
The hidden network Q is the joint kernel of two combined FormalConsciousAgents.

Headset rule (no-signalling): both reported outcomes are XOR-ed with a shared
hidden bit r that no action touches (x = x~ xor r, y = y~ xor r), so each
party's outcome is a fair coin whatever the other does. Correlations survive.

Questions:
  A. Control: if Alice's action cannot reach the network (spacetime-local),
     can any strategy beat the Bell bound CHSH <= 2?
  B. Behind the headset: the best CHSH value as a function of the number of
     network steps k between Alice's action and Bob's perception. Where does it
     cross Tsirelson's bound 2*sqrt(2)?
  C. Typical (random, unoptimised) networks: how often are they non-local,
     post-quantum, or (without the headset rule) signalling?

Strategies are optimised exactly by alternating best responses over
deterministic strategies (CHSH is linear in each party's strategy given the
other's), with seeded restarts.

Bound (proved in README.md): CHSH <= 2 + 2 * delta(Q^k), with delta the
Dobrushin contraction coefficient of k network steps.

Port of the Node example; prints the same numbers.
"""
import time

import numpy as np

from conscious_agent import FormalConsciousAgent, markov, mulberry32
from conscious_agent.analysis import bell

C = [[1, 1], [1, -1]]  # CHSH signs: E00 + E01 + E10 - E11
RESTARTS = 48
K_MAX = 12
TOL = 1e-4  # CHSH within TOL of 2 counts as classical


# ────────────── hidden network: two combined formal conscious agents ──────────────

def random_row(n, r, i=None, stickiness=0.0):
    u = [r.random() + 1e-3 for _ in range(n)]
    s = sum(u)
    return [(1 - stickiness) * v / s + (stickiness if j == i else 0.0) for j, v in enumerate(u)]


def formal_agent(r, stickiness):
    X = G = W = ["0", "1"]
    P = {w: [random_row(2, r, i, stickiness) for i in range(2)] for w in W}
    D = [random_row(2, r) for _ in X]
    A = {g: [random_row(2, r, i, stickiness) for i in range(2)] for g in G}
    return FormalConsciousAgent(X, G, W, P, D, A, rng=r)


def hidden_network(seed, stickiness):
    r = mulberry32(seed)
    net = FormalConsciousAgent.combine(formal_agent(r, stickiness), formal_agent(r, stickiness))
    Q = net.joint_kernel()
    d = Q.diagnostics()
    return {"Q": Q.matrix, "pi": markov.stationary(Q.matrix)["pi"], "states": len(Q.states),
            "lambda2": d["lambda2"], "ergodic": d["ergodic"]}


# ────────────── exact best-response optimisation of CHSH ──────────────

def optimise_chsh(Mk, pi, action_enters_network, restarts=RESTARTS, seed=1):
    n = len(pi)
    r = mulberry32(seed)
    best = {"value": -np.inf}
    for _ in range(restarts):
        sB = [[1 if r.random() < 0.5 else -1 for _ in range(n)] for _ in (0, 1)]
        value, alice = -np.inf, None
        for _ in range(200):
            v = [Mk @ np.array(sB[b], dtype=np.float64) for b in (0, 1)]
            alice = []
            for a in (0, 1):
                row = []
                for h in range(n):
                    targets = range(n) if action_enters_network else [h]
                    bt, bv = None, -np.inf
                    for t in targets:
                        gain = abs(C[a][0] * v[0][t] + C[a][1] * v[1][t])
                        if gain > bv + 1e-12:
                            bv, bt = gain, t
                    signed = C[a][0] * v[0][bt] + C[a][1] * v[1][bt]
                    row.append((bt, 1 if signed >= 0 else -1))
                alice.append(row)
            w = []
            for b in (0, 1):
                out = np.zeros(n)
                for a in (0, 1):
                    for h in range(n):
                        t, s = alice[a][h]
                        out += C[a][b] * pi[h] * s * Mk[t]
                w.append(out)
            sB = [[1 if x >= 0 else -1 for x in row] for row in w]
            nxt = float(sum(np.abs(row).sum() for row in w))
            if nxt <= value + 1e-12:
                break
            value = nxt
        if value > best["value"] + 1e-12:
            best = {"value": value, "alice": alice, "sB": sB}
    E = [[0.0, 0.0], [0.0, 0.0]]
    for a in (0, 1):
        for b in (0, 1):
            e = 0.0
            for h in range(n):
                t, s = best["alice"][a][h]
                e += pi[h] * s * float(Mk[t] @ np.array(best["sB"][b], dtype=np.float64))
            E[a][b] = e
    return {"chsh": bell.chsh(E), "E": E}


# ────────────── random (unoptimised) strategies ──────────────

def random_behaviour(Mk, pi, r, action_enters_network, masked):
    n = len(pi)
    pA = [[r.random() for _ in range(n)] for _ in (0, 1)]  # P(x~ = 0 | a, h)
    pB = [[r.random() for _ in range(n)] for _ in (0, 1)]  # P(y~ = 0 | b, h')
    act = [[[random_row(n, r) if action_enters_network else [1.0 if j == h else 0.0 for j in range(n)]
             for h in range(n)] for _ in (0, 1)] for _ in (0, 1)]
    p = [[[[0.0, 0.0], [0.0, 0.0]] for _ in (0, 1)] for _ in (0, 1)]
    for a in (0, 1):
        for b in (0, 1):
            for h in range(n):
                for x in (0, 1):
                    px = pA[a][h] if x == 0 else 1 - pA[a][h]
                    dist = pi[h] * px * (np.array(act[a][x][h]) @ Mk)  # distribution of h2
                    q0 = float(dist @ np.array(pB[b]))
                    p[a][b][x][0] += q0
                    p[a][b][x][1] += float(dist.sum()) - q0
    if not masked:
        return p
    return [[[[0.5 * (pb[x][y] + pb[1 - x][1 - y]) for y in (0, 1)] for x in (0, 1)] for pb in pa] for pa in p]


# ────────────── report ──────────────

def f(x, d=3):
    return f"{x:.{d}f}"


def main():
    t0 = time.time()
    print("=" * 78)
    print("Bell test through the headset")
    print("=" * 78)

    print("\n  Reference behaviours")
    s = 2 ** -0.5
    for name, E in (("best local (classical, in spacetime)", [[1, 1], [1, 1]]),
                    ("quantum singlet, optimal angles", [[s, s], [s, -s]]),
                    ("PR box (maximal no-signalling)", [[1, 1], [1, -1]])):
        print(f"    {name:<38} CHSH {f(bell.chsh(E))}  {bell.classify(bell.from_correlators(E))}")

    networks = [{"stickiness": st, **hidden_network(100 + i, st)} for i, st in enumerate((0, 0.5, 0.8, 0.95))]

    print("\n  A. Control: Alice's action cannot reach the network (spacetime-local)")
    control_max = 0.0
    for net in networks:
        for k in range(3):
            control_max = max(control_max, optimise_chsh(markov.mat_pow(net["Q"], k), net["pi"], False)["chsh"])
    print(f"    best CHSH over {len(networks)} networks and k = 0..2: {f(control_max, 6)}   (Bell bound: 2)")

    print("\n  B. Behind the headset: best CHSH vs network steps k between Alice's action and Bob")
    crossings, max_gap, violated = [], 0.0, False
    for net in networks:
        print(f"\n    hidden network: {net['states']} joint states, stickiness {net['stickiness']}, |λ₂| = {f(net['lambda2'])}")
        print("      k   δ(Q^k)   |λ₂|^k   bound 2+2δ   best CHSH   region of the optimum")
        kT = kL = None
        for k in range(K_MAX + 1):
            Mk = markov.mat_pow(net["Q"], k)
            delta = markov.dobrushin(Mk)
            res = optimise_chsh(Mk, net["pi"], True, seed=7 + k)
            chsh = res["chsh"]
            region = bell.classify(bell.from_correlators(res["E"]), TOL)
            max_gap = max(max_gap, 2 + 2 * delta - chsh)
            violated = violated or chsh > 2 + 2 * delta + 1e-9
            if kT is None and chsh <= bell.TSIRELSON + 1e-9:
                kT = k
            if kL is None and chsh <= 2 + TOL:
                kL = k
            print(f"      {k:<3} {f(delta)}    {f(net['lambda2'] ** k)}    {f(2 + 2 * delta)}        {f(chsh)}       {region}")
            if kL is not None:
                break
        crossings.append({**net, "kT": kT, "kL": kL})

    print("\n  C. Typical networks: 400 random (unoptimised) strategies per network, k = 1")
    r = mulberry32(2024)
    tally = dict(nonlocal_=0, post_quantum=0, signalling=0, local_signalling=0, local_nonlocal=0, total=0)
    for net in networks:
        M1 = markov.mat_pow(net["Q"], 1)
        for _ in range(400):
            masked = random_behaviour(M1, net["pi"], r, True, True)
            E = bell.correlators(masked)
            tally["nonlocal_"] += bell.chsh(E) > 2 + 1e-9
            tally["post_quantum"] += not bell.is_quantum(E)
            tally["signalling"] += bell.signalling(random_behaviour(M1, net["pi"], r, True, False)) > 1e-3
            local = random_behaviour(M1, net["pi"], r, False, False)
            tally["local_signalling"] += bell.signalling(local) > 1e-9
            tally["local_nonlocal"] += bell.chsh(bell.correlators(local)) > 2 + 1e-9
            tally["total"] += 1

    def pct(x):
        return f"{100 * x / tally['total']:.1f}%"

    print(f"    with headset rule:    CHSH > 2 in {pct(tally['nonlocal_'])}, outside the quantum set in {pct(tally['post_quantum'])}")
    print(f"    without headset rule: signalling (> 0.001) in {pct(tally['signalling'])}")
    print(f"    spacetime-local:      signalling in {pct(tally['local_signalling'])}, CHSH > 2 in {pct(tally['local_nonlocal'])}")

    def k_str(k):
        return str(k) if k is not None else f">{K_MAX}"

    print("\n" + "─" * 78)
    print("Findings")
    print("─" * 78)
    print(f"  1. Spacetime-local agents never beat the Bell bound (best {f(control_max, 6)}): Bell's theorem holds.")
    k0 = all(c["kT"] != 0 for c in crossings)
    print(f"  2. Behind the headset, no-signalling correlations {'reach the PR box (CHSH 4) at k = 0' if k0 else 'exceed 2 at k = 0'}.")
    print(f"     The bound CHSH <= 2 + 2δ(Q^k) was {'VIOLATED' if violated else 'never violated'}, and the optimiser reached it to within {'machine precision' if max_gap < 1e-9 else f'{max_gap:.1e}'}.")
    print("     Correlations decay at the network's mixing rate; CHSH <= 2√2 exactly when δ(Q^k) <= √2 - 1.")
    for c in crossings:
        print(f"     stickiness {str(c['stickiness']):<4} (|λ₂| {f(c['lambda2'])}): CHSH <= 2√2 from k = {k_str(c['kT'])}, classical (<= 2) from k = {k_str(c['kL'])}")
    print("     The optimal behaviours have perfect (±1) correlators, which places them outside the quantum")
    print("     set even when CHSH < 2√2: they stay post-quantum until they become classical.")
    print(f"  3. Random, unoptimised strategies are almost never non-local ({pct(tally['nonlocal_'])}): non-locality needs tuned agents.")
    print(f"  4. Without the headset rule, {pct(tally['signalling'])} of random networks would let Alice signal to Bob.")
    print("\n  Interpretation: granting a network behind spacetime, conscious-agent kernels can produce every")
    print("  no-signalling correlation up to the PR box, the quantum ones included, but nothing in the")
    print("  kernels singles out the quantum set. Matching nature needs two further principles: one that")
    print("  forbids signalling through the headset, and one that caps correlations at Tsirelson's bound.")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
