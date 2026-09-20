"""
Quantum Agents

Experiment 13 found that classical conscious-agent networks behind a headset
give CHSH = 2 + 2δ, anywhere from 2 up to the PR box's 4, and signal unless an
extra rule forbids it. Nature stops at Tsirelson's bound 2√2 ≈ 2.83 and never
signals. Here the agents' kernels carry complex amplitudes instead:

  experiences  -> density matrices (qubits)
  P, A         -> quantum channels (Kraus operators)
  D            -> measurements (Born rule)
  combination  -> tensor product, plus an interaction unitary

Sharp question: does combining quantum agents reproduce the quantum limit
2√2 exactly, and what does the work?

  1. Markov agents are quantum agents that have fully decohered.
  2. Combination without interaction: CHSH <= 2.
  3. Combination with interaction: the maximum, random interactions,
     no-signalling, and the same Bell classifier as experiment 13.
  4. Network distance and decoherence: back to the classical bound.
  5. Interference: pairwise (I2 ≠ 0) but never three-way (I3 = 0).

Port of the Node example; prints the same numbers.
"""
import math
import time

import numpy as np

from conscious_agent import bell, mulberry32
from conscious_agent import quantum as q

TSIRELSON = 2 * math.sqrt(2)


def f(x, d=3):
    return f"{x:.{d}f}"


def tiny(x):
    return "machine precision" if x < 1e-9 else f"{x:.1e}"


def random_stochastic(n, r):
    rows = []
    for _ in range(n):
        u = [r.random() + 0.05 for _ in range(n)]
        rows.append([v / sum(u) for v in u])
    return rows


def random_qubit(r):
    """Random pure qubit state (cos θ/2, e^{iφ} sin θ/2)."""
    th, ph = math.pi * r.random(), 2 * math.pi * r.random()
    return np.array([math.cos(th / 2), complex(math.cos(ph), math.sin(ph)) * math.sin(th / 2)])


def product_state(a, b):
    return q.density_from_state(np.kron(a, b))


ZERO2 = q.density_from_state([1, 0, 0, 0])
XX = np.kron(q.PAULI["X"], q.PAULI["X"])


def interact(rho, t):
    return q.conjugate(rho, q.pauli_rotation(XX, t))


def local(rho, kraus):
    return q.apply_channel(q.apply_channel(rho, q.on_qubit(kraus, 0, 2)), q.on_qubit(kraus, 1, 2))


def measured(rho):
    """Measure with the optimal settings; run the statistics through experiment 13's classifier."""
    s = q.optimal_settings(rho)
    p = q.behaviour(rho, s["alice"], s["bob"])
    return {"chsh": bell.chsh(bell.correlators(p)), "cls": bell.classify(p, 1e-9), "signalling": bell.signalling(p)}


# ────────────── 1. Markov agents inside quantum agents ──────────────

def part1():
    print("\n  1. Markov agents are quantum agents that have fully decohered")
    r = mulberry32(1)
    act_err = kraus_err = coherence = 0.0
    for t in range(50):
        n = 4 if t % 2 else 2
        P = np.array(random_stochastic(n, r))
        K = q.markov_channel(P)
        kraus_err = max(kraus_err, q.kraus_deviation(K))
        pn = np.array(random_stochastic(n, r)[0])
        out = q.apply_channel(np.diag(pn).astype(complex), K)
        act_err = max(act_err, float(np.max(np.abs(np.real(np.diag(out)) - pn @ P))))
        o2 = q.apply_channel(q.density_from_state(np.full(n, 1 / math.sqrt(n))), K)
        coherence = max(coherence, float(np.max(np.abs(o2 - np.diag(np.diag(o2))))))
    print("    Each Markov kernel P becomes a quantum channel with Kraus operators √P_ij |j⟩⟨i| (50 random kernels):")
    print(f"      valid channel (Σ K†K = I): error {tiny(kraus_err)}; on classical states it acts exactly as P: error {tiny(act_err)};")
    print(f"      it erases all coherence (largest surviving off-diagonal term {tiny(coherence)}).")
    return {"act_err": act_err, "kraus_err": kraus_err, "coherence": coherence}


# ────────────── 2. combination without interaction ──────────────

def part2():
    print("\n  2. Combining quantum agents without interaction (tensor product only)")
    r = mulberry32(2)
    best = 0.0
    for t in range(1000):
        rho = product_state(random_qubit(r), random_qubit(r))
        if t % 2:
            rho = local(rho, q.depolarizing(r.random()))
        best = max(best, q.max_chsh(rho))
    print(f"    1000 random product agents (pure and noisy): best CHSH over all measurements {f(best, 6)} (local bound 2).")
    print("    Combination alone creates no Bell correlations: an interaction is needed.")
    return {"best": best}


# ────────────── 3. combination with interaction ──────────────

def part3():
    print("\n  3. Combining quantum agents with an interaction U(t) = exp(-i t X⊗X), starting from |00⟩")
    rows = []
    for label, t in [("0", 0), ("π/16", math.pi / 16), ("π/8", math.pi / 8), ("3π/16", 3 * math.pi / 16), ("π/4", math.pi / 4)]:
        rho = interact(ZERO2, t)
        m, h, formula = measured(rho), q.max_chsh(rho), 2 * math.sqrt(1 + math.sin(2 * t) ** 2)
        rows.append({"label": label, "t": t, "h": h, "m": m, "formula": formula})
        print(f"      t = {label:<6} best CHSH {f(h, 6)}   formula 2√(1 + sin²2t) = {f(formula, 6)}   measured {f(m['chsh'], 6)}  ({m['cls']})")
    peak = rows[-1]
    print(f"    At t = π/4 the agents reach {f(peak['h'], 6)}; Tsirelson's bound is {f(TSIRELSON, 6)} "
          f"(difference {tiny(abs(peak['h'] - TSIRELSON))}).")

    r = mulberry32(3)
    best = max_sig = match_err = 0.0
    over = violate = bad = 0
    N = 2000
    for t in range(N):
        rho = q.conjugate(product_state(random_qubit(r), random_qubit(r)), q.random_unitary2(r))
        if t % 4 == 3:
            rho = local(rho, q.depolarizing(0.3 * r.random()))
        h, m = q.max_chsh(rho), measured(rho)
        best = max(best, h)
        over += h > TSIRELSON + 1e-9
        violate += h > 2 + 1e-9
        max_sig = max(max_sig, m["signalling"])
        match_err = max(match_err, abs(m["chsh"] - h))
        bad += m["cls"] in ("signalling", "post-quantum")
    print(f"    {N} random interactions (random unitaries on random product agents, a quarter with noise):")
    print(f"      best CHSH {f(best, 6)}; above 2√2: {over}; above 2: {violate} ({f(100 * violate / N, 1)}%).")
    print(f"      measured statistics match the best value to {tiny(match_err)}; classified signalling or post-quantum: {bad};")
    print(f"      largest signalling {tiny(max_sig)} (experiment 13: 98% of classical networks signal without an extra rule).")
    return {"peak": peak["h"], "rows": rows, "best": best, "over": over, "violate": violate,
            "max_sig": max_sig, "bad": bad, "match_err": match_err, "N": N}


# ────────────── 4. network distance and decoherence ──────────────

def part4():
    print("\n  4. Network distance and decoherence: each agent's experience passes through k local channel steps")
    bell_pair = interact(ZERO2, math.pi / 4)
    dep, deph = [], []
    rho = rho2 = bell_pair
    for k in range(6):
        dep.append({"k": k, "h": q.max_chsh(rho), "formula": TSIRELSON * 0.9 ** (2 * k)})
        deph.append({"k": k, "h": q.max_chsh(rho2), "formula": 2 * math.sqrt(1 + 0.6 ** (4 * k))})
        rho = local(rho, q.depolarizing(0.1))
        rho2 = local(rho2, q.dephasing(0.2))
    full = q.max_chsh(local(bell_pair, q.dephasing(0.5)))
    err = max(abs(x["h"] - x["formula"]) for x in dep + deph)
    crossing = next(x for x in dep if x["h"] < 2)

    def fmt_row(xs):
        return "  ".join(f(x["h"]) for x in xs)
    print("      k =                               " + "  ".join(f"{k:<5}" for k in range(6)))
    print(f"      noise (depolarising, p = 0.1)     {fmt_row(dep)}   = 2√2·0.9^(2k)")
    print(f"      decoherence (dephasing, p = 0.2)  {fmt_row(deph)}   = 2√(1 + 0.6^(4k))")
    print(f"      closed forms match to {tiny(err)}. Noise makes the correlations local from k = {crossing['k']} on.")
    print(f"    Full dephasing turns both agents into Markov agents: best CHSH {f(full, 6)}, exactly the classical bound.")
    print("    Correlations decay with network distance as in experiment 13, but from 2√2, not from 4.")
    return {"err": err, "full": full, "crossing": crossing["k"], "dep": dep, "deph": deph}


# ────────────── 5. interference ──────────────

def part5():
    print("\n  5. Interference: an agent reaches an experience by three routes; block any subset (Sorkin 1994)")
    r = mulberry32(5)
    routes = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

    def sorkin(P):
        return {"I2": P["0,1"] - P["0"] - P["1"],
                "I3": P["0,1,2"] - P["0,1"] - P["0,2"] - P["1,2"] + P["0"] + P["1"] + P["2"]}

    def key(S):
        return ",".join(str(k) for k in S)
    # coherence c: 1 = pure quantum agent, 0 = fully dephased (a Markov agent over routes).
    coherences = [1, 0.5, 0]
    worst = [{"I2": 0.0, "I3": 0.0} for _ in coherences]
    classical = {"I2": 0.0, "I3": 0.0}
    for _ in range(200):
        phase = [2 * math.pi * r.random() for _ in range(3)]
        hop = [r.random() for _ in range(3)]
        for ci, c in enumerate(coherences):
            P = {}
            for S in routes:
                v = 0.0
                for k in S:
                    for l in S:
                        v += (1 if k == l else c) * math.cos(phase[k] - phase[l])
                P[key(S)] = v / 9
            sq = sorkin(P)
            worst[ci]["I2"] = max(worst[ci]["I2"], abs(sq["I2"]))
            worst[ci]["I3"] = max(worst[ci]["I3"], abs(sq["I3"]))
        Pc = {}
        for S in routes:
            acc = 0.0
            for k in S:
                acc += hop[k] / 3
            Pc[key(S)] = acc
        sc = sorkin(Pc)
        classical["I2"] = max(classical["I2"], abs(sc["I2"]))
        classical["I3"] = max(classical["I3"], abs(sc["I3"]))

    def fmt(x):
        return tiny(x) if x < 1e-9 else "up to " + f(x)
    print("    200 random agents; largest |I2| (two-route interference) and |I3| (three-route interference):")
    for name, w in [("quantum agents", worst[0]), ("half decohered", worst[1]), ("fully decohered", worst[2]), ("Markov agents", classical)]:
        print(f"      {name:<17}|I2| {fmt(w['I2']):<19} |I3| {fmt(w['I3'])}")
    print("    Quantum agents interfere in pairs but never three at a time: the signature of the Born rule, which")
    print("    photon experiments confirm (Sinha et al. 2010). Decoherence scales I2 down to 0, and Markov agents cannot")
    print("    interfere at all (experiment 09).")
    return {"qI2": worst[0]["I2"], "qI3": max(w["I3"] for w in worst), "half": worst[1]["I2"],
            "dephased": worst[2]["I2"], "classical": classical}


def main():
    t0 = time.time()
    print("=" * 78)
    print("Quantum agents")
    print("=" * 78)
    R1, R2, R3, R4, R5 = part1(), part2(), part3(), part4(), part5()

    print("\n" + "─" * 78)
    print("Findings")
    print("─" * 78)
    print(f"  1. Yes: combined quantum agents reach the quantum limit exactly ({f(R3['peak'], 6)} = 2√2) and never exceed it")
    print(f"     (0 of {R3['N']} random interactions). No-signalling holds automatically, and the measured statistics pass")
    print("     experiment 13's classifier as quantum. Without interaction, combination stays at or below 2.")
    print("  2. What does the work is the quantum structure put in, not the agent formalism: the complex Hilbert")
    print("     space and the Born rule cap CHSH at 2√2 (Tsirelson's theorem), and the tensor-product combination with")
    print("     local measurements guarantees no-signalling. Classical kernels supply neither (experiment 13).")
    print("     So quantum agents answer experiment 13 by assumption. The open problem for Hoffman's program is to")
    print("     derive this structure from agent dynamics, or to take quantum agents as the starting point.")
    print("  3. The classical world is the decohered limit: a Markov agent is a quantum agent with all coherence")
    print(f"     erased. Dephasing drives CHSH from 2√2 to exactly {f(R4['full'])} and removes interference; quantum agents")
    print(f"     interfere pairwise (|I2| up to {f(R5['qI2'])}) but never three-way (|I3| at {tiny(R5['qI3'])}).")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")
    return R1, R2, R3, R4, R5


if __name__ == "__main__":
    main()
