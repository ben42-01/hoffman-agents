"""
Time in the Traces

Hoffman's trace logic: an observer who can only see part of a network of
conscious agents (its window S) experiences the TRACE CHAIN on S:

    P_S = P_SS + P_SC (I - P_CC)^(-1) P_CS

This experiment asks what TIME looks like from inside a trace.

  1. Every observer experiences its own Markov chain: simulated observations
     match the formula, and nested observers are consistent.
  2. Proper time: an observer's clock ticks only when the network is in its
     window, so observers age at rate pi(S) (Kac's lemma); nested clocks agree.
  3. The arrow of time is observer-dependent.
     a. Reversible network: every observer's uncertainty H(X_n | X_0) still grows
        (the entropic arrow from projection; Hoffman, Prakash & Prentner 2023).
     b. Irreversible network: windows see different amounts of arrow. A window of
        two states never sees one (every two-state chain satisfies detailed
        balance), so perceiving an arrow of time needs >= 3 states.

Not tested here: the conjecture that Minkowski spacetime emerges from traces of
n-cycle chains as n -> infinity.

Port of the Node example; prints the same numbers.
"""
import math
import time

import numpy as np

from conscious_agent import FormalConsciousAgent, markov, mulberry32, trace

STEPS = 200000


# ────────────── networks ──────────────

def random_row(n, r, i=None, stickiness=0.0):
    u = [r.random() + 1e-3 for _ in range(n)]
    s = sum(u)
    return [(1 - stickiness) * v / s + (stickiness if j == i else 0.0) for j, v in enumerate(u)]


def formal_agent(r, stickiness):
    X, G, W = ["calm", "alert"], ["stay", "move"], ["L", "R"]
    P = {w: [random_row(2, r, i, stickiness) for i in range(2)] for w in W}
    D = [random_row(2, r) for _ in X]
    A = {g: [random_row(2, r, i, stickiness) for i in range(2)] for g in G}
    return FormalConsciousAgent(X, G, W, P, D, A, rng=r)


def agent_network(seed, stickiness=0.3):
    """Two conscious agents combined, acting on a shared world: 8 joint states "x1|x2|w"."""
    r = mulberry32(seed)
    K = FormalConsciousAgent.combine(formal_agent(r, stickiness), formal_agent(r, stickiness)).joint_kernel()
    return K.states, K.matrix


def reversible_network(n, seed):
    """Random symmetric conductances: detailed balance holds."""
    r = mulberry32(seed)
    Wt = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            Wt[i, j] = Wt[j, i] = r.random() if r.random() < 0.5 else 0.02
    return Wt / Wt.sum(axis=1, keepdims=True)


# ────────────── observers ──────────────

def window_of(states, pred):
    return [i for i, s in enumerate(states) if pred(s.split("|"))]


def simulate(P, steps, seed, windows):
    r = mulberry32(seed)
    n = len(P)
    rows = [list(P[i]) for i in range(n)]
    pos = [{s: k for k, s in enumerate(S)} for S in windows]
    counts = [np.zeros((len(S), len(S))) for S in windows]
    last = [None] * len(windows)
    ticks = [0] * len(windows)
    gaps = [[] for _ in windows]
    last_tick = [None] * len(windows)
    x = 0
    for t in range(steps):
        u, acc = r.random(), 0.0
        for j in range(n):
            acc += rows[x][j]
            if u < acc:
                x = j
                break
        for w in range(len(windows)):
            k = pos[w].get(x)
            if k is None:
                continue
            ticks[w] += 1
            if last[w] is not None:
                counts[w][last[w], k] += 1
            if last_tick[w] is not None:
                gaps[w].append(t - last_tick[w])
            last[w], last_tick[w] = k, t
    return [{"counts": counts[w], "ticks": ticks[w], "gaps": gaps[w]} for w in range(len(windows))]


def f(x, d=3):
    return f"{x:.{d}f}"


def e1(x):
    """JavaScript-style toExponential(1)."""
    if x == 0:
        return "0.0e+0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** exp
    if round(abs(mant), 1) >= 10:
        mant, exp = mant / 10, exp + 1
    return f"{mant:.1f}e{'+' if exp >= 0 else '-'}{abs(exp)}"


def tiny(x):
    return "machine precision" if x < 1e-9 else e1(x)


def max_abs_diff(A, B):
    return float(np.max(np.abs(np.asarray(A) - np.asarray(B))))


def main():
    t0 = time.time()
    print("=" * 78)
    print("Time in the traces")
    print("=" * 78)

    states, P = agent_network(15)
    pi = markov.stationary(P)["pi"]
    observers = [
        {"name": "O_world  (sees only world = L)", "S": window_of(states, lambda s: s[2] == "L")},
        {"name": "O_agent  (sees only agent 1 alert)", "S": window_of(states, lambda s: s[0] == "alert")},
        {"name": "O_both   (both conditions)", "S": window_of(states, lambda s: s[0] == "alert" and s[2] == "L")},
    ]
    print(f"\n  Network: two combined conscious agents on a shared world, {len(states)} joint states.")
    print(f"  Simulated for {STEPS:,} network steps; each observer records only its window.")

    print("\n  1. Each observer experiences its own Markov chain (the trace)")
    print(f"    {'observer':<38} states  max |empirical − trace formula|")
    sims = simulate(P, STEPS, 1, [o["S"] for o in observers])
    for o, sim in zip(observers, sims):
        o["T"] = trace.trace_chain(P, o["S"])
        emp, _ = markov.normalize_rows(sim["counts"])
        o["err"] = max_abs_diff(emp, o["T"])
        print(f"    {o['name']:<38} {len(o['S']):<7} {f(o['err'])}")
    inner = [observers[0]["S"].index(s) for s in observers[2]["S"]]
    nested_err = max_abs_diff(trace.trace_chain(observers[0]["T"], inner), observers[2]["T"])
    print(f"    O_both as a trace of O_world's experience vs directly: max difference {tiny(nested_err)}")

    print("\n  2. Proper time: each observer ages at its own rate")
    print(f"    {'observer':<38} ticks per network step   mean gap   Kac 1/π(S)   gap spread (sd)")
    for o, sim in zip(observers, sims):
        rate = trace.clock_rate(pi, o["S"])
        measured = sim["ticks"] / STEPS
        g = np.array(sim["gaps"], dtype=float)
        mean, sd = float(g.mean()), float(g.std())
        o["rate"] = rate
        o["kac_err"] = abs(trace.mean_return_time(P, o["S"], pi) - 1 / rate)
        print(f"    {o['name']:<38} {f(measured)} (π(S) = {f(rate)})   {f(mean, 2):<8} {f(1 / rate, 2):<12} {f(sd, 2)}")
    ow, ob = observers[0], observers[2]
    pi_world_trace = markov.stationary(ow["T"])["pi"]
    both_in_world_time = float(sum(pi_world_trace[k] for k in inner))
    print(f"    O_both's clock measured in O_world's own time: {f(both_in_world_time)} ticks per O_world tick;")
    print(f"    from the network: π(O_both)/π(O_world) = {f(ob['rate'] / ow['rate'])}. Nested clocks agree exactly.")
    print("    Between two of its own moments an observer cannot tell how much hidden time passed: the gaps")
    print("    vary (sd above), but every observer experiences one step per event.")

    print("\n  3a. A reversible network: no arrow of time in the dynamics")
    R = reversible_network(8, 4)
    piR = markov.stationary(R)["pi"]
    TR = trace.trace_chain(R, [0, 1, 2])
    piTR = markov.stationary(TR)["pi"]
    h_net = trace.conditional_entropy_profile(R, piR, 5)
    h_obs = trace.conditional_entropy_profile(TR, piTR, 5)
    print(f"    irreversibility: network {f(markov.irreversibility(R, piR))}, observer's trace {f(markov.irreversibility(TR, piTR))}")
    print(f"    H(X_n) stays {f(markov.entropy(piR))} nats (network) and {f(markov.entropy(piTR))} (observer) at every n, but")
    print(f"    H(X_n | X_0), network:  {'  '.join(f(v) for v in h_net)}")
    print(f"    H(X_n | X_0), observer: {'  '.join(f(v) for v in h_obs)}")

    def monotone(h):
        return all(i == 0 or v >= h[i - 1] - 1e-12 for i, v in enumerate(h))

    ok = "monotone in both" if monotone(h_net) and monotone(h_obs) else "NOT monotone"
    print(f"    → uncertainty about the future grows ({ok}), with no arrow in the dynamics.")

    print("\n  3b. An irreversible network: is the arrow the same for every observer?")
    irr_net = markov.irreversibility(P, pi)
    print(f"    network irreversibility: {f(irr_net)}")
    for o in observers:
        o["irr"] = markov.irreversibility(o["T"], markov.stationary(o["T"])["pi"])
        note = "  (two states: always 0)" if len(o["S"]) == 2 else ""
        print(f"    {o['name']:<38} {f(o['irr'])}{note}")
    cycle = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    ct = trace.trace_chain(cycle, [0, 1])
    fmt_row = lambda row: ", ".join(str(int(v)) if float(v).is_integer() else str(v) for v in row)
    print(f"    one-way 3-cycle (irreversibility {f(markov.irreversibility(cycle))}), observer seeing 2 of 3 states: "
          f"trace [[{fmt_row(ct[0])}], [{fmt_row(ct[1])}]], irreversibility {f(markov.irreversibility(ct))}")

    print("\n" + "─" * 78)
    print("Findings")
    print("─" * 78)
    worst = max(o["err"] for o in observers)
    print("  1. Observers experience exactly the trace chain of their window (simulation vs formula within")
    print(f"     {f(worst)} after {STEPS:,} steps), and nested observers are consistent ({tiny(nested_err)}).")
    print("  2. Time is per-observer: each observer's clock runs at π(S) ticks per network step (Kac's")
    print(f"     formula confirmed to {tiny(max(o['kac_err'] for o in observers))}), clocks of nested observers compose exactly,")
    print("     and no observer can perceive the hidden time between its own moments.")
    print("  3. The arrow of time depends on the observer. With reversible dynamics, every observer's")
    print("     uncertainty about its future still grows: the entropic arrow comes from observing, not from")
    print("     the dynamics. With irreversible dynamics, windows see different amounts of arrow")
    print(f"     (network {f(irr_net)}; observers {', '.join(f(o['irr']) for o in observers)}). An observer that sees only two")
    print("     states never perceives an arrow, because every two-state chain satisfies detailed balance: even a")
    print("     one-way cycle looks time-symmetric through a two-state window. An arrow needs a window of >= 3 states.")
    print("\n  What this does not show: that these per-observer clocks obey Einstein's time dilation. That is")
    print("  Hoffman's open conjecture (Minkowski spacetime from traces of n-cycles as n → ∞), and it is the")
    print("  natural next experiment once the exact construction is available.")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
