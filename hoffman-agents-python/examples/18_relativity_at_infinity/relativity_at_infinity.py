"""
Relativity at Infinity

Hoffman conjectures that Minkowski spacetime emerges from the traces of
n-cycle Markov chains as n -> infinity. No finite network has relativity; the
question is what appears in the limit.

The agent: its experience is a position on a ring of n sites per unit length
plus one bit of memory, the direction it is heading. Each step it moves one
site; with probability a/n it reverses (for a quantum agent: amplitude
i sin(m/n), a discrete-time quantum walk). Two observers:
  - the lab, which sees the agent's position after t·n steps;
  - the agent's clock, which registers only its changes of experience (the
    reversals), the observer-restricted time of experiment 15.

  A. Light cone: does a maximal speed survive n -> infinity?
  B. Time dilation: does the clock depend only on proper time √(t² − x²)?
  C. Preferred frame: is the whole dynamics Lorentz invariant, classically
     and with complex amplitudes?

Everything is computed exactly (dynamic programming), not sampled.
Port of the Node example; prints the same numbers.
"""
import math
import time

import numpy as np

A_RATE = 5  # classical reversal rate per unit time
MASS = 5    # quantum reversal amplitude rate (the walk's mass)
NS = [20, 80, 320, 1280]


def f(x, d=3):
    return f"{x:.{d}f}"


def bessel_i(nu, z):
    t = (z / 2) ** nu
    for k in range(1, nu + 1):
        t /= k
    s = 0.0
    for k in range(80):
        s += t
        t *= (z / 2) ** 2 / ((k + 1) * (k + 1 + nu))
    return s


def bessel_j0(z):
    s, t = 0.0, 1.0
    for k in range(120):
        s += t
        t *= -((z / 2) ** 2) / ((k + 1) ** 2)
    return s


def erfc(x):
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), identical in both languages."""
    t = 1 / (1 + 0.3275911 * x)
    y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return y * math.exp(-x * x)


def clock_prediction(a, tau):
    return 1 + a * tau * bessel_i(1, a * tau) / bessel_i(0, a * tau)


def density_prediction(a, t, tau):
    return a * math.exp(-a * t) * bessel_i(0, a * tau)


def amplitude_prediction(m, tau):
    return m * bessel_j0(m * tau)


def proper_time_from_clock(a, reading):
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if clock_prediction(a, mid) < reading:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def evolve(n, marks, a=A_RATE, m=MASS):
    """Exact evolution with n steps per unit time; see the Node version."""
    T = max(round(k["t"] * n) for k in marks)
    W = 2 * T + 1
    MR, ML, FR, FL = (np.zeros(W) for _ in range(4))
    RR, RI, LR, LI = (np.zeros(W) for _ in range(4))
    MR[T] = 1.0
    RR[T] = 1.0
    p, c, s = a / n, math.cos(m / n), math.sin(m / n)
    out = [None] * len(marks)

    def from_left(A):   # right-movers arrive from i-1
        B = np.zeros(W)
        B[1:] = A[:-1]
        return B

    def from_right(A):  # left-movers arrive from i+1
        B = np.zeros(W)
        B[:-1] = A[1:]
        return B
    for step in range(1, T + 1):
        mr, fr, rr, ri = from_left(MR), from_left(FR), from_left(RR), from_left(RI)
        ml, fl, lr, li = from_right(ML), from_right(FL), from_right(LR), from_right(LI)
        MR, ML = (1 - p) * mr + p * ml, (1 - p) * ml + p * mr
        FR, FL = (1 - p) * fr + p * (fl + ml), (1 - p) * fl + p * (fr + mr)
        RR, RI = c * rr - s * li, c * ri + s * lr
        LR, LI = c * lr - s * ri, c * li + s * rr
        for idx, k in enumerate(marks):
            if round(k["t"] * n) != step:
                continue
            i = T + round(k["x"] * n)
            out[idx] = {**k, "tau": math.sqrt(k["t"] ** 2 - k["x"] ** 2), "P": ML[i] * n,
                        "E": FL[i] / ML[i], "Q": LI[i] * n, "Qre": LR[i] * n}
    norm = float(np.sum(RR ** 2 + RI ** 2 + LR ** 2 + LI ** 2))
    return {"marks": out, "norm": norm}


def memoryless_outside_cone(n, a=A_RATE):
    """Memoryless control, same long-run diffusion constant: mass beyond |x| > 1 at t = 1."""
    h = math.sqrt(1 / (a * n))
    log_fact = [0.0]
    for k in range(1, n + 1):
        log_fact.append(log_fact[k - 1] + math.log(k))
    outside = 0.0
    for k in range(n + 1):
        if abs(h * (2 * k - n)) > 1 + 1e-12:
            outside += math.exp(log_fact[n] - log_fact[k] - log_fact[n - k] - n * math.log(2))
    return {"outside": outside, "max_speed": h * n}


def part_a():
    print("\n  A. Light cone. Where can the agent be after one unit of lab time?")
    limit = erfc(math.sqrt(A_RATE / 2))
    print("      n       memoryless agent: top speed, mass beyond x = ±1      agent with a direction bit: top speed, mass beyond")
    rows = []
    for n in NS:
        m = memoryless_outside_cone(n)
        rows.append(m)
        print(f"      {str(n):<6}  {f(m['max_speed'], 1):>8}   {f(m['outside'], 4)}                               {f(1, 1):>8}   0 (exactly)")
    print(f"      limit      ∞          {f(limit, 4)} (diffusion)                         1.0        0")
    print("    With one bit of memory the agent has a maximal speed c = 1 at every n, and it survives the limit.")
    print("    Without memory, the same spread needs ever faster steps: top speed √(n/a) → ∞ and no light cone.")
    return {"rows": rows, "limit": limit}


EQUAL_TAU = [{"t": 0.6, "x": 0}, {"t": 0.75, "x": 0.45}, {"t": 1, "x": 0.8}]


def part_b():
    print("\n  B. Time dilation. The clock counts the agent's changes of experience (reversals) during one unit of lab time;")
    print("     readings are averaged over walks that start heading right and end heading left at displacement x;")
    print(f"     the continuum prediction depends only on proper time τ = √(t² − x²): 1 + aτ·I1(aτ)/I0(aτ), a = {A_RATE}.")
    moving = [{"t": 1, "x": 0}, {"t": 1, "x": 0.6}, {"t": 1, "x": 0.8}]
    by_n = [{"n": n, "moving": evolve(n, moving)["marks"], "equal": evolve(n, EQUAL_TAU)["marks"]} for n in NS]
    last = by_n[-1]
    print(f"      speed v   clock reading (n = {last['n']})   prediction   proper time read off the clock   √(1 − v²)")
    dil = []
    for k in last["moving"]:
        v = k["x"] / k["t"]
        tau, pred = proper_time_from_clock(A_RATE, k["E"]), clock_prediction(A_RATE, k["tau"])
        print(f"      {f(v, 1):<8}  {f(k['E'], 4):<26}  {f(pred, 4):<11}  {f(tau, 4):<31}  {f(math.sqrt(1 - v * v), 4)}")
        dil.append({"v": v, "E": k["E"], "pred": pred, "tau": tau})
    dil_err = max(abs(d["tau"] - math.sqrt(1 - d["v"] ** 2)) for d in dil)
    print(f"    Moving clocks run slow by the Lorentz factor (within {f(dil_err, 4)} at n = {last['n']}). Three agents with the same proper time")
    print("    τ = 0.6 but speeds 0, 0.6, 0.8 (lab times 0.6, 0.75, 1):")
    print("      n       clock readings                      spread     largest error vs prediction")
    conv = []
    for b in by_n:
        Es = [k["E"] for k in b["equal"]]
        spread = max(Es) - min(Es)
        err = max(abs(k["E"] - clock_prediction(A_RATE, k["tau"])) for k in b["equal"])
        print(f"      {str(b['n']):<6}  {'  '.join(f(e, 4) for e in Es):<34}  {f(spread, 4):<9}  {f(err, 4)}")
        conv.append({"n": b["n"], "spread": spread, "err": err})
    rate = conv[-2]["spread"] / conv[-1]["spread"]
    print(f"    The spread shrinks by {f(rate, 1)}× per 4× in n (∝ 1/n): at every finite n the lattice has a preferred frame;")
    print("    in the limit the clock depends on proper time alone.")
    return {"dil": dil, "conv": conv, "rate": rate, "dil_err": dil_err}


def part_c():
    print("\n  C. Is the whole dynamics Lorentz invariant? Same three agents with τ = 0.6:")
    by_n = [{"n": n, **evolve(n, EQUAL_TAU)} for n in NS]
    last = by_n[-1]["marks"]
    print(f"      classical agent, probability density   {'  '.join(f(k['P'], 4) for k in last)}   continuum "
          f"{'  '.join(f(density_prediction(A_RATE, k['t'], 0.6), 4) for k in EQUAL_TAU)}")
    print(f"      quantum agent, amplitude density       {'  '.join(f(k['Q'], 4) + 'i' for k in last)}   continuum "
          f"{f(amplitude_prediction(MASS, 0.6), 4)}i at all three")
    q_spread = [{"n": b["n"], "spread": max(k["Q"] for k in b["marks"]) - min(k["Q"] for k in b["marks"])} for b in by_n]
    norm_err = max(abs(b["norm"] - 1) for b in by_n)
    print("      quantum spread across the three by n: " + ", ".join(f"{q['n']}: {f(q['spread'], 4)}" for q in q_spread))
    print(f"      (quantum evolution unitary: total probability 1 within {'machine precision' if norm_err < 1e-9 else f'{norm_err:.1e}'})")
    print("    The classical density carries e^(−at): it depends on lab time, so the network's rest frame is special.")
    print("    The quantum amplitude tends to i·m·J0(mτ), a function of τ alone. It is one component of the 1+1-dimensional")
    print("    Dirac propagator (Feynman's checkerboard), which is Lorentz covariant: no preferred frame, and the reversal")
    print("    rate m plays the role of mass.")
    return {"last": last, "q_spread": q_spread, "norm_err": norm_err}


def main():
    t0 = time.time()
    print("=" * 78)
    print("Relativity at infinity")
    print("=" * 78)
    A, B, C = part_a(), part_b(), part_c()
    print("\n" + "─" * 78)
    print("Findings")
    print("─" * 78)
    print("  1. A light cone needs memory. An agent that remembers its direction (one bit of experience) keeps a")
    print("     maximal speed as n → ∞. A memoryless agent's top speed diverges: in the limit it spreads by diffusion,")
    print(f"     {f(A['limit'], 4)} of it lands beyond distance 1 after unit time, and no finite speed bounds it.")
    print("  2. Time dilation appears in the limit. The clock that counts the agent's own changes of experience")
    print(f"     converges to a function of proper time alone: at speed 0.6 it reads {f(B['dil'][1]['tau'], 3)} and at 0.8 it reads {f(B['dil'][2]['tau'], 3)} units")
    print("     per unit of lab time, Einstein's √(1 − v²). In this model the time-dilation part of Hoffman's conjecture")
    print("     holds, and only in the limit: every finite network has a preferred frame, with errors shrinking like 1/n.")
    print("  3. Full Lorentz invariance needs complex amplitudes. The classical agent keeps its network's rest frame;")
    print("     the quantum agent's limit is the Dirac propagator, Lorentz covariant, with mass as reversal rate.")
    print("     The mathematics is classical (Goldstein 1951, Kac 1974, Feynman & Hibbs 1965, Gaveau et al. 1984);")
    print("     the model is 1+1-dimensional and ours, not Hoffman's exact construction.")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")
    return A, B, C


if __name__ == "__main__":
    main()
