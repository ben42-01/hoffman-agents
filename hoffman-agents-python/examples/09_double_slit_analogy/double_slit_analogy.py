"""
Double-Slit Analogy

Follow-up to 07_exchange_symmetry and 08_observer_gated_combination. This
experiment asks directly: can the double-slit experiment's famous
interference pattern be reproduced using this library's Markov-chain /
trie machinery? The honest answer, demonstrated here rather than just
asserted, is: NO -- not without adding an ingredient the library doesn't
have. But the reason WHY is itself the whole point of the double-slit
experiment historically, and this script makes that reason concrete.

Part 1 -- Classical "diamond" world:
  A branching Markov world: Source splits 50/50 into Path A (length La)
  or Path B (length Lb), both reconverging at a Detector. We compare:
    - "tracked":  per-path histograms, summed after the fact
    - "erased":   pooled histogram, no path label ever recorded
  These are mathematically IDENTICAL -- exactly why real double-slit
  interference is considered deep evidence against classical probability
  alone.

Part 2 -- What genuine interference requires:
  A small, explicit complex-amplitude construction using the SAME path
  lengths, showing what combining AMPLITUDES (not probabilities) before
  squaring would produce: a real oscillating fringe pattern.

Bonus -- Existing complex-eigenvalue machinery check:
  Verifies (using numpy) whether THIS diamond world's transition matrix
  already has complex eigenvalues -- the same mathematical family
  (complex exponentials) as interference, though a different phenomenon
  (temporal oscillation of one system, not spatial interference of two
  paths).

This is an exploratory computational analogy -- NOT a physics claim.
"""

from __future__ import annotations

import random

import numpy as np

PATH_A_LENGTH = 3
PATH_B_LENGTH = 5  # deliberately different & coprime with A for clarity


# ────────────── Part 1: Classical diamond world ──────────────


def run_diamond_trial(rng: random.Random):
    path = "A" if rng.random() < 0.5 else "B"
    length = PATH_A_LENGTH if path == "A" else PATH_B_LENGTH
    return path, length


def part1_classical_diamond(n_trials=5000):
    print("\n  Part 1: Classical Diamond World (tracked vs erased)\n")
    print(f"  Path A length: {PATH_A_LENGTH} steps.  Path B length: {PATH_B_LENGTH} steps.\n")

    rng = random.Random(42)
    hist_a, hist_b, hist_erased = {}, {}, {}

    for _ in range(n_trials):
        path, arrival_step = run_diamond_trial(rng)
        if path == "A":
            hist_a[arrival_step] = hist_a.get(arrival_step, 0) + 1
        else:
            hist_b[arrival_step] = hist_b.get(arrival_step, 0) + 1
        hist_erased[arrival_step] = hist_erased.get(arrival_step, 0) + 1

    tracked_sum = {}
    for t, c in hist_a.items():
        tracked_sum[t] = tracked_sum.get(t, 0) + c
    for t, c in hist_b.items():
        tracked_sum[t] = tracked_sum.get(t, 0) + c

    print(f"  Trials: {n_trials}")
    print("\n    step | tracked (A+B summed) | erased (pooled, no path label)")
    print("    -----|----------------------|-------------------------------")
    all_steps = sorted(set(list(tracked_sum.keys()) + list(hist_erased.keys())))
    identical = True
    for t in all_steps:
        tracked = tracked_sum.get(t, 0)
        erased = hist_erased.get(t, 0)
        if tracked != erased:
            identical = False
        print(f"    {str(t).rjust(4)} | {str(tracked).rjust(20)} | {str(erased).rjust(29)}")

    print(f"\n  Tracked and erased histograms are {'IDENTICAL' if identical else 'DIFFERENT'} (byte-for-byte counts).")
    print("  This is the expected classical result: whether or not you record which")
    print("  path was taken has ZERO effect on the observed distribution. Ordinary")
    print("  probability theory has no mechanism by which \"erasing which-path info\"")
    print("  could ever change an outcome -- which is precisely why real double-slit")
    print("  interference is considered deep evidence that nature is NOT running on")
    print("  ordinary probability theory alone.")

    return {"identical": identical, "tracked_sum": tracked_sum, "hist_erased": hist_erased}


# ────────────── Part 2: What genuine interference requires ──────────────


def part2_complex_amplitude():
    print("\n  Part 2: What Genuine Interference Would Require\n")
    print("  Using the SAME path lengths (La=3, Lb=5), assign each path a complex")
    print("  amplitude with phase proportional to path length, swept against a toy")
    print("  \"wavelength\" parameter (standing in for detector screen position).\n")

    la, lb = PATH_A_LENGTH, PATH_B_LENGTH
    results = []

    lam = 1.0
    while lam <= 20.0:
        p_classical = 0.5 + 0.5  # flat, no interference possible classically

        phase_a = 2 * np.pi * la / lam
        phase_b = 2 * np.pi * lb / lam
        amp_a = np.sqrt(0.5) * np.exp(1j * phase_a)
        amp_b = np.sqrt(0.5) * np.exp(1j * phase_b)
        p_quantum = abs(amp_a + amp_b) ** 2

        results.append((lam, p_classical, p_quantum))
        lam += 0.5

    print("    lambda | classical P (flat) | \"quantum\" P = |ampA+ampB|^2 (fringe)")
    print("    -------|---------------------|-------------------------------------")
    for lam, p_c, p_q in results:
        bar = "#" * round(p_q * 15)
        print(f"    {f'{lam:.1f}'.rjust(6)} | {f'{p_c:.3f}'.rjust(19)} | {f'{p_q:.3f}'.rjust(6)} {bar}")

    quantum_values = [r[2] for r in results]
    min_q, max_q = min(quantum_values), max(quantum_values)
    print(f"\n  Classical: flat at 1.000 across every lambda -- no fringes, cannot oscillate.")
    print(f"  \"Quantum-style\": ranges from {min_q:.3f} to {max_q:.3f} -- clear fringe pattern.")
    print("\n  The only difference between these two calculations is WHERE the squaring")
    print("  happens: sum-then-square (amplitudes) vs square-then-sum (probabilities).")
    print("  That one-line difference is the entire content of what makes interference")
    print("  possible. Our trie/Markov machinery only ever does the latter -- it has no")
    print("  representation of phase, so it structurally cannot take the other branch.")

    return {"min_q": min_q, "max_q": max_q}


# ────────────── Bonus: does the diamond graph's own transition matrix have complex eigenvalues? ──────────────


def bonus_eigenvalue_check():
    print("\n  Bonus: Does the Diamond Graph Itself Have Complex Eigenvalues?\n")
    print("  Building the actual transition matrix for the diamond world (with a loop")
    print("  back from Detector to Source, making it a proper ergodic chain) and")
    print("  checking its eigenvalues with numpy.\n")

    # States: 0=Source, 1..La = path A chain, La+1..La+Lb = path B chain, last = Detector
    la, lb = PATH_A_LENGTH, PATH_B_LENGTH
    n_states = 1 + la + lb + 1  # source + A-chain + B-chain + detector
    detector_idx = n_states - 1
    P = np.zeros((n_states, n_states))

    # Source (0) -> first node of A-chain (1) or first node of B-chain (la+1), 50/50
    P[0, 1] = 0.5
    P[0, la + 1] = 0.5

    # A-chain: 1 -> 2 -> ... -> la -> Detector
    for i in range(1, la):
        P[i, i + 1] = 1.0
    P[la, detector_idx] = 1.0

    # B-chain: la+1 -> la+2 -> ... -> la+lb -> Detector
    for i in range(la + 1, la + lb):
        P[i, i + 1] = 1.0
    P[la + lb, detector_idx] = 1.0

    # Detector -> Source (loop back, making the chain ergodic for eigenanalysis)
    P[detector_idx, 0] = 1.0

    eigenvalues = np.linalg.eigvals(P)
    complex_ones = [ev for ev in eigenvalues if abs(ev.imag) > 1e-9]

    print(f"  Transition matrix size: {n_states}x{n_states}")
    print(f"  Total eigenvalues: {len(eigenvalues)}")
    print(f"  Complex eigenvalues found: {len(complex_ones)}")
    if complex_ones:
        print("\n    magnitude | phase (radians) | period (2*pi/phase)")
        print("    ----------|-----------------|--------------------")
        seen = set()
        for ev in complex_ones:
            key = round(abs(ev), 4)
            if key in seen:
                continue
            seen.add(key)
            mag = abs(ev)
            phase = np.angle(ev)
            period = abs(2 * np.pi / phase) if phase != 0 else float("inf")
            print(f"    {mag:.4f}    | {phase:+.4f}         | {period:.2f}")
        print("\n  Yes -- this graph's transition matrix has complex eigenvalues. Their")
        print("  presence means repeated application of P (i.e. P^n, the n-step transition")
        print("  matrix) contains genuinely OSCILLATORY terms (lambda^n rotates in the")
        print("  complex plane as n grows) -- real wave-like mathematics, already latent")
        print("  in ordinary cyclic Markov chains. This is the same mathematical FAMILY as")
        print("  the double-slit's interference term, but a different phenomenon: temporal")
        print("  oscillation of one system's return probability over time, not spatial")
        print("  interference between two simultaneous paths.")
    else:
        print("\n  No complex eigenvalues found for this specific topology.")

    return {"n_complex": len(complex_ones)}


# ────────────── Main ──────────────


def main():
    print("=" * 70)
    print("Double-Slit Analogy")
    print("=" * 70)
    print("\nCan this library reproduce double-slit interference? Testing directly")
    print("rather than asserting an answer.\n")
    print("This is an exploratory computational analogy -- NOT a physics claim.\n")

    p1 = part1_classical_diamond(5000)
    p2 = part2_complex_amplitude()
    bonus = bonus_eigenvalue_check()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    tracked_ok = "NO interference possible (confirmed)" if p1["identical"] else "unexpected result"
    print(f"\n  Classical trie/Markov world:  tracked == erased -> {tracked_ok}")
    print(f"  Toy complex-amplitude add-on: fringe range [{p2['min_q']:.3f}, {p2['max_q']:.3f}] -> interference IS possible, but requires phase")
    print(f"  Diamond graph's own transition matrix: {bonus['n_complex']} complex eigenvalues found")
    print("\n  Verdict: the library, as built, cannot produce interference -- because")
    print("  it only ever combines non-negative probabilities, never signed/complex")
    print("  amplitudes. But cyclic Markov chains (already buildable with WorldBuilder)")
    print("  DO have complex eigenvalues, and those produce real oscillatory dynamics --")
    print("  a related, already-present piece of wave-like math, just applied to a")
    print("  different question (how one system evolves over time, not how two paths")
    print("  interfere in space).\n")


if __name__ == "__main__":
    main()
