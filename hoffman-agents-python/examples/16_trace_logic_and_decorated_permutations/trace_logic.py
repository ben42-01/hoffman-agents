"""
Trace Logic and Decorated Permutations

Hoffman's program has two mathematical bridges from conscious agents to
physics. This experiment builds both and checks them.

  A. Decorated permutations (Hoffman, Prakash & Prentner, "Fusions of
     Consciousness", 2023, Definition 2). Each Markov chain maps to a
     decorated permutation, the combinatorial object that indexes cells of
     the positive Grassmannian (the geometry behind the amplituhedron and
     scattering amplitudes). What does the map keep, and what does it depend on?

  B. Trace logic (Hoffman & Prakash, "Traces of Consciousness"). Agents are
     ordered by "A is a trace of B"; "A observes B" means A's qualia kernel
     Q = DAP is a trace of B's dynamics. Claimed: locally Boolean, globally
     not even a lattice, and homomorphic to the Lebesgue logic of belief via
     the stationary measure.

  C. Two physics proposals from the trace-logic papers, stress-tested with
     consistency checks they could fail: mass <-> entropy rate, and
     speed <-> commute time.

Port of the Node example; prints the same numbers.
"""
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np

from conscious_agent import decorated, markov, mulberry32, trace

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "exp14", HERE.parent / "14_spacetime_in_the_headset" / "spacetime_in_the_headset.py")
_exp14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp14)
coupled_rings = _exp14.coupled_rings
APPENDIX_B = HERE.parents[1] / "tests" / "fixtures" / "fusions-appendix-b.json"


def f(x, d=3):
    return f"{x:.{d}f}"


def tiny(x):
    return "machine precision" if x < 1e-9 else f"{x:.1e}"


def random_kernel(n, r, density=1.0):
    rows = []
    for i in range(n):
        u = [r.random() + 0.05 if (j == i or r.random() < density) else 0.0 for j in range(n)]
        s = sum(u)
        rows.append([v / s for v in u])
    return np.array(rows)


def cycle_kernel(n, cycles):
    P = np.zeros((n, n))
    for c in cycles:
        for i, s in enumerate(c):
            P[s - 1, c[(i + 1) % len(c)] - 1] = 1.0
    return P


def preserves_cyclic_order(P, perm):
    """Relabelling preserves the cyclic order of every recurrent class of >= 3 states."""
    for cls in markov.closed_classes(P):
        if len(cls) < 3:
            continue
        img = [perm[i] for i in cls]
        srt = sorted(img)
        k = srt.index(img[0])
        if any(v != srt[(k + i) % len(srt)] for i, v in enumerate(img)):
            return False
    return True


def show(s):
    return "[" + ", ".join(str(v) for v in s) + "]"


# ────────────── A. decorated permutations ──────────────

def part_a():
    print("\n  A. Decorated permutations")
    example = cycle_kernel(9, [[1, 5, 8], [2], [3, 4], [6], [7, 9]])
    sigma = decorated.decorated_permutation(example)
    reproduces = sigma == [8, 11, 4, 12, 10, 15, 9, 14, 16]
    print(f"    paper's example (cycles (1 5 8)(2)(3 4)(6)(7 9)): {show(sigma)}  "
          f"{'matches the paper' if reproduces else 'DIFFERS from the paper'}")

    print("    2-state agents Q = [[1-x, x], [y, 1-y]] (the four cells of the Markov square M2):")
    for name, x, y in [("x = y = 0", 0, 0), ("x, y > 0", 0.3, 0.6), ("x = 0, y > 0", 0, 0.6), ("x > 0, y = 0", 0.3, 0)]:
        print(f"      {name:<14} σ = {show(decorated.decorated_permutation([[1 - x, x], [y, 1 - y]]))}")

    table = json.loads(APPENDIX_B.read_text())["vertices"]
    table_matches = 0
    table_sigmas, pattern_sigmas = set(), set()
    for v in table:
        ours = decorated.decorated_permutation(v["M"])
        table_matches += show(ours) == show(v["sigma"])
        table_sigmas.add(show(v["sigma"]))
    for code in range(343):
        masks = [code % 7 + 1, (code // 7) % 7 + 1, code // 49 + 1]
        P = []
        for m in masks:
            bits = [(m >> k) & 1 for k in range(3)]
            P.append([b / sum(bits) for b in bits])
        pattern_sigmas.add(show(decorated.decorated_permutation(P)))
    print(f"    3-state agents: our map agrees with the paper's Appendix B table on {table_matches} of 27 vertices of M3.")
    print(f"    The table contains {len(table_sigmas)} distinct decorated permutations (the paper's text says 17); all 343")
    print(f"    transition patterns of 3-state agents give {len(pattern_sigmas)}.")

    sticky, switching = [[0.99, 0.01], [0.01, 0.99]], [[0.01, 0.99], [0.99, 0.01]]
    print("    probabilities are discarded: an agent that almost never changes experience and one that almost always does")
    print(f"    both map to {show(decorated.decorated_permutation(sticky))} ({show(decorated.decorated_permutation(switching))}).")

    print("    dependence on labelling (all 720 relabellings of 6-state chains):")
    all_agree = True
    rows = []
    perms = list(itertools.permutations(range(6)))
    for name, P in [("classes (1 3 5)(2 6)(4)", cycle_kernel(6, [[1, 3, 5], [2, 6], [4]])),
                    ("one 6-cycle", cycle_kernel(6, [[1, 2, 3, 4, 5, 6]])),
                    ("classes (1 2)(3 4)(5 6)", cycle_kernel(6, [[1, 2], [3, 4], [5, 6]])),
                    ("random irreducible agent", random_kernel(6, mulberry32(3)))]:
        covariant = identical = agree = 0
        base = show(decorated.decorated_permutation(P))
        for p in perms:
            c, pr = decorated.is_covariant(P, p), preserves_cyclic_order(P, p)
            covariant += c
            agree += c == pr
            identical += show(decorated.decorated_permutation(decorated.relabel(P, p))) == base
        all_agree = all_agree and agree == len(perms)
        rows.append({"name": name, "covariant": covariant, "identical": identical, "base": base})
        print(f"      {name:<26} σ = {base:<22} renames consistently: {covariant:>3}/720   identical list: {identical:>3}/720")
    print("    rule: σ renames consistently exactly when the relabelling keeps the cyclic order of every recurrent")
    print(f"    class of >= 3 states ({'holds for every relabelling tested' if all_agree else 'FAILS for some relabelling'}). For a single class, σ is the same list for")
    print("    every chain and every numbering: it records the class, not the dynamics inside it.")
    return {"reproduces": reproduces, "table_matches": table_matches, "vertices": len(table_sigmas),
            "patterns": len(pattern_sigmas), "all_agree": all_agree, "rows": rows}


# ────────────── B. trace logic ──────────────

def labelled(states, P):
    return {"states": list(states), "P": markov.as_matrix(P)}


def trace_on(K, window):
    idx = [K["states"].index(s) for s in window]
    return labelled(window, trace.trace_chain(K["P"], idx))


def trace_leq(A, B, tol=1e-9):
    """A <=_t B: A's states are a subset of B's and A is B's trace there."""
    if not all(s in B["states"] for s in A["states"]):
        return False
    return trace.is_trace_of(A["P"], B["P"], [B["states"].index(s) for s in A["states"]], tol)


def subsets(items):
    return [[x for i, x in enumerate(items) if (m >> i) & 1] for m in range(1, 1 << len(items))]


def part_b():
    print("\n  B. Trace logic")
    r = mulberry32(16)

    X, G, W = 3, 2, 3

    def row(n):
        u = [r.random() + 0.05 for _ in range(n)]
        return [v / sum(u) for v in u]
    D = [row(G) for _ in range(X)]
    A = [row(W) for _ in range(G)]
    P = [row(X) for _ in range(W)]
    Q = trace.qualia_kernel(D, A, P)
    print(f"    qualia kernel Q = D·A·P of a (3 experiences, 2 actions, 3 world states) agent: stochastic {str(markov.is_stochastic(Q)).lower()}")

    N = labelled(["a", "b", "c", "d", "e"], random_kernel(5, r))
    obs = trace_on(N, ["a", "c", "d"])
    impostor = labelled(["a", "c", "d"], random_kernel(3, r))
    observes, fake = trace_leq(obs, N), trace_leq(impostor, N)
    print(f'    "A observes N" (A\'s kernel is N\'s trace on A\'s window): true observer {str(observes).lower()}, '
          f"random kernel on the same window {str(fake).lower()}")

    N4 = labelled(["a", "b", "c", "d"], random_kernel(4, r))
    windows = subsets(N4["states"])
    traces = [trace_on(N4, w) for w in windows]
    order_ok = pairs = 0
    for i in range(len(windows)):
        for j in range(len(windows)):
            subset = all(s in windows[j] for s in windows[i])
            order_ok += trace_leq(traces[i], traces[j]) == subset
            pairs += 1
    print(f'    locally Boolean: for a 4-experience agent N, "trace ≤ trace" matches "window ⊆ window" in {order_ok} of {pairs} pairs,')
    print("    so the 15 traces of N (plus a zero) form the Boolean algebra of its windows: join = trace on the union,")
    print("    meet = trace on the intersection, complement = trace on the complementary window.")

    K1 = labelled(["a", "b"], [[0.9, 0.1], [0.4, 0.6]])
    K2 = labelled(["a", "b"], [[0.2, 0.8], [0.7, 0.3]])
    la, lb = labelled(["a"], [[1.0]]), labelled(["b"], [[1.0]])
    lower_ok = all(trace_leq(l, K1) and trace_leq(l, K2) for l in (la, lb))
    incomparable = not trace_leq(K1, K2) and not trace_leq(K2, K1) and not trace_leq(la, lb) and not trace_leq(lb, la)
    print(f'    globally not a lattice: agents K1 ≠ K2 on {{a, b}} both have "trace on {{a}}" and "trace on {{b}}" below them ({str(lower_ok).lower()}),')
    print(f"    and all four are pairwise incomparable ({str(incomparable).lower()}). So K1 and K2 have two maximal common lower bounds")
    print("    (no meet), and {a}, {b} have two minimal common upper bounds (no join). There is also no top element.")

    tested = preserved = 0
    for _ in range(200):
        n = 4 + int(r.random() * 3)
        B = labelled(["a", "b", "c", "d", "e", "f"][:n], random_kernel(n, r))
        win = [s for s in B["states"] if r.random() < 0.6]
        if not win:
            continue
        Aw = trace_on(B, win)
        tested += 1
        preserved += trace.lebesgue_leq(trace.stationary_measure(Aw["states"], Aw["P"]),
                                        trace.stationary_measure(B["states"], B["P"]))
    muN = trace.stationary_measure(N["states"], N["P"])
    muN2 = trace.stationary_measure(N["states"], N["P"] @ N["P"])
    same_measure = all(abs(p - muN2[s]) < 1e-9 for s, p in muN.items())
    pi_obs = trace.stationary_measure(obs["states"], obs["P"])
    pi3 = [pi_obs[s] for s in obs["states"]]
    independent = labelled(obs["states"], [pi3 for _ in obs["states"]])
    converse = trace.lebesgue_leq(trace.stationary_measure(independent["states"], independent["P"]), muN) \
        and not trace_leq(independent, N)
    print(f"    homomorphism to Lebesgue logic: A ≤_t B implied π_A ≤_L π_B in {preserved} of {tested} random (agent, window) pairs.")
    print(f"    It is not injective: N and N² have the same stationary measure ({str(same_measure).lower()}); and the converse fails:")
    print(f"    a kernel with the observer's stationary measure but different dynamics satisfies ≤_L without being a trace ({str(converse).lower()}).")
    return {"order_ok": order_ok, "pairs": pairs, "lower_ok": lower_ok, "incomparable": incomparable,
            "preserved": preserved, "tested": tested, "same_measure": same_measure, "converse": converse,
            "observes": observes, "impostor": fake}


# ────────────── C. physics proposals ──────────────

def part_c():
    print("\n  C. Physics proposals from the trace-logic papers (proposals, not theorems), stress-tested")
    r = mulberry32(17)
    P1, P2 = random_kernel(3, r), random_kernel(4, r)
    h1, h2, h12 = markov.entropy_rate(P1), markov.entropy_rate(P2), markov.entropy_rate(markov.kron(P1, P2))
    add_err = abs(h12 - (h1 + h2))
    print("    mass ↔ entropy rate. Mass of independent systems adds; so does entropy rate under ⊗:")
    print(f"      h(P1) + h(P2) = {f(h1 + h2, 4)}, h(P1 ⊗ P2) = {f(h12, 4)} (difference {tiny(add_err)}): passes.")
    hs = [{"c": c, "h": markov.entropy_rate(coupled_rings(8, c))} for c in (0, 0.3, 0.6, 0.9)]
    print("      Bound systems in physics weigh less than their parts (mass defect). Three coupled ring agents (experiment 14):")
    print("      entropy rate " + " → ".join(f"{f(x['h'])} (c = {x['c']})" for x in hs) + ".")
    defect = hs[-1]["h"] < hs[0]["h"]
    print("      " + ("Binding lowers it: qualitatively consistent with a mass defect." if defect
                     else "Binding does not lower it: no mass-defect analogue."))

    violations = triples = 0
    asym = 0.0
    for _ in range(30):
        K = markov.commute_times(random_kernel(6, r))
        asym = max(asym, float(np.max(np.abs(K - K.T))))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    triples += 1
                    if K[i, k] > K[i, j] + K[j, k] + 1e-9:
                        violations += 1
    print("    speed ↔ commute time. For speed to be distance over time, commute time must behave like a distance:")
    print(f"      symmetric (largest asymmetry {tiny(asym)}), triangle inequality violated in {violations} of {triples} triples: passes.")
    return {"add_err": add_err, "defect": defect, "violations": violations, "hs": hs}


def main():
    t0 = time.time()
    print("=" * 78)
    print("Trace logic and decorated permutations")
    print("=" * 78)
    A, B, C = part_a(), part_b(), part_c()

    print("\n" + "─" * 78)
    print("Findings")
    print("─" * 78)
    print(f"  1. The decorated-permutation map is implemented as defined and reproduces the paper's example{'' if A['reproduces'] else ' (NOT)'}.")
    print("     σ is fixed by two things only: which experiences form recurrent classes, and the cyclic order in which")
    print("     they are numbered. All probabilities are discarded, the dynamics inside a class is invisible, and σ")
    print(f"     renames consistently only for numberings that keep each class's cyclic order ({A['rows'][1]['covariant']} of 720 for one class")
    print('     of 6). Scattering-amplitude geometry also needs a cyclic order ("colour ordering"), so this is a')
    print("     requirement, not a bug: to reach physics, the program needs a principle that orders experiences.")
    print(f"     It agrees with all {A['table_matches']} entries of the paper's Appendix B table, which lists {A['vertices']} distinct decorated")
    print("     permutations for M3, not the 17 stated in the text (a miscount; the table itself is correct).")
    print(f"  2. Trace logic behaves as claimed: locally Boolean ({B['order_ok']}/{B['pairs']}), globally not a lattice (no meets, joins or")
    print(f"     top), and the stationary measure carries the trace order into Lebesgue logic ({B['preserved']}/{B['tested']}); the map is")
    print("     neither injective nor reversible. This is not quantum logic: it fails far more than distributivity.")
    print(f"  3. The proposals pass these first consistency checks: entropy rate adds like mass ({tiny(C['add_err'])}) and drops")
    print(f"     with binding{'' if C['defect'] else ' (NOT)'}; commute time is a genuine distance ({C['violations']} triangle violations). Passing is necessary,")
    print("     not sufficient: nothing here derives a mass spectrum or a speed limit.")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
