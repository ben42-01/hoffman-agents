"""
Quantum Signature? — Spectral Analysis of Combination, with Classical Controls

The 2.x version of this experiment reported a "quantum-like" collapse of the
spectral gap at combination levels 1-2. That signal was an artifact:
  - combined agents carry their constituents' chains as disconnected pieces;
    a reducible chain has |lambda_2| = 1, so its gap is 0 by definition
  - unobserved rows were turned into absorbing states
  - agents only combined because the 2.x "I" lock fired in any world
and the criterion itself (small gap + detailed-balance violation) is met by
ordinary classical chains, e.g. a clock.

This version asks answerable questions:
  1. Classical controls: what do gap and irreversibility look like for chains
     that are classical by construction?
  2. Agents: gap (1 - |lambda_2|), period and irreversibility of each agent's
     own recurrent meta-state chain (MetaTrie.ergodic_diagnostics).
  3. Tensor-product prediction: for independent agents A, B the product kernel
     M_A (x) M_B has |lambda_2| = max(|lambda_2(A)|, |lambda_2(B)|). Does the
     combined agent's learned chain match that prediction?
  4. Fusion: does fuse() restore the constituents' chains exactly?

Every quantity here describes a classical Markov chain. Gap measures mixing
speed and irreversibility measures net probability circulation; neither is
evidence of quantum behaviour. See ../09_double_slit_analogy for what that
would require.

Uses the same seeds as the Node example and prints the same numbers.
"""
import time

import numpy as np

from conscious_agent import ConsciousAgent, MarkovKernel, WorldState, combine, fuse, markov
from conscious_agent.combination import meta_kernel, product_kernel

N_BASE = 8
ISOLATED_STEPS = 400
ROUNDS = 200
COMBINE_EVERY = 20


# ────────────── measurements ──────────────

def analyse_kernel(P) -> dict:
    P = np.asarray(P, dtype=np.float64)
    pi = markov.stationary(P)["pi"]
    closed = len(markov.closed_classes(P))
    lambda2 = 1.0 if closed > 1 else markov.second_eigenvalue_modulus(P, pi)
    return {"n": len(P), "closed": closed, "period": None if closed > 1 else markov.period(P, 0),
            "gap": 1.0 - lambda2, "irrev": markov.irreversibility(P, pi)}


def analyse_agent(agent):
    """The agent's own recurrent meta-chain. `evidence` uses the same rule as the
    "I" lock: at least 20 transitions and 2 per state, otherwise the estimate is noise."""
    K = meta_kernel(agent)
    if K is None:
        return None
    d = agent.experience.meta_trie.ergodic_diagnostics()
    r = analyse_kernel(K.matrix)
    r["transitions"] = d["n_transitions"]
    r["evidence"] = d["n_transitions"] >= max(20, 2 * r["n"])
    return r


def counts_of(agent):
    return agent.experience.meta_trie.transition_counts()[1].tolist()


def closed_classes_including_inherited(agent) -> int:
    """What 2.x effectively measured: all transitions, inherited pieces included."""
    _, counts = agent.experience.meta_trie.transition_counts(include_inherited=True)
    kept = markov.prune_unobserved_rows(counts)
    return len(markov.closed_classes(markov.sub_matrix(counts, kept))) if kept else 0


def f(x, d=3):
    return "  -  " if x is None else f"{x:.{d}f}"


# ────────────── 1. classical controls ──────────────

def controls():
    def cycle(n, forward, stay):
        P = np.zeros((n, n))
        for i in range(n):
            P[i, i] += stay
            P[i, (i + 1) % n] += forward
            P[i, (i - 1) % n] += 1 - forward - stay
        return P

    K = MarkovKernel(["a", "b"], [[0.9, 0.1], [0.3, 0.7]])
    two_pieces = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]]
    return [
        ("clock: 10-cycle, forward 0.95", cycle(10, 0.95, 0.05)),
        ("lazy symmetric walk on 10-cycle", cycle(10, 0.25, 0.5)),
        ("i.i.d. uniform over 10 states", np.full((10, 10), 0.1)),
        ("two disconnected coin chains", two_pieces),
        ("independent product K ⊗ K", K.tensor(K).matrix),
        ("single coin chain", [[0.5, 0.5], [0.5, 0.5]]),
    ]


# ────────────── 2-4. agents ──────────────

def print_agents(title, agents):
    print(f"\n  {title}")
    print(f"  {'agent':<16} lvl  locked  class  trans  period  gap     irrev   closed incl. inherited")
    for a in agents:
        r = analyse_agent(a)
        note = "" if r is None else "  (too little data)" if not r["evidence"] else \
            "  (periodic: gap 0 by definition)" if (r["period"] or 1) > 1 else ""
        period = r["period"] if r and r["period"] is not None else "-"
        print(f"  {a.agent_id:<16} {a.cycle_level:<4} {str(a.is_i_locked).lower():<7} {r['n'] if r else 0:<6} "
              f"{int(r['transitions']) if r else 0:<6} {period!s:<7} {f(r and r['gap'])}   {f(r and r['irrev'])}   "
              f"{closed_classes_including_inherited(a)}{note}")


def run():
    t0 = time.time()
    print("=" * 78)
    print("Quantum Signature? — spectral analysis of combination, with classical controls")
    print("=" * 78)

    print("\n  1. Classical controls (classical by construction)")
    print(f"  {'chain':<34} states  closed  gap     irrev")
    for name, P in controls():
        r = analyse_kernel(P)
        print(f"  {name:<34} {r['n']:<7} {r['closed']:<7} {f(r['gap'])}   {f(r['irrev'])}")
    print('  → small gap + irreversibility (the 2.x "quantum" criterion) is a plain clock;')
    print("    gap 0 is what any chain made of disconnected pieces gives.")

    agents: dict[str, ConsciousAgent] = {}
    for i in range(N_BASE):
        aid = f"CA_{i:03d}"
        agent = ConsciousAgent(agent_id=aid, seed=i + 1)
        for t in range(ISOLATED_STEPS):
            agent.step(WorldState(sequences={"world": [f"s{i}_{t}"]}))
        agents[aid] = agent
    print_agents(f"2a. Isolated agents ({ISOLATED_STEPS} steps each)", list(agents.values()))

    combinations = []
    available = list(agents)
    for rnd in range(1, ROUNDS + 1):
        outputs = {aid: a.get_output() for aid, a in agents.items()}
        for aid, agent in agents.items():
            for other, seq in outputs.items():
                if other != aid:
                    agent.step(WorldState(sequences={other: seq}))
        # Combination on a fixed schedule. (Gating on the "I" lock, as 2.x did, never
        # fires here in v3: this world has no dominant experiential attractor.)
        if rnd % COMBINE_EVERY == 0 and len(available) >= 2:
            ready = sorted(
                (aid for aid in available if meta_kernel(agents[aid]) is not None),
                key=lambda aid: (agents[aid].experience.trace_buffer.prediction_error_mean(window=5), aid),
            )
            for i in range(0, len(ready) - 1, 2):
                x, y = agents[ready[i]], agents[ready[i + 1]]
                prior = product_kernel(x, y)
                snapshots = {x.agent_id: counts_of(x), y.agent_id: counts_of(y)}
                c = combine(x, y)
                c.agent_id = f"L{c.cycle_level}_{ready[i][-3:]}_{ready[i + 1][-3:]}"
                agents[c.agent_id] = c
                available = [a for a in available if a not in (ready[i], ready[i + 1])] + [c.agent_id]
                combinations.append({"agent": c, "snapshots": snapshots, "round": rnd,
                                     "prior": analyse_kernel(prior.matrix) if prior is not None else None})
    print_agents(f"2b. After {ROUNDS} interaction rounds (combined every {COMBINE_EVERY} rounds)", list(agents.values()))
    print('  "closed incl. inherited" > 1 means a naive analysis of that trie is reducible → gap 0 (the 2.x artifact).')

    print("\n  3. Tensor-product prediction vs learned joint dynamics")
    print(f"  {'combined':<16} round  prior gap  learned gap  |Δ|     prior irrev  learned irrev")
    deltas = []
    for c in combinations:
        learned, prior = analyse_agent(c["agent"]), c["prior"]
        delta = abs(prior["gap"] - learned["gap"]) if prior and learned else None
        if delta is not None:
            deltas.append(delta)
        print(f"  {c['agent'].agent_id:<16} {c['round']:<6} {f(prior and prior['gap']):<10} {f(learned and learned['gap']):<12} "
              f"{f(delta):<7} {f(prior and prior['irrev']):<12} {f(learned and learned['irrev'])}")
    if deltas:
        print(f"  mean |Δgap| = {f(sum(deltas) / len(deltas))}")
    print("  The prior is exactly the product of the constituents' kernels (gap 0 if either is periodic).")
    print("  The learned chain is what the combined agent experiences afterwards in a shared world;")
    print("  nothing forces the two to agree, and here they do not.")

    print("\n  4. Fusion restores each constituent's chain as it was at combination time")
    for c in combinations:
        parts = fuse(c["agent"])
        ok = all(c["snapshots"][p.agent_id] == counts_of(p) for p in parts)
        names = " + ".join(p.agent_id for p in parts)
        print(f"  fuse({c['agent'].agent_id:<16}) → {names:<28} exact: {'yes' if ok else 'NO'}")

    print("\n" + "─" * 78)
    print("Summary by combination level (agents' own recurrent chains)")
    print("─" * 78)
    by_level: dict[int, list] = {}
    excluded = []
    for a in agents.values():
        r = analyse_agent(a)
        if r is None:
            continue
        if not r["evidence"] or (r["period"] or 1) > 1:
            excluded.append(f"{a.agent_id} ({'too little data' if not r['evidence'] else 'period ' + str(r['period'])})")
            continue
        by_level.setdefault(a.cycle_level, []).append(r)
    for lvl in sorted(by_level):
        rs = by_level[lvl]
        gap = sum(r["gap"] for r in rs) / len(rs)
        irrev = sum(r["irrev"] for r in rs) / len(rs)
        print(f"  level {lvl}  ({len(rs)} agent{'' if len(rs) == 1 else 's'})  gap={f(gap)}  irreversibility={f(irrev)}")
    if excluded:
        print(f"  excluded (gap not meaningful): {', '.join(excluded)}")
    print("\n  All of these are classical Markov chains: gap = mixing speed, irreversibility = net circulation.")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    run()
