"""
Markov Structural Transition — Tree-of-Life Spectral Analysis

Agents interact in a shared network, observing each other's outputs.
When ripe, they combine via ⊗. Meta-trie transition matrices are
analyzed for structural transitions (spectral gap, entropy production).

Expected result:
  Base agents (level 0):     gap ~1.0, EPR ~0.0  (fast mixing, reversible)
  Combined agents (level 1):  gap collapses < 0.3 (slow mixing)
  Higher levels (level 2+):   gap near 0, high EPR (structured, irreversible)
"""
from conscious_agent import ConsciousAgent, WorldState, combine, fuse
import numpy as np
import time


def extract_meta_matrix(agent):
    mt = agent.experience.meta_trie
    if mt.registry_size < 2:
        return None
    all_ids = sorted(mt._registry.keys())
    active = set()
    for sid in all_ids:
        node = mt.trie.lookup([sid])
        if node and node.children:
            active.add(sid)
    if mt.last_meta_state is not None:
        active.add(mt.last_meta_state)
    if len(active) < 2:
        return None

    state_ids = sorted(active)
    idx = {sid: i for i, sid in enumerate(state_ids)}
    n = len(state_ids)
    P = np.zeros((n, n))

    for state_id in state_ids:
        node = mt.trie.lookup([state_id])
        if node and node.children:
            total = sum(c.visit_count for c in node.children.values())
            if total > 0:
                for child_state, child_node in node.children.items():
                    if child_state in idx:
                        P[idx[state_id], idx[child_state]] = child_node.visit_count / total

    for i in range(n):
        if P[i].sum() == 0:
            P[i, i] = 1.0
    return P


def spectral_gap(P):
    eigvals = np.linalg.eigvals(P)
    mags = sorted(np.abs(eigvals), reverse=True)
    return 1.0 - mags[1] if len(mags) > 1 else 1.0


def mixing_time(gap):
    if gap <= 0:
        return float('inf')
    if gap >= 1:
        return 0.0
    t = -1.0 / np.log(1.0 - gap)
    return float('inf') if t > 1e6 else t


def entropy_production_rate(P):
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(500):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-12:
            break
        pi = pi_new

    epr = 0.0
    for i in range(n):
        if pi[i] <= 0:
            continue
        for j in range(n):
            if P[i, j] <= 0 or P[j, i] <= 0:
                continue
            epr += pi[i] * P[i, j] * np.log(P[i, j] / P[j, i])
    return epr


def analyze(agents, label):
    by_level = {}
    print(f"\n  {'─' * 60}")
    print(f"  {label}")
    print(f"  {'─' * 60}")
    print(f"  {'Agent':<22s} {'Lvl':<4s} {'States':<7s} {'Gap':<10s} {'MixTime':<10s} {'EPR':<10s}")
    for aid, agent in sorted(agents.items()):
        P = extract_meta_matrix(agent)
        if P is not None:
            gap = spectral_gap(P)
            epr = entropy_production_rate(P)
            mt = mixing_time(gap)
            gap_str = f"{gap:.4f}"
            epr_str = f"{epr:.4f}"
            mt_str = f"{mt:.1f}" if np.isfinite(mt) else "∞"
            lvl = agent.cycle_level
            by_level.setdefault(lvl, []).append((gap, epr, mt if np.isfinite(mt) else float('nan')))
        else:
            gap_str = "N/A"
            epr_str = "N/A"
            mt_str = "N/A"
            lvl = agent.cycle_level
        print(f"  {aid:<22s} {lvl:<4d} {agent.experience.meta_trie.registry_size:<7d} {gap_str:<10s} {mt_str:<10s} {epr_str:<10s}")
    return by_level


def _build_interaction_graph(agent_ids, connectivity):
    n = len(agent_ids)
    if connectivity >= n:
        return None
    import random
    graph = {}
    for aid in agent_ids:
        others = [x for x in agent_ids if x != aid]
        random.shuffle(others)
        graph[aid] = others[:connectivity]
    return graph


def run_experiment(n_base=8, n_interaction_rounds=400, connectivity=None):
    if connectivity is None:
        connectivity = n_base
    np.random.seed(42)
    t0 = time.time()
    print("=" * 66)
    print("Markov Structural Transition — Tree-of-Life Analysis")
    print("=" * 66)
    print(f"\n{n_base} base agents, {n_interaction_rounds} rounds, connectivity={'all' if connectivity >= n_base else connectivity}...")

    agents = {}
    for i in range(n_base):
        aid = f"CA_{i:03d}"
        agent = ConsciousAgent(agent_id=aid)
        agents[aid] = agent
        for t in range(400):
            ws = WorldState.from_sequence("world", [f"seed_{i}_{t}"])
            agent.step(ws)

    analyze(agents, "Phase 1: Isolated agents")

    def _interact_round(outputs, graph):
        for aid, agent in agents.items():
            targets = graph.get(aid, list(outputs.keys())) if graph else list(outputs.keys())
            for oa in targets:
                if oa != aid:
                    agent.step(WorldState(sequences={oa: outputs[oa]}))

    snapshot_taken = False
    for rnd in range(n_interaction_rounds):
        outputs = {}
        for aid, agent in agents.items():
            outputs[aid] = agent.get_output()

        graph = _build_interaction_graph(list(agents.keys()), connectivity)
        _interact_round(outputs, graph)

        if rnd == 39 and not snapshot_taken:
            analyze(agents, "Phase 2: Interacting (40 rounds)")
            snapshot_taken = True

        if rnd > 0 and rnd % 20 == 0:
            ripe = [aid for aid, ag in agents.items()
                    if ag.experience.self_token.locked
                    and not ag._combined]
            if len(ripe) >= 2:
                scored = sorted(ripe,
                    key=lambda aid: agents[aid].experience.trace_buffer.prediction_error_mean(5))
                for i in range(0, len(scored) - 1, 2):
                    a_id, b_id = scored[i], scored[i + 1]
                    a, b = agents[a_id], agents[b_id]
                    combined = combine(a, b)
                    cid = f"L{combined.cycle_level}_{a_id[-3:]}_{b_id[-3:]}"
                    combined.agent_id = cid
                    combined._agent_id = cid
                    agents[cid] = combined
                    a._combined = True
                    b._combined = True

    post = analyze(agents, "Phase 3: Post-combination")

    top_agents = [(aid, ag) for aid, ag in agents.items() if aid.startswith("L")]
    top_agents.sort()
    if top_agents:
        highest = top_agents[-1][1]
        fused = fuse(highest)
        fused_map = {f.agent_id: f for f in fused}
        analyze(fused_map, f"Phase 4: Fusion of {top_agents[-1][0]}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s\n")

    by_level = post

    print(f"\n{'─' * 66}")
    print("Cross-Level Summary")
    print(f"{'─' * 66}")

    prev_gap = None
    for lvl in sorted(by_level.keys()):
        gaps = [g for g, _, _ in by_level[lvl]]
        ep = [e for _, e, _ in by_level[lvl]]
        mean_gap = np.mean(gaps)
        mean_ep = np.mean(ep)

        change = ""
        if prev_gap is not None:
            if mean_gap > prev_gap + 0.05:
                change = f"  ↑ faster"
            elif mean_gap < prev_gap - 0.05:
                change = f"  ↓ slower"

        label = "Base" if lvl == 0 else f"Level {lvl}"
        tag = ""
        if mean_gap < 0.05:
            tag = "  ← SLOW MIXING"
        elif mean_gap > 0.8:
            tag = "  ← FAST MIXING"
        print(f"  {label:<8s} ({len(by_level[lvl]):>2d} agents)  gap={mean_gap:.4f}  epr={mean_ep:.4f}{tag}{change}")
        prev_gap = mean_gap

    print(f"\n  gap ~ 1.0 = fast mixing (near-uniform transitions)")
    print(f"  gap ~ 0.0 = slow mixing (near-reducible / cyclic structure)")
    print(f"  EPR > 0   = irreversible dynamics (directed flow)")

    return by_level


if __name__ == "__main__":
    run_experiment()
