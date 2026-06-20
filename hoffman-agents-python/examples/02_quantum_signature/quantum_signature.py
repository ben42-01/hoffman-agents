"""
Quantum Signature — Spectral Analysis of Tree-of-Life Combination Run

Agents interact in a shared network, observing each other's outputs.
When ripe, they combine via ⊗. Meta-trie transition matrices are
analyzed for quantum-like signatures.

Expected result:
  Base agents (level 0):     gap ~1.0, db_err ~0.0  (classical)
  Combined agents (level 1):  gap collapses < 0.3    (quantum-like)
  Higher levels (level 2+):   gap recovers            (classical limit)
"""
from conscious_agent import ConsciousAgent, WorldState, combine
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


def detailed_balance_error(P):
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(500):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-10:
            break
        pi = pi_new
    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            if pi[i] > 0 and pi[j] > 0:
                lhs = pi[i] * P[i, j]
                rhs = pi[j] * P[j, i]
                if abs(lhs + rhs) > 1e-12:
                    errors.append(abs(lhs - rhs) / (lhs + rhs))
    return np.mean(errors) if errors else 0.0


def analyze(agents, label):
    """Analyze meta-trie spectra for all agents at current state."""
    by_level = {}
    print(f"\n  {'─' * 50}")
    print(f"  {label}")
    print(f"  {'─' * 50}")
    print(f"  {'Agent':<22s} {'Lvl':<4s} {'States':<7s} {'Gap':<10s} {'DB Err':<10s}")
    for aid, agent in sorted(agents.items()):
        P = extract_meta_matrix(agent)
        if P is not None:
            gap = spectral_gap(P)
            dbe = detailed_balance_error(P)
            gap_str = f"{gap:.4f}"
            dbe_str = f"{dbe:.4f}"
            lvl = agent.cycle_level
            by_level.setdefault(lvl, []).append((gap, dbe))
        else:
            gap_str = "N/A"
            dbe_str = "N/A"
            lvl = agent.cycle_level
        print(f"  {aid:<22s} {lvl:<4d} {agent.experience.meta_trie.registry_size:<7d} {gap_str:<10s} {dbe_str:<10s}")
    return by_level


def run_experiment(n_base=6, n_interaction_rounds=200):
    np.random.seed(42)
    t0 = time.time()
    print("=" * 66)
    print("Quantum Signature — Tree-of-Life Spectral Analysis")
    print("=" * 66)
    print(f"\n{n_base} base agents, {n_interaction_rounds} interaction rounds...")

    # Phase 1: Create isolated agents — each in its own independent world
    agents = {}
    for i in range(n_base):
        aid = f"CA_{i:03d}"
        agent = ConsciousAgent(agent_id=aid)
        agents[aid] = agent
        for t in range(30):
            ws = WorldState.from_sequence("world", [f"seed_{i}_{t}"])
            agent.step(ws)

    # Snapshot: pure base agents before any interaction
    pre = analyze(agents, "Phase 1: Isolated agents (pure classical baseline)")

    # Interaction rounds: agents observe each other's outputs
    snapshot_taken = False
    for rnd in range(n_interaction_rounds):
        outputs = {}
        for aid, agent in agents.items():
            outputs[aid] = agent.get_output()

        for aid, agent in agents.items():
            for other_aid, other_output in outputs.items():
                if other_aid != aid:
                    ws = WorldState(sequences={other_aid: other_output})
                    agent.step(ws)

        # Snapshot RIGHT before first combination: interacting but uncombined
        if rnd == 19 and not snapshot_taken:
            pre_combo = analyze(agents, "Phase 2: Interacting, pre-combination")
            snapshot_taken = True

        # Every 20 rounds, try combining ripe agents
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

    # Phase 3: Post-combination analysis
    post = analyze(agents, "Phase 3: Post-combination hierarchy")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s\n")

    by_level = post

    # Cross-level summary
    print(f"\n{'─' * 66}")
    print("Cross-Level Quantum Signature Summary")
    print(f"{'─' * 66}")

    prev_gap = None
    for lvl in sorted(by_level.keys()):
        gaps = [g for g, _ in by_level[lvl]]
        dbes = [d for _, d in by_level[lvl]]
        mean_gap = np.mean(gaps)
        mean_db = np.mean(dbes)

        change = ""
        if prev_gap is not None:
            if mean_gap > prev_gap + 0.05:
                change = f"  ↑ recovery from {prev_gap:.3f}"
            elif mean_gap < prev_gap - 0.05:
                change = f"  ↓ collapse"

        label = "Base" if lvl == 0 else f"Level {lvl}"
        tag = ""
        if mean_gap < 0.3 and mean_db > 0.1:
            tag = "  ← QUANTUM-LIKE"
        elif mean_gap > 0.85 and mean_db < 0.15:
            tag = "  ← CLASSICAL"
        print(f"  {label:<8s} ({len(by_level[lvl]):>2d} agents)  gap={mean_gap:.4f}  db_err={mean_db:.4f}{tag}{change}")
        prev_gap = mean_gap

    print(f"\n  Interpretation:")
    print(f"    gap~1.0, db_err~0.0  = classical, reversible (deterministic)")
    print(f"    gap~0.0, db_err>0.1  = quantum-like (stochastic, irreversible)")
    print(f"    gap recovery at higher levels = classical limit of quantum systems")

    return by_level


if __name__ == "__main__":
    run_experiment()
