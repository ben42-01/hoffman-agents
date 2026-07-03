"""
Kronecker Product Distance Test

Measures whether the combination operator ⊗ produces dynamics closer to
a tensor product (P₁ ⊗ P₂) or a direct sum (P₁ ⊕ P₂).

Run separately from the main experiment — computationally expensive.

Usage: python -m examples.02_markov_transition.kronecker_test
"""
from conscious_agent import ConsciousAgent, WorldState, combine, fuse
import numpy as np


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


def build_direct_sum(P1, P2):
    n, m = P1.shape[0], P2.shape[0]
    D = np.zeros((n + m, n + m))
    D[:n, :n] = P1
    D[n:, n:] = P2
    return D


def build_kronecker_product(P1, P2):
    return np.kron(P1, P2)


def run_test():
    print("=" * 66)
    print("Kronecker Product Distance Test")
    print("=" * 66)

    def _build_graph(agent_ids, conn):
        n = len(agent_ids)
        if conn >= n:
            return None
        import random
        graph = {}
        for aid in agent_ids:
            others = [x for x in agent_ids if x != aid]
            random.shuffle(others)
            graph[aid] = others[:conn]
        return graph

    n_base = 6
    connectivity = min(n_base, 10)
    agents = {}

    for i in range(n_base):
        aid = f"CA_{i:03d}"
        agent = ConsciousAgent(agent_id=aid)
        agents[aid] = agent
        for t in range(400):
            ws = WorldState.from_sequence("world", [f"seed_{i}_{t}"])
            agent.step(ws)

    def _interact_round(outputs, graph):
        for aid, ag in agents.items():
            targets = graph.get(aid, list(outputs.keys())) if graph else list(outputs.keys())
            for oa in targets:
                if oa != aid:
                    ag.step(WorldState(sequences={oa: outputs[oa]}))

    for rnd in range(60):
        outputs = {aid: ag.get_output() for aid, ag in agents.items()}
        graph = _build_graph(list(agents.keys()), connectivity)
        _interact_round(outputs, graph)

    results = []
    for rnd in range(10):
        outputs = {aid: ag.get_output() for aid, ag in agents.items()}
        graph = _build_graph(list(agents.keys()), connectivity)
        _interact_round(outputs, graph)

        ripe = [aid for aid, ag in agents.items()
                if ag.experience.self_token.locked and not ag._combined]
        if len(ripe) < 2:
            continue

        scored = sorted(ripe, key=lambda aid: agents[aid].experience.trace_buffer.prediction_error_mean(5))

        for i in range(0, len(scored) - 1, 2):
            a_id, b_id = scored[i], scored[i + 1]
            a, b = agents[a_id], agents[b_id]

            r1 = extract_meta_matrix(a)
            r2 = extract_meta_matrix(b)
            if r1 is None or r2 is None:
                continue
            if r1.shape[0] < 2 or r2.shape[0] < 2:
                continue

            combined = combine(a, b)
            cid = f"L{combined.cycle_level}_{a_id[-3:]}_{b_id[-3:]}"
            combined.agent_id = cid
            combined._agent_id = cid
            agents[cid] = combined
            a._combined = True
            b._combined = True

            rc = extract_meta_matrix(combined)
            if rc is None or rc.shape[0] < 2:
                continue

            P1, P2, Pc = r1, r2, rc
            K = build_kronecker_product(P1, P2)
            D = build_direct_sum(P1, P2)

            max_n = max(Pc.shape[0], K.shape[0], D.shape[0])
            pad_Pc = np.zeros((max_n, max_n))
            pad_K = np.zeros((max_n, max_n))
            pad_D = np.zeros((max_n, max_n))
            pad_Pc[:Pc.shape[0], :Pc.shape[0]] = Pc
            pad_K[:K.shape[0], :K.shape[0]] = K
            pad_D[:D.shape[0], :D.shape[0]] = D

            dK = np.linalg.norm(pad_Pc - pad_K, 'fro')
            dD = np.linalg.norm(pad_Pc - pad_D, 'fro')
            ratio = dK / dD if dD > 0 else float('inf')

            results.append({
                'child': cid,
                'n1': P1.shape[0], 'n2': P2.shape[0], 'nc': Pc.shape[0],
                'nK': K.shape[0], 'nD': D.shape[0],
                'distK': dK, 'distD': dD, 'ratio': ratio,
                'closer': 'PRODUCT' if ratio < 1 else 'SUM',
            })

    print(f"\n  {'─' * 66}")
    print(f"  Kronecker Distance Results")
    print(f"  {'─' * 66}")
    print(f"  {'Child':<16s} {'n₁':<5s} {'n₂':<5s} {'n_c':<5s} {'n_K':<5s} {'|Pc-P₁⊗P₂|':<15s} {'|Pc-P₁⊕P₂|':<15s} {'Ratio':<8s} Closer")
    sum_ratio = 0.0
    count = 0
    for r in results:
        print(f"  {r['child']:<16s} {r['n1']:<5d} {r['n2']:<5d} {r['nc']:<5d} {r['nK']:<5d} {r['distK']:<15.4f} {r['distD']:<15.4f} {r['ratio']:<8.4f} {r['closer']}")
        sum_ratio += r['ratio']
        count += 1
    if count > 0:
        avg = sum_ratio / count
        print(f"\n  Average ratio: {avg:.4f} ({'closer to PRODUCT' if avg < 1 else 'closer to SUM'})")
        print(f"  Ratio < 1 = Pc more product-like; Ratio > 1 = Pc more sum-like")


if __name__ == "__main__":
    run_test()
