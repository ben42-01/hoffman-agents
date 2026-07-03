"""
Experiment 6: Fitness Beats Truth — Tree of Life

Agents inhabit a hidden Markov world with a compressed interface (groups)
vs veridical perception (raw states). They interact and combine.
Tests whether combined agents prefer simplified interfaces,
supporting Hoffman's claim that perception is tuned for fitness, not truth.
"""
from conscious_agent import ConsciousAgent, WorldState, combine
import random
import statistics


def build_fbt_world(n_groups=5, states_per_group=4, seq_len=2000):
    n_states = n_groups * states_per_group
    seq = []
    current = 0
    rng = random.Random(42)
    for _ in range(seq_len):
        r = rng.random()
        if r < 0.5:
            current = current
        elif r < 0.8:
            gs = (current // states_per_group) * states_per_group
            current = gs + rng.randint(0, states_per_group - 1)
        else:
            current = rng.randint(0, n_states - 1)
        seq.append(current)
    iface = [s // states_per_group for s in seq]
    return {"raw": seq, "iface": iface, "n_groups": n_groups, "n_states": n_states}


def run_experiment():
    print("=" * 66)
    print("Experiment 6: Fitness Beats Truth — Tree of Life")
    print("=" * 66)

    world = build_fbt_world(5, 4, 300)
    n_iface, n_truth = 4, 4
    all_agents = {}

    for i in range(n_iface):
        aid = f"IFACE_{i:03d}"
        agent = ConsciousAgent(agent_id=aid)
        all_agents[aid] = agent
        for sid in world["iface"]:
            agent.step(WorldState.from_sequence("world", [f"g{sid}"]))

    for i in range(n_truth):
        aid = f"TRUTH_{i:03d}"
        agent = ConsciousAgent(agent_id=aid)
        all_agents[aid] = agent
        for sid in world["raw"]:
            agent.step(WorldState.from_sequence("world", [f"s{sid}"]))

    for rnd in range(200):
        outputs = {aid: ag.get_output() for aid, ag in all_agents.items()}
        for aid, ag in all_agents.items():
            for oa, o in outputs.items():
                if oa != aid:
                    ag.step(WorldState(sequences={oa: o}))

        if rnd > 0 and rnd % 20 == 0:
            ripe = [aid for aid, ag in all_agents.items()
                    if ag.experience.self_token.locked and not ag._combined]
            if len(ripe) >= 2:
                scored = sorted(ripe, key=lambda aid: all_agents[aid].experience.trace_buffer.prediction_error_mean(5))
                for i in range(0, len(scored) - 1, 2):
                    a, b = all_agents[scored[i]], all_agents[scored[i + 1]]
                    c = combine(a, b)
                    cid = f"L{c.cycle_level}_{scored[i]}_{scored[i + 1]}"
                    c.agent_id = cid
                    all_agents[cid] = c
                    a._combined = True
                    b._combined = True

    print(f"\n  {'─' * 60}")
    print(f"  Final Agent State")
    print(f"  {'─' * 60}")
    print(f"  {'Agent':<22s} {'Type':<10s} {'Lvl':<4s} {'States':<7s} {'Err':<8s} Locked")

    by_type = {"iface": [], "truth": [], "combined": []}
    for aid, ag in sorted(all_agents.items()):
        if aid.startswith("IFACE"):
            t = "iface"
        elif aid.startswith("TRUTH"):
            t = "truth"
        else:
            t = "combined"
        err = ag.experience.trace_buffer.prediction_error_mean(20)
        print(f"  {aid:<22s} {t:<10s} {ag.cycle_level:<4d} {ag.experience.meta_trie.registry_size:<7d} {err:<8.4f} {ag.experience.self_token.locked}")
        by_type[t].append(err)

    print(f"\n  Summary:")
    print(f"    Interface agents mean error:  {statistics.mean(by_type['iface']):.4f} (see {world['n_groups']} groups)")
    print(f"    Veridical agents mean error:  {statistics.mean(by_type['truth']):.4f} (see {world['n_states']} raw states)")
    print(f"    Combined agents mean error:   {statistics.mean(by_type['combined']):.4f}")
    if statistics.mean(by_type["iface"]) < statistics.mean(by_type["truth"]):
        print(f"  ✓ Interface dominates — less information -> better prediction (Hoffman confirmed)")
    else:
        print(f"  ✗ Veridical equal or better — no fitness advantage for compression")


if __name__ == "__main__":
    run_experiment()
