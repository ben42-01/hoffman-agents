"""Cross-language parity trace. Mirrors hoffman-agents-node/test/parity/run.js
line for line; both print the same JSON for the same scenarios."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from conscious_agent import ConsciousAgent, WorldState, combine, fuse  # noqa: E402
from conscious_agent.math import mulberry32  # noqa: E402


class HomeWorld:
    def __init__(self, seed, p_home, n):
        self.r, self.p, self.n = mulberry32(seed + 999), p_home, n

    def step(self):
        return WorldState.from_sequence("world", ["home" if self.r.random() < self.p else f"s{int(self.r.random() * self.n)}"])


class NoiseWorld:
    def __init__(self, seed, n):
        self.r, self.n = mulberry32(seed + 5000), n

    def step(self):
        return WorldState.from_sequence("world", [f"t{int(self.r.random() * self.n)}", f"u{int(self.r.random() * 3)}"])


def js_round(x: float) -> float:
    # Math.round(x * 1e9) / 1e9 (half up, like JavaScript)
    import math
    return math.floor(x * 1e9 + 0.5) / 1e9


def trace(agent, steps):
    outputs = []
    for _ in range(steps):
        o = agent.step()
        outputs.append([o.sequence_str, agent.experience.meta_trie.last_meta_state, o.i_locked, js_round(o.prediction_error)])
    d = agent.experience.meta_trie.ergodic_diagnostics()
    return {
        "outputs": outputs,
        "pi": [[sid, p] for sid, p in d["pi"].items()],
        "period": d["period"],
        "nTransitions": d["n_transitions"],
        "lockHistory": [[e["event"], e["generation"], e["referent"]] for e in agent.experience.self_token.lock_history],
        "registry": sorted(agent.experience.meta_trie._registry),
        "lexicon": [[e.label, e.output_token] for e in agent.experience.lexicon._entries.values()],
        "trieSize": agent.experience.trie.size(),
    }


def main():
    result = {}
    for seed in (1, 2, 3):
        result[f"home_{seed}"] = trace(ConsciousAgent(agent_id=f"H{seed}", seed=seed, world=HomeWorld(seed, 0.9, 20)), 1500)
        result[f"noise_{seed}"] = trace(ConsciousAgent(agent_id=f"N{seed}", seed=seed, world=NoiseWorld(seed, 40)), 1500)

    a = ConsciousAgent(agent_id="A", seed=11, world=HomeWorld(11, 0.9, 8))
    b = ConsciousAgent(agent_id="B", seed=12, world=HomeWorld(12, 0.95, 8))
    a.run(900)
    b.run(900)
    ab = combine(b, a)
    result["combine"] = {
        "id": ab.agent_id,
        "provenance": sorted([[mid, p["constituent_id"], p["local_id"]] for mid, p in ab.experience.meta_trie._provenance.items()]),
        "params": [ab.p_stable, ab.p_lexicon, ab.p_explore],
        "trace": [e.to_state for e in ab.experience.trace_buffer],
        "fused": [[p.agent_id, p.experience.meta_trie.transition_counts()[0], p.is_i_locked] for p in fuse(ab)],
    }
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
