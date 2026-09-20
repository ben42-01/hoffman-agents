"""Golden behaviour of the 2.1.2 math (run with PYTHONHASHSEED=0).

Captured once against 2.1.2; tests/test_legacy.py replays it with
math_version="legacy" in a subprocess using the same hash seed, because 2.x
used the per-process salted builtin hash().
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from conscious_agent import ConsciousAgent, SimpleWorld  # noqa: E402


def run(seed: int, **kwargs) -> dict:
    agent = ConsciousAgent(agent_id=f"CA_legacy_{seed}", world=SimpleWorld(n_states=50, seed=seed),
                           _rng=random.Random(seed), **kwargs)
    h = hashlib.sha256()
    lock_step = None
    for _ in range(300):
        out = agent.step()
        h.update((out.sequence_str + "|").encode())
        if out.i_locked and lock_step is None:
            lock_step = out.step
    return {
        "seed": seed,
        "lockStep": lock_step,
        "outputHash": h.hexdigest(),
        "registrySize": agent.experience.meta_trie.registry_size,
        "stationaryProb": agent.experience.self_token.stationary_prob,
        "lexiconSize": len(agent.experience.lexicon._entries),
    }


if __name__ == "__main__":
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    runs = [run(seed, **extra) for seed in (1, 2, 3)]
    if len(sys.argv) > 1:
        print(json.dumps(runs))
    else:
        (HERE / "legacy-2.1.2.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(json.dumps(runs, indent=2))
