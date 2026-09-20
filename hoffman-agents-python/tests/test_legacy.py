import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conscious_agent import ConsciousAgent, SimpleWorld, combine, fuse
from conscious_agent.io import deserialize

FIXTURES = Path(__file__).parent / "fixtures"


def test_legacy_reproduces_2_1_2_exactly():
    # 2.x used the per-process salted builtin hash(), so the golden run needs
    # PYTHONHASHSEED pinned in a fresh interpreter.
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    out = subprocess.run(
        [sys.executable, str(FIXTURES / "capture_legacy.py"), json.dumps({"math_version": "legacy"})],
        check=True, capture_output=True, env=env,
    ).stdout
    assert json.loads(out) == json.loads((FIXTURES / "legacy-2.1.2.json").read_text())


def test_loads_2x_soul_as_legacy():
    agent = deserialize(str(FIXTURES / "legacy-2.1.2.soul"))
    assert agent.math_version == "legacy"
    assert agent.is_i_locked and agent.step_count == 120
    agent.run(20)


def test_legacy_combination_keeps_tagging():
    import random

    def mk(agent_id, seed):
        a = ConsciousAgent(agent_id=agent_id, world=SimpleWorld(seed=seed), _rng=random.Random(seed), math_version="legacy")
        a.run(100)
        return a

    ab = combine(mk("A", 1), mk("B", 2))
    assert ab.math_version == "legacy"
    assert all((mid >> 28) in (1, 2) for mid in ab.experience.meta_trie._registry)
    assert len(fuse(ab)) == 2


def test_rejects_unknown_math_version():
    with pytest.raises(ValueError, match="Invalid math_version"):
        ConsciousAgent(agent_id="x", math_version="v9")
