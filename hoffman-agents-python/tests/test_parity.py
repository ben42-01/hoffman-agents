"""Node <-> Python parity for the v3 math (requires node on PATH)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NODE_RUNNER = ROOT / "hoffman-agents-node" / "test" / "parity" / "run.js"
PY_RUNNER = Path(__file__).parent / "parity" / "run.py"


def _compare(a, b, path="$", tol=1e-9, diffs=None):
    diffs = [] if diffs is None else diffs
    if len(diffs) > 20:
        return diffs
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not (a == b or abs(a - b) <= tol * max(1.0, abs(a), abs(b))):
            diffs.append(f"{path}: {a} != {b}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]", tol, diffs)
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: missing on one side")
            else:
                _compare(a[k], b[k], f"{path}.{k}", tol, diffs)
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


@pytest.mark.skipif(shutil.which("node") is None or not NODE_RUNNER.exists(), reason="node library not available")
def test_node_python_parity():
    node = json.loads(subprocess.run(["node", str(NODE_RUNNER)], check=True, capture_output=True).stdout)
    py = json.loads(subprocess.run([sys.executable, str(PY_RUNNER)], check=True, capture_output=True).stdout)
    assert _compare(node, py) == []
    assert node["home_1"]["lockHistory"], "scenario exercises the lock"
