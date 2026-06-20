from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..core import (
    ExperienceTrie,
    MetaTrie,
    SelfTokenState,
    TraceBuffer,
    TraceEvent,
    ExperienceLexicon,
)
from ..agent import ConsciousAgent, ExperienceSpace


def _array_to_json(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def _json_to_array(s: str, dtype: type = np.float32, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(s), dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _serialize_trie(trie: ExperienceTrie) -> dict:
    nodes = []

    def _collect(node, path):
        if node is not trie.root:
            nodes.append({
                "path": list(path),
                "visit_count": node.visit_count,
                "prediction_errors": node.prediction_errors,
                "mean_prediction_error": node.mean_prediction_error,
                "word_binding": node.word_binding,
            })
        for child_state, child in node.children.items():
            path.append(child_state)
            _collect(child, path)
            path.pop()

    _collect(trie.root, [])
    return {"max_depth": trie.max_depth, "nodes": nodes}


def _deserialize_trie(data: dict) -> ExperienceTrie:
    trie = ExperienceTrie(max_depth=data.get("max_depth", 10))
    for nd in data.get("nodes", []):
        path = nd["path"]
        trie.insert(path)
        node = trie.lookup(path)
        if node is not None:
            node.visit_count = nd["visit_count"]
            node.prediction_errors = list(nd["prediction_errors"])
            node.mean_prediction_error = nd["mean_prediction_error"]
            node.word_binding = nd.get("word_binding")
    return trie


def _serialize_meta_trie(mt: MetaTrie) -> dict:
    registry = {}
    for mid, snap in mt._registry.items():
        registry[str(mid)] = {
            "state_ids": list(snap.state_ids),
            "mean_prediction_error": snap.mean_prediction_error,
            "timestamp": snap.timestamp,
        }
    token_registry = {}
    if hasattr(mt, "_token_registry"):
        for mid, counts in mt._token_registry.items():
            token_registry[str(mid)] = dict(counts)
    return {
        "trie": _serialize_trie(mt.trie),
        "registry": registry,
        "last_meta_state": mt.last_meta_state,
        "snapshot_window": mt._snapshot_window,
        "max_depth": mt.trie.max_depth,
        "token_registry": token_registry,
    }


def _deserialize_meta_trie(data: dict) -> MetaTrie:
    mt = MetaTrie(
        snapshot_window=data.get("snapshot_window", 10),
        max_depth=data.get("max_depth", 10),
    )
    mt._trie = _deserialize_trie(data["trie"])
    for mid_str, snap_data in data.get("registry", {}).items():
        mid = int(mid_str)
        mt._registry[mid] = type("snap", (), {
            "state_ids": tuple(snap_data["state_ids"]),
            "mean_prediction_error": snap_data["mean_prediction_error"],
            "timestamp": snap_data["timestamp"],
        })()
    mt._last_meta_state = data.get("last_meta_state")
    token_registry = data.get("token_registry", {})
    if token_registry:
        mt._token_registry = {int(k): dict(v) for k, v in token_registry.items()}
    return mt


def _serialize_self_token(st: SelfTokenState) -> dict:
    return {
        "token": st.token,
        "referent_meta_state_id": st.referent_meta_state_id,
        "stationary_prob": st.stationary_prob,
        "locked": st.locked,
        "lock_generation": st.lock_generation,
        "lock_threshold": st.lock_threshold,
        "consecutive_above_threshold": st.consecutive_above_threshold,
        "lock_consecutive_required": st.lock_consecutive_required,
        "stability_history": list(st.stability_history),
        "protection_radius": st.protection_radius,
    }


def _deserialize_self_token(data: dict) -> SelfTokenState:
    st = SelfTokenState(
        token=data.get("token", "I"),
        referent_meta_state_id=data.get("referent_meta_state_id"),
        stationary_prob=data.get("stationary_prob", 0.0),
        locked=data.get("locked", False),
        lock_generation=data.get("lock_generation"),
        lock_threshold=data.get("lock_threshold", 0.25),
        consecutive_above_threshold=data.get("consecutive_above_threshold", 0),
        lock_consecutive_required=data.get("lock_consecutive_required", 3),
        protection_radius=data.get("protection_radius", 2),
    )
    st.stability_history = list(data.get("stability_history", []))
    return st


def _serialize_trace_buffer(buf: TraceBuffer) -> dict:
    events = []
    for e in buf:
        events.append({
            "from_state": e.from_state,
            "to_state": e.to_state,
            "timestamp": e.timestamp,
            "prediction": e.prediction,
            "prediction_correct": e.prediction_correct,
            "prediction_error": e.prediction_error,
            "token": e.token,
        })
    return {"maxlen": buf.maxlen, "events": events}


def _deserialize_trace_buffer(data: dict) -> TraceBuffer:
    buf = TraceBuffer(maxlen=data["maxlen"])
    for ed in data["events"]:
        buf.append(TraceEvent(
            from_state=ed["from_state"],
            to_state=ed["to_state"],
            timestamp=ed["timestamp"],
            prediction=ed["prediction"],
            prediction_correct=ed["prediction_correct"],
            prediction_error=ed["prediction_error"],
            token=ed.get("token"),
        ))
    return buf


def serialize(agent: ConsciousAgent, path: str) -> None:
    state = {
        "ca_version": "1.0",
        "agent_id": agent.agent_id,
        "generation": agent.generation,
        "step": agent.step_count,
        "components": {
            "trace_buffer": _serialize_trace_buffer(agent.experience.trace_buffer),
            "experience_trie": _serialize_trie(agent.experience.trie),
            "meta_trie": _serialize_meta_trie(agent.experience.meta_trie),
            "self_token": _serialize_self_token(agent.experience.self_token),
            "lexicon": agent.experience.lexicon.to_dict(),
        },
        "config": {
            "trace_buffer_length": agent.experience.trace_buffer.maxlen,
            "trie_max_depth": agent.experience.trie.max_depth,
            "meta_observation_interval": agent.meta_observation_interval,
            "lock_threshold": agent.experience.self_token.lock_threshold,
            "lock_consecutive_required": agent.experience.self_token.lock_consecutive_required,
            "protection_radius": agent.experience.self_token.protection_radius,
        },
        "metadata": {
            "cycle_level": agent.cycle_level,
            "constituent_ids": sorted(agent.constituent_ids),
            "leaf_constituent_ids": sorted(agent.leaf_constituent_ids),
            "combined": agent._combined,
        },
    }

    path = str(path)
    tmp_path = str(Path(path).parent / f"tmp_{uuid.uuid4().hex}.json")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, cls=_NumpyEncoder)
    os.replace(tmp_path, path)


def deserialize(path: str) -> ConsciousAgent:
    with open(path) as f:
        state = json.load(f)

    comp = state.get("components", {})
    exp = ExperienceSpace(
        trace_buffer=_deserialize_trace_buffer(comp.get("trace_buffer", {"maxlen": 50, "events": []})),
        trie=_deserialize_trie(comp.get("experience_trie", {"max_depth": 10, "nodes": []})),
        meta_trie=_deserialize_meta_trie(comp.get("meta_trie", {})),
        self_token=_deserialize_self_token(comp.get("self_token", {})),
        lexicon=ExperienceLexicon.from_dict(comp.get("lexicon", {"entries": []})),
    )

    meta = state.get("metadata", {})
    agent = ConsciousAgent(
        agent_id=state.get("agent_id", "unknown"),
        experience=exp,
        generation=state.get("generation", 0),
        step_count=state.get("step", 0),
        meta_observation_interval=state.get("config", {}).get("meta_observation_interval", 20),
        constituent_ids=frozenset(meta.get("constituent_ids", [])),
        leaf_constituent_ids=frozenset(meta.get("leaf_constituent_ids", [])),
        cycle_level=meta.get("cycle_level", 0),
    )
    agent._combined = meta.get("combined", False)
    return agent


def clone(agent: ConsciousAgent, new_id: str | None = None) -> ConsciousAgent:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp = f.name
    try:
        serialize(agent, tmp)
        cloned = deserialize(tmp)
        cloned.agent_id = new_id or f"{agent.agent_id}_clone"
        return cloned
    finally:
        os.unlink(tmp)


def fingerprint(agent: ConsciousAgent) -> str:
    import hashlib
    comp = {
        "trie": _serialize_trie(agent.experience.trie),
        "meta_trie": _serialize_meta_trie(agent.experience.meta_trie),
        "self_token": _serialize_self_token(agent.experience.self_token),
    }
    return hashlib.sha256(json.dumps(comp, sort_keys=True).encode()).hexdigest()


def save_agent(agent: ConsciousAgent, directory: str = "./souls") -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = str(Path(directory) / f"agent_{agent.agent_id}_gen{agent.generation:06d}_step{agent.step_count:010d}.soul")
    serialize(agent, path)
    return path


def load_agent(path: str) -> ConsciousAgent:
    return deserialize(path)


def load_latest(soul_dir: str, agent_id: str) -> ConsciousAgent | None:
    pattern = f"agent_{agent_id}_*.soul"
    from pathlib import Path as P
    files = sorted(P(soul_dir).glob(pattern))
    if not files:
        return None
    return deserialize(str(files[-1]))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return _array_to_json(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
