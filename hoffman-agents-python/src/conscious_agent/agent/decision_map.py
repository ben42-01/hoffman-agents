from __future__ import annotations

import random
from typing import Literal

import numpy as np

from .experience_space import ExperienceSpace

OutputState = Literal["core", "lexicon", "explore", "idle"]

CORE_TOKENS = ["I", "notice", "familiar", "different", "wait"]

SIGNATURE_MATCH_THRESHOLD = 0.3


def _build_transition_signature(
    prev_id: int | None,
    curr_id: int,
    embedding_dim: int,
) -> np.ndarray:
    sig = np.zeros(embedding_dim, dtype=np.float64)
    if prev_id is not None:
        combined = hash(f"{prev_id}->{curr_id}")
        sig[combined % embedding_dim] = 1.0
    sig[curr_id % embedding_dim] = 1.0
    norm = np.linalg.norm(sig)
    if norm > 0:
        sig = sig / norm
    return sig


def _sample_lexicon_label(experience: ExperienceSpace, vocab_size: int = 5,
                          rng: random.Random = random._inst) -> str | None:
    entries = experience.lexicon.sorted_by_integration()
    if not entries:
        return None

    meta_id = experience.meta_trie.last_meta_state
    if meta_id is not None:
        predicted = experience.meta_trie.predict_token(meta_id, min_observations=3)
        if predicted is not None:
            for e in entries:
                if e.output_token == predicted:
                    e.encounter_count += 1
                    e.integration_depth = min(e.integration_depth + 0.02, 1.0)
                    return predicted

    curr_id = experience.last_world_state_id
    if curr_id is not None:
        prev_id = None
        recent = experience.trace_buffer.get_recent(2)
        if len(recent) >= 2:
            prev_id = recent[-2].to_state
        query_sig = _build_transition_signature(prev_id, curr_id, experience.lexicon.embedding_dim)
        best_label, best_dist = experience.lexicon.nearest_label(query_sig)
        if best_label and best_dist < SIGNATURE_MATCH_THRESHOLD:
            entry = experience.lexicon.lookup_by_label(best_label)
            if entry is not None and entry.integration_depth > 0.01:
                entry.encounter_count += 1
                entry.integration_depth = min(entry.integration_depth + 0.02, 1.0)
                return entry.output_token

    weighted = []
    for e in entries:
        w = max(e.integration_depth, 0.01)
        w *= (1.0 + 0.1 * e.encounter_count)
        if e.labeling_source == "adopted":
            w *= 3.0
        weighted.append((w, e))
    weighted.sort(key=lambda x: -x[0])
    top_n = weighted[:vocab_size]
    weights = [w for w, _ in top_n]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for w, entry in top_n:
        cumulative += w
        if r <= cumulative:
            return entry.output_token
    return top_n[-1][1].output_token if top_n else None


def _next_state(current: OutputState, p_stable: float, p_lexicon: float,
                p_explore: float,
                rng: random.Random = random._inst) -> OutputState:
    r = rng.random()
    if current == "lexicon":
        if r < 0.70:
            return "core"
        r -= 0.70
        if r < 0.15:
            return "lexicon"
        r -= 0.15
        if r < 0.10:
            return "explore"
        return "idle"
    else:
        if r < p_stable:
            return "core"
        r -= p_stable
        if r < p_lexicon:
            return "lexicon"
        r -= p_lexicon
        if r < p_explore:
            return "explore"
        return "idle"


def decide(
    experience: ExperienceSpace,
    max_tokens: int = 8,
    p_stable: float = 0.80,
    p_lexicon: float = 0.10,
    p_explore: float = 0.05,
    ergodic_state: OutputState | None = None,
) -> tuple[list[str], OutputState]:
    if not experience.self_token.is_locked():
        return (["wait"], "idle")

    state = ergodic_state if ergodic_state is not None else "core"
    next_state_val = _next_state(state, p_stable, p_lexicon, p_explore)

    if next_state_val == "core":
        tokens = [experience.self_token.token, "notice"]
        snapshot = experience.meta_trie.get_meta_state_snapshot(
            experience.meta_trie.last_meta_state
        ) if experience.meta_trie.last_meta_state is not None else None
        tokens.append(experience.self_token.token)
        if snapshot is not None and snapshot.mean_prediction_error > 0.3:
            tokens.append("different")
        else:
            tokens.append("familiar")

    elif next_state_val == "lexicon":
        label = _sample_lexicon_label(experience)
        if label is not None:
            tokens = [experience.self_token.token, "notice", label]
        else:
            tokens = [experience.self_token.token, "notice", "familiar"]

    elif next_state_val == "explore":
        token = random.choice(CORE_TOKENS)
        tokens = [token]

    else:
        tokens = ["wait"]

    return (tokens[:max_tokens], next_state_val)
