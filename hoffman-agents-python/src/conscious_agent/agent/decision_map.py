from __future__ import annotations

import random
from typing import Literal

from ..kernels.markov_kernel import MarkovKernel
from ..legacy import decision as legacy
from ..math.signature import build_transition_signature
from .experience_space import ExperienceSpace

OutputState = Literal["core", "lexicon", "explore", "idle"]

CORE_TOKENS = ["I", "notice", "familiar", "different", "wait"]

SIGNATURE_MATCH_THRESHOLD = 0.3

# Output (decision) states; their dynamics is the Markov kernel D.
DECISION_STATES: tuple[str, ...] = ("core", "lexicon", "explore", "idle")
DEFAULT_LEXICON_ROW: tuple[float, ...] = (0.70, 0.15, 0.10, 0.05)


def build_decision_kernel(
    p_stable: float = 0.80,
    p_lexicon: float = 0.10,
    p_explore: float = 0.05,
    lexicon_row=DEFAULT_LEXICON_ROW,
) -> MarkovKernel:
    """Decision kernel D over DECISION_STATES.

      from core / explore / idle : [p_stable, p_lexicon, p_explore, 1 - sum]
      from lexicon               : lexicon_row
    """
    for name, v in (("p_stable", p_stable), ("p_lexicon", p_lexicon), ("p_explore", p_explore)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    p_idle = 1.0 - p_stable - p_lexicon - p_explore
    if p_idle < -1e-9:
        raise ValueError(f"p_stable + p_lexicon + p_explore must be <= 1, got {p_stable + p_lexicon + p_explore}")
    base = [p_stable, p_lexicon, p_explore, max(0.0, p_idle)]
    return MarkovKernel(list(DECISION_STATES), [base, list(lexicon_row), base, base])


_kernel_cache: dict[tuple, MarkovKernel] = {}


def _cached_kernel(p_stable, p_lexicon, p_explore, lexicon_row) -> MarkovKernel:
    key = (p_stable, p_lexicon, p_explore, tuple(lexicon_row))
    k = _kernel_cache.get(key)
    if k is None:
        k = build_decision_kernel(p_stable, p_lexicon, p_explore, lexicon_row)
        if len(_kernel_cache) > 256:
            _kernel_cache.clear()
        _kernel_cache[key] = k
    return k


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
        query_sig = build_transition_signature(prev_id, curr_id, experience.lexicon.embedding_dim,
                                               experience.math_version)
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
    """Deprecated 2.x helper; v3 samples from build_decision_kernel()."""
    return legacy.next_state(current, p_stable, p_lexicon, p_explore, rng)


def decide(
    experience: ExperienceSpace,
    max_tokens: int = 8,
    p_stable: float = 0.80,
    p_lexicon: float = 0.10,
    p_explore: float = 0.05,
    ergodic_state: OutputState | None = None,
    rng: random.Random = random._inst,
    lexicon_row=DEFAULT_LEXICON_ROW,
    kernel: MarkovKernel | None = None,
) -> tuple[list[str], OutputState]:
    if not experience.self_token.is_locked():
        return (["wait"], "idle")

    state = ergodic_state if ergodic_state is not None else "core"
    if experience.math_version == "legacy":
        next_state_val = legacy.next_state(state, p_stable, p_lexicon, p_explore, rng)
    else:
        k = kernel or _cached_kernel(p_stable, p_lexicon, p_explore, lexicon_row)
        next_state_val = k.sample(state, rng)

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
        label = _sample_lexicon_label(experience, rng=rng)
        if label is not None:
            tokens = [experience.self_token.token, "notice", label]
        else:
            tokens = [experience.self_token.token, "notice", "familiar"]

    elif next_state_val == "explore":
        token = rng.choice(CORE_TOKENS)
        tokens = [token]

    else:
        tokens = ["wait"]

    return (tokens[:max_tokens], next_state_val)
