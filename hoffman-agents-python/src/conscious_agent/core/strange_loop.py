from __future__ import annotations

from collections.abc import Sequence

SELF_TOKEN = "I"

_PREDICATE_TOKENS = frozenset({
    "remember", "notice", "think", "see", "feel", "know",
    "cross", "arrive", "return", "enter", "leave",
    "expect", "predict", "observe", "experience", "am",
})


def compute_self_reference_score(sequence: Sequence[str]) -> float:
    n_i = sum(1 for t in sequence if t == SELF_TOKEN)

    if n_i == 0:
        return 0.0
    if n_i == 1:
        return 0.5

    positions = [i for i, t in enumerate(sequence) if t == SELF_TOKEN]
    valid_pairs = 0

    for idx, i in enumerate(positions):
        for j in positions[idx + 1:]:
            between = sequence[i + 1:j]
            has_predicate = any(t in _PREDICATE_TOKENS for t in between)
            if has_predicate:
                valid_pairs += 1

    if valid_pairs == 0:
        return 0.5

    max_depth = len(positions)
    score = 0.5 * max_depth * (valid_pairs / (n_i * (n_i - 1) / 2))

    return min(score, 2.0)


def population_reference_score(sequences: Sequence[Sequence[str]]) -> float:
    if not sequences:
        return 0.0
    return sum(compute_self_reference_score(s) for s in sequences) / len(sequences)


strange_loop_score = compute_self_reference_score
population_loop_score = population_reference_score


def first_depth_n_generation(
    loop_history: Sequence[float],
    threshold: float,
) -> int | None:
    for gen, score in enumerate(loop_history):
        if score >= threshold:
            return gen
    return None
