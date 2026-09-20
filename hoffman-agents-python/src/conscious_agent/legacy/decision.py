"""2.x output-state chain, preserved verbatim for math_version='legacy'."""
from __future__ import annotations


def next_state(current, p_stable, p_lexicon, p_explore, rng):
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
