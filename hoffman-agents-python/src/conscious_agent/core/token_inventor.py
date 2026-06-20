from __future__ import annotations

import random

CONSONANTS = "bdfghjklmnprstvwxyz"
VOWELS = "aeiou"


def invent_token(rng: random.Random | None = None) -> str:
    if rng is None:
        rng = random
    c1 = rng.choice(CONSONANTS)
    v = rng.choice(VOWELS)
    c2 = rng.choice(CONSONANTS)
    return f"{c1}{v}{c2}"


def is_invented_token(token: str) -> bool:
    if token in ("I", "notice", "familiar", "different", "wait"):
        return False
    if token.startswith("p:"):
        return False
    if len(token) != 3:
        return False
    return (token[0] in CONSONANTS and token[1] in VOWELS and token[2] in CONSONANTS)
