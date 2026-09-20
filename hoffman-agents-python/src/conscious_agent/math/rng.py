"""Deterministic, cross-language (Python <-> Node) primitives.

Bit-exact ports of hoffman-agents-node/src/math/rng.js.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")

_M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """32-bit integer multiply, as JavaScript's Math.imul (returned unsigned)."""
    return (a * b) & _M32


class Mulberry32:
    """Seeded PRNG with the subset of the random.Random API the library uses.

    ``choice`` and ``randrange`` consume exactly one draw, as the Node code
    does with ``Math.floor(rng() * n)``, so both languages stay in lock-step.
    """

    def __init__(self, seed: int) -> None:
        self._a = seed & _M32

    def random(self) -> float:
        self._a = (self._a + 0x6D2B79F5) & _M32
        t = self._a
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & _M32)) & _M32
        return ((t ^ (t >> 14)) & _M32) / 4294967296

    __call__ = random

    def choice(self, seq: Sequence[T]) -> T:
        return seq[int(self.random() * len(seq))]

    def randrange(self, n: int) -> int:
        return int(self.random() * n)


def mulberry32(seed: int) -> Mulberry32:
    return Mulberry32(seed)


def fnv1a32(value: object) -> int:
    """FNV-1a 32-bit over the UTF-8 bytes of ``str(value)``."""
    h = 0x811C9DC5
    for b in str(value).encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & _M32
    return h
