from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Self


@dataclass
class TraceEvent:
    from_state: int
    to_state: int
    timestamp: int
    prediction: int
    prediction_correct: bool
    prediction_error: float
    token: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.prediction_error <= 1.0:
            raise ValueError(f"prediction_error must be in [0, 1], got {self.prediction_error}")


class TraceBuffer:
    def __init__(self, maxlen: int = 50) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = maxlen
        self._events: list[TraceEvent] = []
        self._cursor = 0

    def append(self, event: TraceEvent) -> None:
        if len(self._events) < self._maxlen:
            self._events.append(event)
        else:
            self._events[self._cursor] = event
            self._cursor = (self._cursor + 1) % self._maxlen

    def get_recent(self, n: int) -> list[TraceEvent]:
        if n <= 0:
            return []
        n = min(n, len(self._events))
        if len(self._events) < self._maxlen:
            return self._events[-n:]
        return (self._events[self._cursor:] + self._events[:self._cursor])[-n:]

    def as_state_sequence(self) -> list[int]:
        return [e.to_state for e in self]

    def prediction_error_mean(self, window: int = 20) -> float:
        recent = self.get_recent(window)
        if not recent:
            return 0.0
        return sum(e.prediction_error for e in recent) / len(recent)

    def is_full(self) -> bool:
        return len(self._events) >= self._maxlen

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[TraceEvent]:
        if len(self._events) < self._maxlen:
            yield from self._events
        else:
            yield from self._events[self._cursor:]
            yield from self._events[:self._cursor]

    def __getitem__(self, index: int) -> TraceEvent:
        if not (0 <= index < len(self._events)):
            raise IndexError("TraceBuffer index out of range")
        if len(self._events) < self._maxlen:
            return self._events[index]
        return self._events[(self._cursor + index) % self._maxlen]

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def resize(self, new_maxlen: int) -> None:
        if new_maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {new_maxlen}")
        ordered = list(self)
        self._maxlen = new_maxlen
        if len(ordered) > new_maxlen:
            self._events = ordered[-new_maxlen:]
        else:
            self._events = ordered
        self._cursor = 0

    def clear(self) -> None:
        self._events.clear()
        self._cursor = 0

    @classmethod
    def from_events(cls, events: list[TraceEvent], maxlen: int | None = None) -> Self:
        if maxlen is None:
            maxlen = len(events)
        buf = cls(maxlen=maxlen)
        for e in events:
            buf.append(e)
        return buf
