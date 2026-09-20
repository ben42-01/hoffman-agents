from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from collections.abc import Sequence


def sequence_to_state_id(sequence: Sequence[str], bits: int = 64) -> int:
    """SHA-256 of the joined sequence, first ``bits`` (32 or 64) bits, big-endian.

    32-bit ids are identical to the Node implementation and are used by
    math_version='v3'; 64-bit ids are the 2.x Python default.
    """
    data = "|".join(sequence)
    hash_bytes = hashlib.sha256(data.encode()).digest()
    return int.from_bytes(hash_bytes[:bits // 8], "big")


DIMENSION_BINS = {
    "temperature": 5,
    "intensity": 5,
    "valence": 5,
    "rhythm": 4,
}


@dataclass
class WorldState:
    sequences: dict[str, list[str]] = field(default_factory=dict)
    dimensions: dict[str, float] = field(default_factory=dict)

    def get_state_id(self, bits: int = 64) -> int:
        flat: list[str] = []
        for aid, seq in sorted(self.sequences.items()):
            flat.append(aid)
            flat.extend(seq)
        for k, v in sorted(self.dimensions.items()):
            n_bins = DIMENSION_BINS.get(k, 5)
            normalized = max(0.0, min(1.0, v))
            bin_idx = min(int(normalized * n_bins), n_bins - 1)
            flat.append(f"{k}:{bin_idx}")
        return sequence_to_state_id(flat, bits)

    def dimension_delta(self, other: WorldState | None) -> dict[str, float]:
        if other is None:
            return {}
        delta = {}
        for k in self.dimensions:
            ov = other.dimensions.get(k, 0.0)
            delta[k] = self.dimensions[k] - ov
        return delta

    @staticmethod
    def from_sequence(agent_id: str, sequence: list[str]) -> WorldState:
        return WorldState(sequences={agent_id: sequence})

    def __bool__(self) -> bool:
        return len(self.sequences) > 0


EnvironmentState = WorldState
