from __future__ import annotations

from dataclasses import dataclass, field

MATH_VERSIONS = ("v3", "legacy")

from ..core import (
    ExperienceTrie,
    MetaTrie,
    SelfTokenState,
    ExperienceLexicon,
    TraceBuffer,
)


@dataclass
class ExperienceSpace:
    trie: ExperienceTrie = field(default_factory=lambda: ExperienceTrie(max_depth=10))
    meta_trie: MetaTrie = field(default_factory=lambda: MetaTrie(snapshot_window=10, max_depth=10))
    self_token: SelfTokenState = field(default_factory=SelfTokenState)
    lexicon: ExperienceLexicon = field(default_factory=lambda: ExperienceLexicon(embedding_dim=64))
    trace_buffer: TraceBuffer = field(default_factory=lambda: TraceBuffer(maxlen=50))
    last_world_state_id: int | None = None
    # Assigned through the property defined below the class: the meta-trie and
    # self-token hold the actual value, None leaves them unchanged.
    math_version: str | None = None

    def set_math_version(self, math_version: str) -> ExperienceSpace:
        """'v3' (default) or 'legacy' (reproduces 2.x dynamics)."""
        if math_version not in MATH_VERSIONS:
            raise ValueError(f"Invalid math_version '{math_version}'. Use: {', '.join(MATH_VERSIONS)}")
        self.meta_trie.math_version = math_version
        self.self_token.math_version = math_version
        return self

    @property
    def is_identity_stable(self) -> bool:
        return self.self_token.is_stable()

    is_i_locked = is_identity_stable


def _get_math_version(self: ExperienceSpace) -> str:
    return self.meta_trie.math_version


def _set_math_version(self: ExperienceSpace, value: str | None) -> None:
    if value is not None:
        self.set_math_version(value)


ExperienceSpace.math_version = property(_get_math_version, _set_math_version)

MemorySpace = ExperienceSpace
