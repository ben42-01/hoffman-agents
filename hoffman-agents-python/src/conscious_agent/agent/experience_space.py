from __future__ import annotations

from dataclasses import dataclass, field

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

    @property
    def is_identity_stable(self) -> bool:
        return self.self_token.is_stable()

    is_i_locked = is_identity_stable


MemorySpace = ExperienceSpace
