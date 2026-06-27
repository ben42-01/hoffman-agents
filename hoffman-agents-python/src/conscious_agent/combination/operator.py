from __future__ import annotations

import hashlib

from ..core import (
    ExperienceTrie,
    MetaTrie,
    SelfTokenState,
    ExperienceLexicon,
    TraceBuffer,
    TraceEvent,
)
from ..agent import ConsciousAgent, ExperienceSpace, WorldState


def combine(*agents: ConsciousAgent) -> ConsciousAgent:
    if not agents:
        return trivial_agent()
    if len(agents) == 1:
        return agents[0]
    if len(agents) == 2:
        return _binary_combine(agents[0], agents[1])
    mid = len(agents) // 2
    return combine(combine(*agents[:mid]), combine(*agents[mid:]))


def _binary_combine(agent1: ConsciousAgent, agent2: ConsciousAgent) -> ConsciousAgent:
    if agent1.agent_id == "CA_0":
        return agent2
    if agent2.agent_id == "CA_0":
        return agent1

    combined_id = _hash_agent_ids(agent1.agent_id, agent2.agent_id)

    merged_trie = agent1.experience.trie.merge(agent2.experience.trie)

    leaf_ids_1 = agent1.leaf_constituent_ids or frozenset([agent1.agent_id])
    leaf_ids_2 = agent2.leaf_constituent_ids or frozenset([agent2.agent_id])
    leaf_constituent_ids = leaf_ids_1 | leaf_ids_2
    constituent_ids = frozenset({agent1.agent_id, agent2.agent_id})

    joint_mt = _build_joint_meta_trie(
        agent1.experience.meta_trie,
        agent2.experience.meta_trie,
    )

    combined_self = _combine_attractors(
        agent1.experience.self_token,
        agent2.experience.self_token,
    )

    merged_lexicon = _merge_lexicons(
        agent1.experience.lexicon,
        agent2.experience.lexicon,
    )

    init_state: int | None = None
    trace_buf = TraceBuffer(maxlen=50)
    if agent1.experience.last_world_state_id is not None or agent2.experience.last_world_state_id is not None:
        init_state = _canonical_hash(
            agent1.experience.last_world_state_id or -1,
            agent2.experience.last_world_state_id or -1,
        )
        trace_buf.append(TraceEvent(
            from_state=-1, to_state=init_state, timestamp=0,
            prediction=-1, prediction_correct=False, prediction_error=0.5,
        ))

    exp = ExperienceSpace(
        trie=merged_trie,
        meta_trie=joint_mt,
        self_token=combined_self,
        lexicon=merged_lexicon,
        trace_buffer=trace_buf,
        last_world_state_id=init_state,
    )

    return ConsciousAgent(
        agent_id=combined_id,
        experience=exp,
        generation=max(agent1.generation, agent2.generation),
        constituent_ids=constituent_ids,
        leaf_constituent_ids=leaf_constituent_ids,
        cycle_level=max(agent1.cycle_level, agent2.cycle_level) + 1,
    )


def trivial_agent() -> ConsciousAgent:
    return ConsciousAgent(agent_id="CA_0")


def _hash_agent_ids(id1: str, id2: str) -> str:
    data = f"{id1}->{id2}"
    h = hashlib.sha256(data.encode()).hexdigest()[:12]
    return f"CA_{h}"


def _canonical_hash(s1: int, s2: int) -> int:
    sorted_ids = sorted([s1, s2])
    data = f"ws:{sorted_ids[0]}:{sorted_ids[1]}"
    hash_bytes = hashlib.sha256(data.encode()).digest()
    return int.from_bytes(hash_bytes[:8], "big")


def _build_joint_meta_trie(mt1: MetaTrie, mt2: MetaTrie) -> MetaTrie:
    joint = MetaTrie(
        snapshot_window=max(mt1._snapshot_window, mt2._snapshot_window),
        max_depth=max(mt1.trie.max_depth, mt2.trie.max_depth),
    )

    for mid, snap in mt1._registry.items():
        joint._registry[mid | 0x10000000] = snap
    for mid, snap in mt2._registry.items():
        joint._registry[mid | 0x20000000] = snap

    if hasattr(mt1, "_token_registry"):
        for mid, counts in mt1._token_registry.items():
            joint._token_registry[mid | 0x10000000] = dict(counts)
    if hasattr(mt2, "_token_registry"):
        for mid, counts in mt2._token_registry.items():
            joint._token_registry[mid | 0x20000000] = dict(counts)

    return joint


def _combine_attractors(st1: SelfTokenState, st2: SelfTokenState) -> SelfTokenState:
    combined = SelfTokenState(
        token=st1.token,
        lock_threshold=min(st1.lock_threshold, st2.lock_threshold),
        lock_consecutive_required=max(st1.lock_consecutive_required, st2.lock_consecutive_required),
        stationary_prob=(st1.stationary_prob + st2.stationary_prob) / 2,
        locked=st1.locked and st2.locked,
    )
    if st1.locked and st2.locked:
        combined.referent_meta_state_id = st1.referent_meta_state_id
        combined.lock_generation = max(st1.lock_generation or 0, st2.lock_generation or 0)
    return combined


def _merge_lexicons(lex1: ExperienceLexicon, lex2: ExperienceLexicon) -> ExperienceLexicon:
    merged = ExperienceLexicon(
        embedding_dim=max(lex1.embedding_dim, lex2.embedding_dim),
        association_threshold=min(lex1._association_threshold, lex2._association_threshold),
    )
    for entry in list(lex1._entries.values()) + list(lex2._entries.values()):
        merged.bind(
            label=entry.label,
            output_token=entry.output_token,
            trace_signature=entry.trace_signature,
            prediction_error_peak=entry.prediction_error_peak,
            source=entry.labeling_source,
            generation=entry.generation_bound,
            step=entry.step_bound,
        )
        e = merged.lookup_by_label(entry.label)
        if e is not None:
            e.integration_depth = entry.integration_depth
            e.encounter_count = entry.encounter_count
    return merged


def experience_space_distance(exp1: ExperienceSpace, exp2: ExperienceSpace) -> float:
    p1 = set(tuple(p) for p in exp1.trie.get_all_paths(min_visits=0))
    p2 = set(tuple(p) for p in exp2.trie.get_all_paths(min_visits=0))
    union = p1 | p2
    if not union:
        return 0.0
    trie_dist = 1.0 - len(p1 & p2) / len(union)

    l1 = set(exp1.lexicon._entries.keys())
    l2 = set(exp2.lexicon._entries.keys())
    union_l = l1 | l2
    lexicon_dist = 0.0 if not union_l else 1.0 - len(l1 & l2) / len(union_l)

    return trie_dist * 0.6 + lexicon_dist * 0.4
