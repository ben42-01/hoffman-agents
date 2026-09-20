"""Combination operator (x).

v3 semantics (see hoffman-agents-node/src/combination/operator.js):
  - Commutative and associative on identity: the combined id hashes the sorted
    set of leaf constituents, so A(x)B = B(x)A and (A(x)B)(x)C = A(x)(B(x)C).
  - The combined agent's own meta-chain starts empty. Each constituent's chain
    is carried under collision-free provenance ids so fuse() restores it
    exactly at any nesting depth.
  - Hoffman-Prakash product semantics: before joint experience accrues, the
    meta dynamics is the independent product kernel M1 (x) M2.
  - "I" starts unlocked and re-locks on the joint chain.
  - Decision parameters are a convex mixture (default equal weights).
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from ..core import ExperienceTrie, MetaTrie, SelfTokenState, ExperienceLexicon, TraceBuffer
from ..core.meta_trie import ID_MASK
from ..agent import ConsciousAgent, ExperienceSpace
from ..kernels.markov_kernel import MarkovKernel
from ..legacy import combination as legacy
from ..math import markov
from ..math.rng import fnv1a32


def combine(*agents, weights: tuple[float, float] | None = None) -> ConsciousAgent:
    """Combine agents. ``combine(a, b, weights=(w1, w2))`` weights the decision kernels."""
    if not agents:
        return trivial_agent()
    if len(agents) == 1:
        return agents[0]
    if len(agents) == 2:
        if weights is not None:
            return _weighted_combine(agents[0], agents[1], *weights)
        return _combine_pair(agents[0], agents[1])
    mid = len(agents) // 2
    return combine(combine(*agents[:mid]), combine(*agents[mid:]))


def _is_legacy(*agents: ConsciousAgent) -> bool:
    return any(a.math_version == "legacy" for a in agents)


def _combine_pair(a: ConsciousAgent, b: ConsciousAgent, weights=(0.5, 0.5)) -> ConsciousAgent:
    if _is_legacy(a, b):
        return legacy.binary_combine(a, b)
    if a.agent_id == "CA_0":
        return b
    if b.agent_id == "CA_0":
        return a

    # Canonical order makes every downstream choice symmetric.
    agent1, agent2 = a, b
    w1, w2 = weights
    if b.agent_id < a.agent_id:
        agent1, agent2, w1, w2 = b, a, w2, w1

    leaf_ids = frozenset(_leaf_ids(agent1) | _leaf_ids(agent2))
    st1, st2 = agent1.experience.self_token, agent2.experience.self_token
    self_token = SelfTokenState(**{
        **SelfTokenState.lock_options(st1),
        "token": st1.token,
        "lock_consecutive_required": max(st1.lock_consecutive_required, st2.lock_consecutive_required),
        "min_transitions": max(st1.min_transitions, st2.min_transitions),
        "math_version": "v3",
    })

    trace_buffer = _interleave_traces(agent1.experience.trace_buffer, agent2.experience.trace_buffer)
    recent = trace_buffer.get_recent(1)
    exp = ExperienceSpace(
        trie=agent1.experience.trie.merge(agent2.experience.trie),
        meta_trie=_build_joint_meta_trie([agent1, agent2]),
        self_token=self_token,
        lexicon=_merge_lexicons(agent1.experience.lexicon, agent2.experience.lexicon),
        trace_buffer=trace_buffer,
        last_world_state_id=recent[0].to_state if recent else None,
        math_version="v3",
    )

    total = w1 + w2
    n1, n2 = (w1 / total, w2 / total) if total > 0 else (0.5, 0.5)

    combined = ConsciousAgent(
        agent_id=combined_agent_id(leaf_ids),
        experience=exp,
        generation=max(agent1.generation, agent2.generation),
        meta_observation_interval=agent1.meta_observation_interval,
        constituent_ids=(agent1.agent_id, agent2.agent_id),
        leaf_constituent_ids=leaf_ids,
        cycle_level=max(agent1.cycle_level, agent2.cycle_level) + 1,
        p_stable=n1 * agent1.p_stable + n2 * agent2.p_stable,
        p_lexicon=n1 * agent1.p_lexicon + n2 * agent2.p_lexicon,
        p_explore=n1 * agent1.p_explore + n2 * agent2.p_explore,
        lexicon_row=tuple(n1 * x + n2 * y for x, y in zip(agent1.lexicon_row, agent2.lexicon_row)),
        _rng=agent1._rng,
        math_version="v3",
    )
    combined._combined = True
    prior = product_kernel(agent1, agent2)
    combined.combination_prior = (
        {"states": len(prior.states), "diagnostics": _prior_summary(prior)} if prior is not None else None
    )
    return combined


def _weighted_combine(agent1: ConsciousAgent, agent2: ConsciousAgent, w1: float, w2: float) -> ConsciousAgent:
    if not (w1 >= 0 and w2 >= 0):
        raise ValueError(f"combination weights must be non-negative, got ({w1}, {w2})")
    if _is_legacy(agent1, agent2):
        base = legacy.binary_combine(agent1, agent2)
        total = w1 + w2
        if total > 0:
            base.p_stable = (agent1.p_stable * w1 + agent2.p_stable * w2) / total
            base.p_lexicon = (agent1.p_lexicon * w1 + agent2.p_lexicon * w2) / total
            base.p_explore = (agent1.p_explore * w1 + agent2.p_explore * w2) / total
        return base
    if agent2.agent_id == "CA_0":
        return agent1
    if agent1.agent_id == "CA_0":
        return agent2
    return _combine_pair(agent1, agent2, (w1, w2))


def trivial_agent() -> ConsciousAgent:
    return ConsciousAgent(agent_id="CA_0")


def _leaf_ids(agent: ConsciousAgent) -> frozenset[str]:
    return frozenset(agent.leaf_constituent_ids) if agent.leaf_constituent_ids else frozenset([agent.agent_id])


def combined_agent_id(leaf_ids) -> str:
    key = json.dumps(sorted(leaf_ids), separators=(",", ":"), ensure_ascii=False)
    return f"CA_{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def _provenance_id(taken: set[int], constituent_id: str, local_id: int) -> int:
    mid = fnv1a32(f"{constituent_id}:{local_id}") & ID_MASK
    salt = 1
    while mid in taken:
        mid = fnv1a32(f"{constituent_id}:{local_id}#{salt}") & ID_MASK
        salt += 1
    return mid


def _provenance_snapshot(agent: ConsciousAgent) -> dict:
    mt = agent.experience.meta_trie
    return {
        "agent_id": agent.agent_id,
        "provenance": [[mid, dict(p)] for mid, p in mt._provenance.items()],
        "tree": mt._provenance_tree,
        "history": list(mt._history),
        "last_meta_state": mt.last_meta_state,
        "self_token": agent.experience.self_token.to_dict(),
        "constituent_ids": sorted(agent.constituent_ids),
        "leaf_constituent_ids": sorted(agent.leaf_constituent_ids),
        "cycle_level": agent.cycle_level,
        "params": {"p_stable": agent.p_stable, "p_lexicon": agent.p_lexicon, "p_explore": agent.p_explore,
                   "lexicon_row": list(agent.lexicon_row)},
        "combination_prior": agent.combination_prior,
    }


def _copy_node_stats(dst, src) -> None:
    dst.visit_count = src.visit_count
    dst.prediction_errors = list(src.prediction_errors)
    dst.mean_prediction_error = src.mean_prediction_error


def _build_joint_meta_trie(agents: list[ConsciousAgent]) -> MetaTrie:
    joint = MetaTrie(
        snapshot_window=max(a.experience.meta_trie._snapshot_window for a in agents),
        max_depth=max(a.experience.meta_trie.trie.max_depth for a in agents),
        math_version="v3",
    )
    taken: set[int] = set()
    for agent in agents:
        mt = agent.experience.meta_trie
        cid = agent.agent_id
        local_ids = set(mt._registry) | set(mt._token_registry)
        for frm, node in mt.trie.root.children.items():
            local_ids.add(frm)
            local_ids.update(node.children)
        mapping = {}
        for local in sorted(local_ids):
            mid = _provenance_id(taken, cid, local)
            taken.add(mid)
            mapping[local] = mid
            joint._provenance[mid] = {"constituent_id": cid, "local_id": local}

        for local, snap in mt._registry.items():
            joint._registry[mapping[local]] = snap
        for local, counts in mt._token_registry.items():
            joint._token_registry[mapping[local]] = dict(counts)
        for frm, node in mt.trie.root.children.items():
            for to, child in node.children.items():
                path = [mapping[frm], mapping[to]]
                joint._trie.insert(path)
                _copy_node_stats(joint._trie.lookup(path), child)
        joint._provenance_tree[cid] = _provenance_snapshot(agent)
    return joint


def _interleave_traces(buf1: TraceBuffer, buf2: TraceBuffer) -> TraceBuffer:
    tagged = [(e.timestamp, 0, i, e) for i, e in enumerate(buf1)] + [(e.timestamp, 1, i, e) for i, e in enumerate(buf2)]
    tagged.sort(key=lambda t: t[:3])
    out = TraceBuffer(maxlen=max(buf1.maxlen, buf2.maxlen))
    for *_, e in tagged:
        out.append(e)
    return out


def _merge_lexicons(lex1: ExperienceLexicon, lex2: ExperienceLexicon) -> ExperienceLexicon:
    """Same label in both: keep the more integrated entry, pool encounters."""
    merged = ExperienceLexicon(
        embedding_dim=max(lex1.embedding_dim, lex2.embedding_dim),
        association_threshold=min(lex1._association_threshold, lex2._association_threshold),
    )
    by_label: dict[str, tuple] = {}
    for entry in list(lex1._entries.values()) + list(lex2._entries.values()):
        prev = by_label.get(entry.label)
        if prev is None:
            by_label[entry.label] = (entry, entry.encounter_count)
            continue
        best = entry if entry.integration_depth > prev[0].integration_depth else prev[0]
        by_label[entry.label] = (best, prev[1] + entry.encounter_count)
    for label, (entry, encounters) in by_label.items():
        e = merged.bind(
            label=label,
            output_token=entry.output_token,
            trace_signature=entry.trace_signature,
            prediction_error_peak=entry.prediction_error_peak,
            source=entry.labeling_source,
            generation=entry.generation_bound,
            step=entry.step_bound,
        )
        e.integration_depth = entry.integration_depth
        e.encounter_count = encounters
    return merged


def meta_kernel(agent: ConsciousAgent) -> MarkovKernel | None:
    """The agent's empirical meta-state kernel on its recurrent class."""
    mt = agent.experience.meta_trie
    d = mt.ergodic_diagnostics()
    if d["class_size"] == 0:
        return None
    ids, counts = mt.transition_counts()
    pos = {sid: i for i, sid in enumerate(ids)}
    idx = [pos[s] for s in d["states"]]
    return MarkovKernel(d["states"], markov.normalize_rows(markov.sub_matrix(counts, idx))[0])


def product_kernel(a: ConsciousAgent, b: ConsciousAgent) -> MarkovKernel | None:
    """(M1 (x) M2)((x1,x2),(y1,y2)) = M1(x1,y1) M2(x2,y2); pi = pi1 (x) pi2."""
    k1, k2 = meta_kernel(a), meta_kernel(b)
    if k1 is None or k2 is None or len(k1.states) * len(k2.states) > 4096:
        return None
    return k1.tensor(k2)


def _prior_summary(kernel: MarkovKernel) -> dict:
    d = kernel.diagnostics()
    return {"ergodic": d["ergodic"], "period": d["period"], "lambda2": d["lambda2"],
            "entropy": d["entropy"], "dominant_prob": max(d["stationary"].values())}


def fuse(agent: ConsciousAgent) -> list[ConsciousAgent]:
    """Inverse of combine: split into the direct constituents.

    Restores each constituent's meta-chain, provenance, lock state and
    parameters. The world trie and lexicon (which keep learning after
    combination) are copied into each part; the joint meta-chain is not
    attributable to either constituent and is dropped.
    """
    if not agent.constituent_ids:
        return [agent]
    if agent.math_version == "legacy":
        return legacy.fuse(agent)

    joint = agent.experience.meta_trie
    parts = []
    for cid in sorted(agent.constituent_ids):
        snap = joint._provenance_tree.get(cid)
        if snap is None:
            raise ValueError(f"fuse: no provenance recorded for constituent {cid}")

        mt = MetaTrie(snapshot_window=joint._snapshot_window, max_depth=joint.trie.max_depth, math_version="v3")
        to_local = {mid: p["local_id"] for mid, p in joint._provenance.items() if p["constituent_id"] == cid}
        for mid, s in joint._registry.items():
            if mid in to_local:
                mt._registry[to_local[mid]] = s
        for mid, counts in joint._token_registry.items():
            if mid in to_local:
                mt._token_registry[to_local[mid]] = dict(counts)
        for frm, node in joint.trie.root.children.items():
            if frm not in to_local:
                continue
            for to, child in node.children.items():
                if to not in to_local:
                    continue
                path = [to_local[frm], to_local[to]]
                mt._trie.insert(path)
                _copy_node_stats(mt._trie.lookup(path), child)
        mt._provenance = {int(mid): dict(p) for mid, p in snap["provenance"]}
        mt._provenance_tree = snap["tree"]
        mt._history = list(snap["history"])
        mt._last_meta_state = snap["last_meta_state"]

        src_buf = agent.experience.trace_buffer
        buf = TraceBuffer(maxlen=src_buf.maxlen)
        for e in src_buf:
            buf.append(e)

        exp = ExperienceSpace(
            trie=agent.experience.trie.merge(ExperienceTrie(max_depth=agent.experience.trie.max_depth)),
            meta_trie=mt,
            self_token=SelfTokenState.from_dict(snap["self_token"]),
            lexicon=agent.experience.lexicon.clone(),
            trace_buffer=buf,
            last_world_state_id=agent.experience.last_world_state_id,
            math_version="v3",
        )
        params = snap["params"]
        part = ConsciousAgent(
            agent_id=cid,
            experience=exp,
            generation=agent.generation,
            meta_observation_interval=agent.meta_observation_interval,
            constituent_ids=tuple(snap["constituent_ids"]),
            leaf_constituent_ids=frozenset(snap["leaf_constituent_ids"]),
            cycle_level=snap["cycle_level"],
            p_stable=params["p_stable"],
            p_lexicon=params["p_lexicon"],
            p_explore=params["p_explore"],
            lexicon_row=tuple(params["lexicon_row"]),
            _rng=agent._rng,
            math_version="v3",
        )
        part.combination_prior = copy.deepcopy(snap["combination_prior"])
        part._combined = len(snap["constituent_ids"]) > 0
        parts.append(part)
    return parts


def experience_space_distance(exp1: ExperienceSpace, exp2: ExperienceSpace, mode: str = "jaccard",
                              weights: dict | None = None) -> float:
    """Distance between experience spaces.

    'jaccard' (2.x default): 0.6 path-set Jaccard + 0.4 lexicon-label Jaccard.
    'kernel': weights['kernel'] * mean total variation between the empirical
    world-transition kernels on shared source states (1 if none shared)
    + weights['lexicon'] * lexicon-label Jaccard.
    """
    l1 = set(exp1.lexicon._entries.keys())
    l2 = set(exp2.lexicon._entries.keys())
    union_l = l1 | l2
    lexicon_dist = 0.0 if not union_l else 1.0 - len(l1 & l2) / len(union_l)

    if mode == "kernel":
        w = {"kernel": 0.6, "lexicon": 0.4, **(weights or {})}
        return w["kernel"] * kernel_distance(exp1.trie, exp2.trie) + w["lexicon"] * lexicon_dist
    if mode != "jaccard":
        raise ValueError(f"experience_space_distance: unknown mode '{mode}'")

    w = {"trie": 0.6, "lexicon": 0.4, **(weights or {})}
    p1 = set(tuple(p) for p in exp1.trie.get_all_paths(min_visits=0))
    p2 = set(tuple(p) for p in exp2.trie.get_all_paths(min_visits=0))
    union = p1 | p2
    if not union:
        return 0.0
    trie_dist = 1.0 - len(p1 & p2) / len(union)
    return trie_dist * w["trie"] + lexicon_dist * w["lexicon"]


def kernel_distance(trie1: ExperienceTrie, trie2: ExperienceTrie) -> float:
    def row_of(trie: ExperienceTrie, frm: int) -> dict | None:
        node = trie.root.children.get(frm)
        if node is None:
            return None
        row = {to: c.visit_count for to, c in node.children.items() if c.visit_count > 0}
        total = sum(row.values())
        return {k: v / total for k, v in row.items()} if total > 0 else None

    total_tv, shared = 0.0, 0
    for frm in trie1.root.children:
        r1, r2 = row_of(trie1, frm), row_of(trie2, frm)
        if r1 is None or r2 is None:
            continue
        keys = set(r1) | set(r2)
        total_tv += sum(abs(r1.get(k, 0.0) - r2.get(k, 0.0)) for k in keys) / 2
        shared += 1
    return 1.0 if shared == 0 else total_tv / shared


_split_meta_trie = legacy._split_meta_trie
