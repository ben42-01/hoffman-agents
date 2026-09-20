from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import random as _random

import numpy as np

from ..core import (
    compute_self_reference_score as _compute_self_reference_score,
    ExperienceTrie,
    MetaTrie,
    SelfTokenState,
    TraceBuffer,
)
from .world_state import WorldState
from .experience_space import ExperienceSpace
from .perceptual_map import perceive
from .decision_map import decide, OutputState, build_decision_kernel, DEFAULT_LEXICON_ROW
from ..math import markov
from ..math.rng import Mulberry32
from ..kernels.markov_kernel import MarkovKernel, StochasticMatrix

_VALID_MODES = frozenset({"learning", "frozen", "debug"})


@dataclass
class StepOutput:
    step: int
    generation: int
    state: int
    state_label: str
    prediction_error: float
    sequence: list[str]
    sequence_str: str
    loop_depth: float
    i_locked: bool
    i_stability: float
    interrupt: Any = None
    action_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class Prediction:
    state_id: int
    state_label: str
    confidence: float
    _top_k: list[dict] = field(default_factory=list)

    def top_k(self, n: int) -> list[dict]:
        return self._top_k[:max(0, n)]


@dataclass
class ConsciousAgent:
    agent_id: str
    experience: ExperienceSpace = field(default_factory=ExperienceSpace)
    world: Any = None
    generation: int = 0
    step_count: int = 0
    meta_observation_interval: int = 20
    constituent_ids: tuple[str, ...] = field(default_factory=tuple)
    leaf_constituent_ids: frozenset[str] = field(default_factory=frozenset)
    cycle_level: int = 0
    expression_temp: float = 1.0
    p_stable: float = 0.80
    p_lexicon: float = 0.10
    p_explore: float = 0.05
    _mode: str = "learning"
    _rng: Any = None
    _allowable_tokens: set[str] | None = None
    _combined: bool = False
    _ergodic_state: OutputState = "idle"
    _last_output: list[str] = field(default_factory=lambda: ["wait"])
    seed: int | None = None
    lexicon_row: tuple[float, ...] = DEFAULT_LEXICON_ROW
    combination_prior: dict | None = None
    # 'v3' (default) or 'legacy'. Assigned through the property defined below
    # the class; the experience space holds the value. None keeps its current one.
    math_version: str | None = None

    def __post_init__(self) -> None:
        if self._rng is None:
            self._rng = Mulberry32(self.seed) if self.seed is not None else _random.Random()
        self.lexicon_row = tuple(self.lexicon_row)
        if self.math_version != "legacy":
            self.decision_kernel  # validate p-parameters early

    @property
    def decision_kernel(self) -> MarkovKernel:
        """The decision kernel D over ('core', 'lexicon', 'explore', 'idle')."""
        return build_decision_kernel(self.p_stable, self.p_lexicon, self.p_explore, self.lexicon_row)

    @staticmethod
    def from_config(agent_id: str, config: dict) -> ConsciousAgent:
        agent_cfg = config.get("agent", {})
        st_cfg = agent_cfg.get("self_token", {})
        st = SelfTokenState(**SelfTokenState.lock_options(st_cfg))
        exp = ExperienceSpace(self_token=st)
        return ConsciousAgent(
            agent_id=agent_id,
            experience=exp,
            meta_observation_interval=agent_cfg.get("meta_observation_interval", 20),
            expression_temp=agent_cfg.get("expression_temp", 1.0),
            p_stable=agent_cfg.get("p_stable", 0.80),
            p_lexicon=agent_cfg.get("p_lexicon", 0.10),
            p_explore=agent_cfg.get("p_explore", 0.05),
            lexicon_row=tuple(agent_cfg.get("lexicon_row", DEFAULT_LEXICON_ROW)),
            seed=agent_cfg.get("seed"),
            math_version=agent_cfg.get("math_version", "v3"),
        )

    def step(self, world: WorldState | None = None) -> StepOutput:
        if world is not None:
            pass

        if world is None and self.world is not None:
            if hasattr(self.world, 'step'):
                world = self.world.step()

        is_frozen = self._mode in ("frozen", "debug")

        if world is not None:
            self.experience = perceive(
                world,
                self.experience,
                step=self.step_count,
                meta_observation_interval=self.meta_observation_interval,
                frozen=is_frozen,
                ergodic_state=self._ergodic_state,
                rng=self._rng,
                generation=self.generation,
            )

        p_stable = 1.0 if is_frozen else self.p_stable
        p_lexicon = 0.0 if is_frozen else self.p_lexicon
        p_explore = 0.0 if is_frozen else self.p_explore

        output, self._ergodic_state = decide(
            self.experience,
            p_stable=p_stable,
            p_lexicon=p_lexicon,
            p_explore=p_explore,
            ergodic_state=self._ergodic_state,
            rng=self._rng,
            lexicon_row=self.lexicon_row,
        )

        if self._allowable_tokens is not None:
            filtered = [t for t in output if t in self._allowable_tokens]
            if filtered:
                output = filtered
            else:
                output = [next(iter(self._allowable_tokens))]

        if not is_frozen:
            for token in output:
                if self.experience.meta_trie.last_meta_state is not None:
                    self.experience.meta_trie.record_token(
                        self.experience.meta_trie.last_meta_state, token
                    )

        self._last_output = output
        self.step_count += 1
        if self.step_count > 0 and self.step_count % self.meta_observation_interval == 0:
            self.generation += 1

        action_distribution = self._compute_action_distribution(output)

        return StepOutput(
            action_distribution=action_distribution,
            step=self.step_count,
            generation=self.generation,
            state=self.experience.last_world_state_id if self.experience.last_world_state_id is not None else -1,
            state_label=str(self.experience.last_world_state_id if self.experience.last_world_state_id is not None else "?"),
            prediction_error=self.experience.trace_buffer.prediction_error_mean(window=5),
            sequence=list(output),
            sequence_str=" ".join(output),
            loop_depth=float(_compute_self_reference_score(output)),
            i_locked=self.experience.self_token.locked,
            i_stability=self.experience.self_token.stationary_variance(),
            interrupt=self.experience.self_token.consume_event(),
        )

    def run(self, n_steps: int) -> list[StepOutput]:
        outputs = []
        for _ in range(n_steps):
            outputs.append(self.step())
        return outputs

    def observe(self, output_sequence: list[str], source_id: str) -> StepOutput:
        return self.step(WorldState.from_sequence(source_id, output_sequence))

    def inject_observation(self, world_state: WorldState) -> StepOutput:
        return self.step(world_state)

    def predict_next(self) -> Prediction | None:
        last_state_id = self.experience.last_world_state_id
        if last_state_id is None:
            return None

        node = self.experience.trie.lookup([last_state_id])
        if node is None or not node.children:
            return None

        children = [(sid, child.visit_count) for sid, child in node.children.items()]
        total = sum(vc for _, vc in children)
        if total == 0:
            return None

        children.sort(key=lambda x: x[1], reverse=True)
        best_id, best_count = children[0]
        confidence = best_count / total

        top_k = [
            {"state_id": sid, "state_label": str(sid), "confidence": c / total}
            for sid, c in children
        ]

        return Prediction(
            state_id=best_id,
            state_label=str(best_id),
            confidence=confidence,
            _top_k=top_k,
        )

    def ergodic_stats(self) -> dict:
        """Ergodic analysis of the decision kernel D and the meta-state chain."""
        meta = self.experience.meta_trie.ergodic_diagnostics()
        st = self.experience.self_token
        return {
            "math_version": self.math_version,
            "decision": self.decision_kernel.diagnostics(),
            "meta": meta,
            "lock": {
                "locked": st.locked,
                "referent": st.referent_meta_state_id,
                "criteria": st.lock_criteria(meta)["checks"] if meta["class_size"] > 0 else None,
                "history": list(st.lock_history),
            },
            "combination_prior": self.combination_prior,
        }

    def to_formal(self) -> dict:
        """Learned dynamics as explicit kernels (Hoffman & Prakash's X, G, P, D, A, N).

        X: meta-states of the recurrent class; P: empirical world-transition
        kernel; M: meta-state kernel on X; D: decision kernel; A: meta-state ->
        emitted-token distribution; N: step counter.
        """
        meta = self.experience.meta_trie.ergodic_diagnostics()
        ids, counts = self.experience.meta_trie.transition_counts()
        pos = {sid: i for i, sid in enumerate(ids)}
        M = None
        if meta["class_size"] > 0:
            idx = [pos[s] for s in meta["states"]]
            M = MarkovKernel(meta["states"], markov.normalize_rows(markov.sub_matrix(counts, idx))[0])

        registry = self.experience.meta_trie._token_registry
        rows = sorted(mid for mid, c in registry.items() if c)
        cols = sorted({t for mid in rows for t in registry[mid]})
        A = (StochasticMatrix.from_counts(rows, cols, [[registry[mid].get(t, 0) for t in cols] for mid in rows])
             if rows else None)

        return {"X": meta["states"], "G": cols, "P": world_kernel(self.experience.trie), "M": M,
                "D": self.decision_kernel, "A": A, "N": self.step_count, "diagnostics": meta}

    def get_output(self) -> list[str]:
        return list(self._last_output)

    def set_world(self, world: Any) -> None:
        self.world = world

    def set_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Use: {', '.join(sorted(_VALID_MODES))}")
        self._mode = mode

    def thaw(self) -> None:
        self._mode = "learning"

    def refreeze(self) -> None:
        self._mode = "frozen"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def metrics(self) -> dict:
        return {
            "prediction_error": self.experience.trace_buffer.prediction_error_mean(window=5),
            "i_locked": self.experience.self_token.locked,
            "i_stability": self.experience.self_token.stationary_variance(),
            "loop_depth": float(_compute_self_reference_score(self._last_output)),
            "output_tokens": list(self._last_output),
        }

    @property
    def loop_score(self) -> float:
        return float(_compute_self_reference_score(self._last_output))

    @property
    def mean_prediction_error(self) -> float:
        return self.experience.trace_buffer.prediction_error_mean(window=100)

    @property
    def is_i_locked(self) -> bool:
        return self.experience.self_token.locked

    @property
    def is_identity_stable(self) -> bool:
        return self.experience.is_identity_stable

    is_ripe = is_identity_stable

    def set_allowable_tokens(self, tokens: set[str]) -> None:
        self._allowable_tokens = tokens

    @staticmethod
    def _compute_action_distribution(output: list[str]) -> dict[str, float]:
        if not output:
            return {}
        counts: dict[str, int] = {}
        for token in output:
            counts[token] = counts.get(token, 0) + 1
        total = len(output)
        return {token: count / total for token, count in counts.items()}

    def clear_memory(self) -> None:
        self.experience.trace_buffer.clear()
        self.step_count = 0
        self.generation = 0
        self._last_output = ["wait"]
        self._ergodic_state = "idle"

    def clear(self) -> None:
        self.clear_memory()
        self.experience.trie.clear()
        self.experience.meta_trie.clear()
        self.experience.lexicon.clear()


def world_kernel(trie: ExperienceTrie) -> MarkovKernel | None:
    """Empirical world-state transition kernel from depth-2 trie counts,
    restricted to states whose outgoing transitions have been observed."""
    edges = [(frm, to, child.visit_count)
             for frm, node in trie.root.children.items()
             for to, child in node.children.items() if child.visit_count > 0]
    if not edges:
        return None
    ids = sorted({e[0] for e in edges} | {e[1] for e in edges})
    idx = {sid: i for i, sid in enumerate(ids)}
    counts = np.zeros((len(ids), len(ids)))
    for frm, to, c in edges:
        counts[idx[frm], idx[to]] += c
    kept = markov.prune_unobserved_rows(counts)
    if not kept:
        return None
    return MarkovKernel([ids[i] for i in kept], markov.normalize_rows(markov.sub_matrix(counts, kept))[0])


def _get_math_version(self: ConsciousAgent) -> str:
    return self.experience.math_version


def _set_math_version(self: ConsciousAgent, value: str | None) -> None:
    if value is not None:
        self.experience.set_math_version(value)


ConsciousAgent.math_version = property(_get_math_version, _set_math_version)


ConsciousAgent.__init__.__doc__ = """Create a ConsciousAgent.

Args:
    agent_id: Unique identifier for this agent.
    experience: The agent's internal experience space (trie, meta-trie, etc.)
    world: Optional world object with a .step() method returning WorldState.
    generation: Current generation counter.
    step_count: Current step counter.
    meta_observation_interval: Steps between meta-trie observations (default 20).
    constituent_ids: IDs of agents that combined to create this agent.
    leaf_constituent_ids: IDs of leaf agents below combined agents.
    cycle_level: Depth in combination tree (0 = base agent).
    expression_temp: Temperature controlling output style (1.0=poetic, 0.0=clinical).
    p_stable: Probability of remaining in core output state.
    p_lexicon: Probability of transitioning to lexicon output state.
    p_explore: Probability of exploring random tokens.
    seed: Seed for the deterministic Mulberry32 RNG (identical to the Node library).
    lexicon_row: Decision-kernel row used after a lexicon utterance.
    math_version: 'v3' (default) or 'legacy' to reproduce 2.x dynamics.
"""
