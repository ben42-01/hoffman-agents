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
from .decision_map import decide, OutputState

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
class ConsciousAgent:
    agent_id: str
    experience: ExperienceSpace = field(default_factory=ExperienceSpace)
    world: Any = None
    generation: int = 0
    step_count: int = 0
    meta_observation_interval: int = 20
    constituent_ids: frozenset[str] = field(default_factory=frozenset)
    leaf_constituent_ids: frozenset[str] = field(default_factory=frozenset)
    cycle_level: int = 0
    expression_temp: float = 1.0
    p_stable: float = 0.80
    p_lexicon: float = 0.10
    p_explore: float = 0.05
    _mode: str = "learning"
    _rng: Any = field(default_factory=_random.Random)
    _allowable_tokens: set[str] | None = None
    _combined: bool = False
    _ergodic_state: OutputState = "idle"
    _last_output: list[str] = field(default_factory=lambda: ["wait"])

    @staticmethod
    def from_config(agent_id: str, config: dict) -> ConsciousAgent:
        agent_cfg = config.get("agent", {})
        st_cfg = agent_cfg.get("self_token", {})
        st = SelfTokenState(
            lock_threshold=st_cfg.get("lock_threshold", 0.25),
            lock_consecutive_required=st_cfg.get("lock_consecutive_required", 3),
        )
        exp = ExperienceSpace(self_token=st)
        return ConsciousAgent(
            agent_id=agent_id,
            experience=exp,
            meta_observation_interval=agent_cfg.get("meta_observation_interval", 20),
            expression_temp=agent_cfg.get("expression_temp", 1.0),
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
            state=self.experience.last_world_state_id or -1,
            state_label=str(self.experience.last_world_state_id or "?"),
            prediction_error=self.experience.trace_buffer.prediction_error_mean(window=5),
            sequence=list(output),
            sequence_str=" ".join(output),
            loop_depth=float(_compute_self_reference_score(output)),
            i_locked=self.experience.self_token.locked,
            i_stability=self.experience.self_token.stationary_variance(),
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
"""
