from __future__ import annotations

from typing import Any, Callable

from ..agent import ConsciousAgent, WorldState


def _default_get_state(agent: ConsciousAgent) -> dict[str, float]:
    return {
        "sp": agent.experience.self_token.stationary_prob,
        "pe": agent.mean_prediction_error,
        "lock": 1.0 if agent.is_i_locked else 0.0,
        "loop": agent.loop_score,
        "meta_states": float(agent.experience.meta_trie.registry_size),
        "vocab_size": float(agent.experience.lexicon.vocabulary_size()),
        "trie_nodes": float(agent.experience.trie.size()),
    }


class SelfWorld:
    def __init__(
        self,
        world: Any,
        agent: ConsciousAgent,
        get_state_fn: Callable[[ConsciousAgent], dict[str, float]] | None = None,
    ) -> None:
        if not hasattr(world, "step") or not callable(world.step):
            raise ValueError("SelfWorld requires a world with a step() method")
        self._inner = world
        self._agent = agent
        self._get_state = get_state_fn or _default_get_state
        self._step_count = 0
        self.n_states = getattr(world, "n_states", 1_000_000)

    def step(self) -> WorldState:
        world_state = self._inner.step()
        self_metrics = self._get_state(self._agent)

        tokens = []
        for key, val in self_metrics.items():
            if isinstance(val, float):
                tokens.append(f"{key}:{val:.2f}")
            else:
                tokens.append(f"{key}:{val}")

        world_state.sequences["self"] = tokens
        if hasattr(world_state, "dimensions"):
            world_state.dimensions = {**world_state.dimensions, **self_metrics}
        else:
            world_state.dimensions = dict(self_metrics)

        return world_state

    @property
    def initial_state(self) -> WorldState:
        return self.step()
