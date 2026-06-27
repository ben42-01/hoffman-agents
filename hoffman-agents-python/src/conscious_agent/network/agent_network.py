from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..agent import ConsciousAgent, StepOutput, WorldState


@dataclass
class Connection:
    strength: float = 0.5
    age: int = 0


class Topology:
    def __init__(
        self,
        n_agents: int,
        initial_connections_per_agent: int = 5,
        seed: int = 42,
        learning_rate: float = 0.01,
        prune_threshold: float = 0.05,
        add_threshold: float = 0.5,
    ) -> None:
        self._n_agents = n_agents
        self._learning_rate = learning_rate
        self._prune_threshold = prune_threshold
        self._add_threshold = add_threshold
        self._rng = random.Random(seed)
        self._connections: dict[tuple[int, int], Connection] = {}
        self._agent_ids: list[str] = []
        self._build_initial_topology(initial_connections_per_agent)

    def _build_initial_topology(self, initial_per_agent: int) -> None:
        for i in range(self._n_agents):
            candidates = [j for j in range(self._n_agents) if j != i]
            if len(candidates) <= initial_per_agent:
                targets = candidates
            else:
                targets = self._rng.sample(candidates, initial_per_agent)
            for j in targets:
                key = (i, j)
                if key not in self._connections:
                    self._connections[key] = Connection(strength=self._rng.uniform(0.3, 0.7))

    def set_agent_ids(self, ids: list[str]) -> None:
        self._agent_ids = list(ids)

    def get_connections(self, agent_idx: int) -> list[int]:
        return [j for (i, j), conn in self._connections.items()
                if i == agent_idx and conn.strength > 0]

    def get_connection_strength(self, i: int, j: int) -> float:
        conn = self._connections.get((i, j))
        return conn.strength if conn is not None else 0.0

    def update_connection(self, i: int, j: int, delta: float) -> None:
        key = (i, j)
        if key not in self._connections:
            self._connections[key] = Connection(strength=0.0)
        conn = self._connections[key]
        conn.strength += self._learning_rate * delta
        conn.strength = max(0.0, min(1.0, conn.strength))
        conn.age += 1

    def prune_weak_connections(self) -> int:
        to_remove = [key for key, conn in self._connections.items()
                     if conn.strength < self._prune_threshold]
        for key in to_remove:
            del self._connections[key]
        return len(to_remove)

    def maybe_add_connection(self, i: int, j: int) -> bool:
        key = (i, j)
        if key in self._connections:
            return False
        if self._rng.random() < self._add_threshold:
            self._connections[key] = Connection(strength=0.1)
            return True
        return False

    def get_agent_observers(self, agent_idx: int) -> list[int]:
        return [i for (i, j), conn in self._connections.items()
                if j == agent_idx and conn.strength > 0]

    @property
    def connection_count(self) -> int:
        return len(self._connections)


class InteractionCycle:
    def __init__(
        self,
        topology: Topology,
        meta_observation_interval: int = 20,
        topology_update_interval: int = 10,
        prediction_window: int = 10,
    ) -> None:
        self._topology = topology
        self._meta_observation_interval = meta_observation_interval
        self._topology_update_interval = topology_update_interval
        self._prediction_window = prediction_window

    def step(
        self,
        agents: dict[str, ConsciousAgent],
        agent_ids: list[str],
        generation: int = 0,
    ) -> dict[str, list[str]]:
        outputs: dict[str, list[str]] = {}

        for idx, agent_id in enumerate(agent_ids):
            agent = agents.get(agent_id)
            if agent is None:
                continue

            if agent.constituent_ids:
                world = self._build_combined_world(agent, agents)
            else:
                world = self._build_world_from_topology(idx, agent, agents, agent_ids)
            out = agent.step(world)
            outputs[agent_id] = out.sequence

        self._update_topology(agents, agent_ids, generation)
        return outputs

    @staticmethod
    def _build_combined_world(agent: ConsciousAgent, all_agents: dict[str, ConsciousAgent]) -> WorldState:
        sequences: dict[str, list[str]] = {}
        for cid in agent.constituent_ids:
            other = all_agents.get(cid)
            if other is not None and cid != agent.agent_id:
                sequences[cid] = other.get_output()
        return WorldState(sequences=sequences)

    def _build_world_from_topology(
        self, idx: int, agent: ConsciousAgent,
        all_agents: dict[str, ConsciousAgent], agent_ids: list[str],
    ) -> WorldState:
        connected = self._topology.get_connections(idx)
        sequences: dict[str, list[str]] = {}
        for other_idx in connected:
            if other_idx >= len(agent_ids):
                continue
            other_id = agent_ids[other_idx]
            if other_id == agent.agent_id:
                continue
            other = all_agents.get(other_id)
            if other is not None:
                sequences[other_id] = other.get_output()
        return WorldState(sequences=sequences)

    def _update_topology(self, agents: dict[str, ConsciousAgent], agent_ids: list[str], generation: int) -> None:
        if self._topology_update_interval == 0 or generation % self._topology_update_interval != 0:
            return

        for i, id_i in enumerate(agent_ids):
            agent_i = agents.get(id_i)
            if agent_i is None:
                continue
            for j in range(len(agent_ids)):
                if i == j:
                    continue
                pred_error = agent_i.experience.trace_buffer.prediction_error_mean(window=self._prediction_window)
                reduction = 0.1 - pred_error if pred_error < 0.5 else -0.05
                self._topology.update_connection(i, j, reduction)

        self._topology.prune_weak_connections()


class AgentNetwork:
    def __init__(
        self,
        n_agents: int = 20,
        initial_connections: int = 5,
        meta_observation_interval: int = 20,
        seed: int = 42,
        self_referential: bool = True,
    ) -> None:
        self._n_agents = n_agents
        self._agents: dict[str, ConsciousAgent] = {}
        self._agent_ids: list[str] = []
        self._generation = 0
        self._output_history: list[dict[str, list[str]]] = []

        self._topology = Topology(
            n_agents=n_agents,
            initial_connections_per_agent=initial_connections,
            seed=seed,
        )

        self._cycle = InteractionCycle(
            topology=self._topology,
            meta_observation_interval=meta_observation_interval,
        )

        self._init_agents(seed)
        self.combination_log: list[dict] = []

    def _init_agents(self, seed: int) -> None:
        for i in range(self._n_agents):
            agent_id = f"CA_{i:03d}"
            agent = ConsciousAgent(agent_id=agent_id)
            self._agents[agent_id] = agent
            self._agent_ids.append(agent_id)
        self._topology.set_agent_ids(self._agent_ids)

    def step(self) -> dict:
        outputs = self._cycle.step(self._agents, self._agent_ids, generation=self._generation)
        self._output_history.append(outputs)
        state = {
            "generation": self._generation,
            "outputs": outputs,
        }
        self._generation += 1
        return state

    def run(self, n_generations: int) -> list[dict]:
        states = []
        for _ in range(n_generations):
            states.append(self.step())
        return states

    def get_agent(self, agent_id: str) -> ConsciousAgent | None:
        return self._agents.get(agent_id)

    @property
    def agents(self) -> dict[str, ConsciousAgent]:
        return dict(self._agents)

    @property
    def agent_list(self) -> list[ConsciousAgent]:
        return [self._agents[aid] for aid in self._agent_ids if aid in self._agents]

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    @property
    def generation(self) -> int:
        return self._generation

    def step_all(self, world_state: WorldState) -> list[StepOutput]:
        return [self._agents[aid].step(world_state)
                for aid in self._agent_ids if aid in self._agents]

    def avg_prediction_error(self) -> float:
        if not self._agent_ids:
            return 0.0
        errors = []
        for aid in self._agent_ids:
            ag = self._agents.get(aid)
            if ag is not None:
                errors.append(ag.experience.trace_buffer.prediction_error_mean(window=10))
        return sum(errors) / len(errors) if errors else 0.0

    def get_metrics(self) -> dict:
        step_results = [a.metrics for a in self.agent_list]
        if not step_results:
            return {
                "agent_count": 0, "mean_prediction_error": 0.0,
                "prediction_error_variance": 0.0, "i_lock_rate": 0.0,
                "mean_loop_depth": 0.0, "dominant_token_ratio": 0.0,
            }
        errors = [r["prediction_error"] for r in step_results]
        mean_pe = sum(errors) / len(errors)
        var_pe = sum((e - mean_pe) ** 2 for e in errors) / len(errors)
        i_lock_rate = sum(1 for r in step_results if r["i_locked"]) / len(step_results)
        mean_loop_depth = sum(r["loop_depth"] for r in step_results) / len(step_results)
        token_counts: dict[str, int] = {}
        for r in step_results:
            for t in r["output_tokens"]:
                token_counts[t] = token_counts.get(t, 0) + 1
        max_count = max(token_counts.values()) if token_counts else 0
        total_tokens = sum(token_counts.values()) if token_counts else 0
        dominant_token_ratio = max_count / total_tokens if total_tokens > 0 else 0.0
        return {
            "agent_count": len(step_results),
            "mean_prediction_error": mean_pe,
            "prediction_error_variance": var_pe,
            "i_lock_rate": i_lock_rate,
            "mean_loop_depth": mean_loop_depth,
            "dominant_token_ratio": dominant_token_ratio,
        }

    def get_agent_metrics(self, agent_id: str) -> dict | None:
        agent = self._agents.get(agent_id)
        return agent.metrics if agent is not None else None
