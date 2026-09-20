from conscious_agent import SimpleWorld, WorldState
from conscious_agent.math import mulberry32


class HomeWorld:
    """Stays in "home" with probability p_home, otherwise one of n states."""

    def __init__(self, seed: int, p_home: float = 0.9, n: int = 20) -> None:
        self.r, self.p, self.n = mulberry32(seed + 999), p_home, n

    def step(self) -> WorldState:
        return WorldState.from_sequence("world", ["home" if self.r.random() < self.p else f"s{int(self.r.random() * self.n)}"])


def noise_world(seed: int, n_states: int = 50) -> SimpleWorld:
    return SimpleWorld(n_states=n_states, seed=seed)


class SwitchingWorld:
    """Attractor until ``switch_at``, noise afterwards."""

    def __init__(self, seed: int, switch_at: int) -> None:
        self.home, self.noise, self.t, self.switch_at = HomeWorld(seed, 0.95), noise_world(seed), 0, switch_at

    def step(self) -> WorldState:
        self.t += 1
        return self.home.step() if self.t < self.switch_at else self.noise.step()


class ConstantWorld:
    def step(self) -> WorldState:
        return WorldState.from_sequence("world", ["x"])
