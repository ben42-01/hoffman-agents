from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..agent import WorldState as _WorldState


@dataclass
class FeatureSpec:
    name: str
    normalization: str = "minmax"
    n_bins: int = 4
    params: dict[str, Any] = field(default_factory=dict)


class Normalizer:
    def __init__(self, feature: FeatureSpec) -> None:
        self.feature = feature
        self.min_val: float | None = None
        self.max_val: float | None = None
        self.percentiles: list[float] | None = None
        self.bin_edges: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        if self.feature.normalization == "minmax":
            self.min_val = float(np.min(values))
            self.max_val = float(np.max(values))
            if self.max_val == self.min_val:
                self.max_val = self.min_val + 1.0
        elif self.feature.normalization == "percentile":
            window = self.feature.params.get("window", 500)
            n = min(len(values), window)
            self.percentiles = [np.percentile(values[-n:], p) for p in [25, 50, 75]]
        elif self.feature.normalization == "tanh":
            scale = self.feature.params.get("tanh_scale", 0.01)
            self.min_val = -scale
            self.max_val = scale

        # Compute bin edges
        raw = self._normalize_array(values)
        self.bin_edges = np.linspace(0.0, 1.0, self.feature.n_bins + 1)

    def transform(self, value: float) -> int:
        normalized = self._normalize_value(value)
        if self.bin_edges is None:
            return 0
        bin_idx = np.searchsorted(self.bin_edges, normalized, side="right") - 1
        return max(0, min(bin_idx, self.feature.n_bins - 1))

    def _normalize_array(self, values: np.ndarray) -> np.ndarray:
        if self.feature.normalization == "minmax":
            return (values - self.min_val) / (self.max_val - self.min_val)
        elif self.feature.normalization == "percentile":
            return np.clip((values - self.percentiles[0]) / (self.percentiles[-1] - self.percentiles[0] + 1e-8), 0, 1)
        elif self.feature.normalization == "tanh":
            return np.tanh(values * self.feature.params.get("tanh_scale", 0.01)) * 0.5 + 0.5
        elif self.feature.normalization == "categorical":
            return values / max(values.max(), 1)
        return values

    def _normalize_value(self, value: float) -> float:
        if self.feature.normalization == "minmax":
            return (value - self.min_val) / (self.max_val - self.min_val)
        elif self.feature.normalization == "percentile":
            return np.clip((value - self.percentiles[0]) / (self.percentiles[-1] - self.percentiles[0] + 1e-8), 0, 1)
        elif self.feature.normalization == "tanh":
            return np.tanh(value * self.feature.params.get("tanh_scale", 0.01)) * 0.5 + 0.5
        return value


class WorldBuilder:
    def __init__(self) -> None:
        self._features: list[FeatureSpec] = []
        self._smoothing: float = 0.001
        self._metadata: dict[str, Any] = {}

    def add_feature(
        self,
        name: str,
        normalization: str = "minmax",
        n_bins: int = 4,
        **params: Any,
    ) -> WorldBuilder:
        self._features.append(FeatureSpec(
            name=name,
            normalization=normalization,
            n_bins=n_bins,
            params=params,
        ))
        return self

    def set_smoothing(self, smoothing: float) -> WorldBuilder:
        self._smoothing = smoothing
        return self

    def set_metadata(self, **metadata: Any) -> WorldBuilder:
        self._metadata.update(metadata)
        return self

    def build(self, data: np.ndarray) -> World:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Fit normalizers
        normalizers = []
        for i, feature in enumerate(self._features):
            col = data[:, i] if data.shape[1] > i else np.zeros(len(data))
            norm = Normalizer(feature)
            norm.fit(col)
            normalizers.append(norm)

        # Discretize
        state_ids = []
        n_rows = data.shape[0]
        for row_idx in range(n_rows):
            bins = []
            for i, norm in enumerate(normalizers):
                col = data[row_idx, i] if data.shape[1] > i else 0.0
                bins.append(norm.transform(float(col)))
            state_id = hash(tuple(bins)) & 0x7FFFFFFF
            state_ids.append(state_id)

        # Build transition matrix
        unique = sorted(set(state_ids))
        id_to_idx = {sid: i for i, sid in enumerate(unique)}
        n_states = len(unique)
        P = np.full((n_states, n_states), self._smoothing)
        for i in range(len(state_ids) - 1):
            from_idx = id_to_idx[state_ids[i]]
            to_idx = id_to_idx[state_ids[i + 1]]
            P[from_idx, to_idx] += 1.0
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P = P / row_sums

        labels = {sid: f"s{sid % 1000:03d}" for sid in unique}

        return World(
            n_states=n_states,
            transition_matrix=P,
            state_ids=state_ids,
            state_labels=labels,
            normalizers=normalizers,
            initial_state=state_ids[0] if state_ids else 0,
        )


class World:
    def __init__(
        self,
        n_states: int,
        transition_matrix: np.ndarray,
        state_ids: list[int],
        state_labels: dict[int, str],
        normalizers: list[Normalizer] | None = None,
        initial_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.transition_matrix = transition_matrix
        self.state_ids = state_ids
        self.state_labels = state_labels
        self.normalizers = normalizers or []
        self._current_state = initial_state

    def step(self) -> _WorldState:
        row = self.transition_matrix[self._current_state]
        next_idx = np.random.choice(self.n_states, p=row)
        self._current_state = next_idx
        return _WorldState.from_sequence("world", [self.get_label(next_idx)])

    def step_from_data(self, state_id: int) -> int:
        self._current_state = state_id
        return state_id

    def state_from_new_data(self, data: dict[str, float]) -> int:
        bins = []
        for i, norm in enumerate(self.normalizers):
            feature_name = norm.feature.name
            val = data.get(feature_name, 0.0)
            bins.append(norm.transform(float(val)))
        return hash(tuple(bins)) & 0x7FFFFFFF

    def get_label(self, state_id: int) -> str:
        return self.state_labels.get(state_id, str(state_id))

    @property
    def initial_state(self) -> int:
        return self._current_state

    @staticmethod
    def from_components(
        state_ids: list[int],
        transition_matrix: np.ndarray,
        state_labels: dict[int, str] | None = None,
    ) -> World:
        n_states = transition_matrix.shape[0]
        if state_labels is None:
            state_labels = {i: f"s{i}" for i in range(n_states)}
        return World(
            n_states=n_states,
            transition_matrix=transition_matrix,
            state_ids=state_ids,
            state_labels=state_labels,
            initial_state=state_ids[0] if state_ids else 0,
        )


class CoinTossWorld:
    def __init__(self, n_coins: int = 4) -> None:
        self.n_coins = n_coins
        self.n_states = 2 ** n_coins
        self._rng = np.random.RandomState(42)
        self._state = 0
        self._transition_matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        P = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        return P

    def step(self) -> _WorldState:
        self._state = int(self._rng.choice(self.n_states, p=self._transition_matrix[self._state]))
        return _WorldState.from_sequence("world", [str(self._state)])


def build_world_from_dataframe(
    df: Any,
    feature_specs: list[dict] | None = None,
) -> World:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "build_world_from_dataframe requires pandas. "
            "Install it with: pip install pandas"
        )
    builder = WorldBuilder()
    if feature_specs:
        for spec in feature_specs:
            builder.add_feature(**spec)
    else:
        for col in df.select_dtypes(include=[np.number]).columns[:6]:
            builder.add_feature(name=col, normalization="minmax", n_bins=4)
    data = df.to_numpy(dtype=np.float64)
    return builder.build(data)
