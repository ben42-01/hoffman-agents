from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self


@dataclass
class TrieNode:
    state_id: int
    children: dict[int, TrieNode] = field(default_factory=dict)
    visit_count: int = 0
    prediction_errors: list[float] = field(default_factory=list)
    mean_prediction_error: float = 0.0
    word_binding: str | None = None
    depth: int = 0

    def add_prediction_error(self, error: float) -> None:
        n = len(self.prediction_errors)
        self.mean_prediction_error = (self.mean_prediction_error * n + error) / (n + 1)
        self.prediction_errors.append(error)


class ExperienceTrie:
    def __init__(self, max_depth: int = 10) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self._root = TrieNode(state_id=-1)
        self._max_depth = max_depth

    def insert(self, path: list[int], prediction_error: float = 0.0) -> None:
        if not path:
            return
        if len(path) > self._max_depth:
            path = path[:self._max_depth]
        node = self._root
        for depth, state_id in enumerate(path, start=1):
            if state_id not in node.children:
                node.children[state_id] = TrieNode(state_id=state_id, depth=depth)
            node = node.children[state_id]
            node.visit_count += 1
        node.add_prediction_error(prediction_error)

    def lookup(self, path: list[int]) -> TrieNode | None:
        node = self._root
        for state_id in path:
            if state_id not in node.children:
                return None
            node = node.children[state_id]
        return node

    def predict_next(self, path: list[int]) -> int | None:
        node = self.lookup(path)
        if node is None or not node.children:
            return None
        return max(node.children.items(), key=lambda x: x[1].visit_count)[0]

    def get_all_paths(self, min_visits: int = 1) -> list[list[int]]:
        paths: list[list[int]] = []

        def dfs(node: TrieNode, current: list[int]) -> None:
            if node is not self._root and node.visit_count >= min_visits:
                paths.append(list(current))
            for child_state, child in node.children.items():
                current.append(child_state)
                dfs(child, current)
                current.pop()

        dfs(self._root, [])
        return paths

    def get_leaf_paths(self, min_visits: int = 1) -> list[list[int]]:
        paths: list[list[int]] = []

        def dfs(node: TrieNode, current: list[int]) -> None:
            if not node.children:
                if node is not self._root and node.visit_count >= min_visits:
                    paths.append(list(current))
                return
            for child_state, child in node.children.items():
                current.append(child_state)
                dfs(child, current)
                current.pop()

        dfs(self._root, [])
        return paths

    def compress(self, min_visits: int) -> None:
        def _prune(node: TrieNode) -> bool:
            children_to_delete: list[int] = []
            for child_state, child in node.children.items():
                if _prune(child):
                    children_to_delete.append(child_state)
            for cs in children_to_delete:
                del node.children[cs]
            if node is not self._root and node.visit_count < min_visits and not node.children:
                return True
            return False

        _prune(self._root)

    def merge(self, other: ExperienceTrie) -> ExperienceTrie:
        merged = ExperienceTrie(max_depth=max(self._max_depth, other._max_depth))

        path_data: dict[tuple[int, ...], tuple[int, list[float]]] = {}

        def _collect(node: TrieNode, path: list[int]) -> None:
            for child_state, child in node.children.items():
                path.append(child_state)
                key = tuple(path)
                vc, errs = path_data.get(key, (0, []))
                path_data[key] = (vc + child.visit_count, errs + child.prediction_errors)
                _collect(child, path)
                path.pop()

        _collect(self._root, [])
        _collect(other._root, [])

        for path_tuple in sorted(path_data.keys(), key=len):
            path = list(path_tuple)
            visit_count, pred_errors = path_data[path_tuple]
            merged.insert(path)
            node = merged.lookup(path)
            if node is not None:
                node.visit_count = visit_count
                node.prediction_errors = pred_errors
                if pred_errors:
                    node.mean_prediction_error = sum(pred_errors) / len(pred_errors)

        return merged

    def size(self) -> int:
        count = 0

        def _count(node: TrieNode) -> None:
            nonlocal count
            count += 1
            for child in node.children.values():
                _count(child)

        _count(self._root)
        return count

    def compression_ratio(self, total_steps: int) -> float:
        if total_steps == 0:
            return 0.0
        return self.size() / total_steps

    @property
    def root(self) -> TrieNode:
        return self._root

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def clear(self) -> None:
        self._root = TrieNode(state_id=-1)
