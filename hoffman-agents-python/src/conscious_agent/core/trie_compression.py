from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .experience_trie import ExperienceTrie


def prune(
    trie: ExperienceTrie,
    min_visits: int,
    protected_nodes: set[int] | None = None,
) -> None:
    if protected_nodes is None:
        protected_nodes = set()

    def _can_prune(node_state: int) -> bool:
        return node_state not in protected_nodes

    def _prune_recursive(parent_path: list[int]) -> None:
        node = trie.lookup(parent_path) if parent_path else trie.root
        if node is None:
            return

        children_to_check = list(node.children.keys())
        for child_state in children_to_check:
            child_path = parent_path + [child_state]
            _prune_recursive(child_path)

        children_to_delete = [
            cs for cs in node.children.keys()
            if node.children[cs].visit_count < min_visits
            and not node.children[cs].children
            and _can_prune(node.children[cs].state_id)
        ]
        for cs in children_to_delete:
            del node.children[cs]

    _prune_recursive([])


def trace_distance(
    path_a: Sequence[int],
    path_b: Sequence[int],
    transition_matrix: np.ndarray | None = None,
) -> float:
    n = len(path_a)
    m = len(path_b)

    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    dp[:, 0] = np.arange(n + 1, dtype=np.float64)
    dp[0, :] = np.arange(m + 1, dtype=np.float64)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if path_a[i - 1] == path_b[j - 1]:
                cost = 0.0
            elif transition_matrix is not None:
                a_state = path_a[i - 1]
                b_state = path_b[j - 1]
                max_idx = max(a_state, b_state)
                tm = transition_matrix
                if max_idx < tm.shape[0] and max_idx < tm.shape[1]:
                    sim = _cosine_similarity(tm[a_state], tm[b_state])
                    cost = 1.0 - sim
                else:
                    cost = 1.0
            else:
                cost = 1.0

            dp[i, j] = min(
                dp[i - 1, j] + 1.0,
                dp[i, j - 1] + 1.0,
                dp[i - 1, j - 1] + cost,
            )

    max_len = max(n, m)
    if max_len == 0:
        return 0.0
    return float(dp[n, m] / max_len)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def merge_similar_paths(
    trie: ExperienceTrie,
    transition_matrix: np.ndarray | None = None,
    threshold: float = 0.15,
    protected_nodes: set[int] | None = None,
) -> int:
    if protected_nodes is None:
        protected_nodes = set()

    leaf_paths = trie.get_leaf_paths(min_visits=1)
    merged_count = 0
    merged_paths: set[tuple[int, ...]] = set()

    for i in range(len(leaf_paths)):
        path_i = leaf_paths[i]
        key_i = tuple(path_i)

        if key_i in merged_paths:
            continue

        node_i = trie.lookup(path_i)
        if node_i is None:
            continue
        if node_i.state_id in protected_nodes:
            continue

        for j in range(i + 1, len(leaf_paths)):
            path_j = leaf_paths[j]
            key_j = tuple(path_j)

            if key_j in merged_paths:
                continue

            node_j = trie.lookup(path_j)
            if node_j is None:
                continue
            if node_j.state_id in protected_nodes:
                continue

            dist = trace_distance(path_i, path_j, transition_matrix)
            if dist < threshold:
                if node_i.visit_count >= node_j.visit_count:
                    node_i.visit_count += node_j.visit_count
                    _remove_path(trie, path_j)
                    merged_paths.add(key_j)
                else:
                    node_j.visit_count += node_i.visit_count
                    _remove_path(trie, path_i)
                    merged_paths.add(key_i)
                    break

                merged_count += 1

    return merged_count


def _remove_path(trie: ExperienceTrie, path: list[int]) -> None:
    if not path:
        return

    for depth in range(len(path), 0, -1):
        prefix = path[:depth]
        node = trie.lookup(prefix)
        if node is None:
            continue

        if depth < len(path):
            child_state = path[depth]
            if child_state in node.children:
                if node.children[child_state].visit_count <= 0 and not node.children[child_state].children:
                    del node.children[child_state]
        else:
            node.visit_count = 0
            node.prediction_errors.clear()
            node.mean_prediction_error = 0.0
