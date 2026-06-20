from __future__ import annotations

from ..core import ExperienceLexicon


class SharedMeaningTracker:
    def __init__(self) -> None:
        self._epochs: list[dict] = []

    def snapshot(
        self,
        agents: dict[str, ExperienceLexicon],
        generation: int,
    ) -> dict:
        all_tokens: dict[str, set[str]] = {}
        all_labels: dict[str, set[str]] = {}
        for aid, lex in agents.items():
            tokens = set()
            labels = set()
            for entry in lex._entries.values():
                if entry.output_token:
                    tokens.add(entry.output_token)
                labels.add(entry.label)
            all_tokens[aid] = tokens
            all_labels[aid] = labels

        sharedness = self._compute_sharedness(all_tokens)
        self._epochs.append({
            "generation": generation,
            "sharedness": sharedness,
            "token_overlap": self._compute_overlap(all_tokens),
        })
        return self._epochs[-1]

    def _compute_sharedness(self, all_tokens: dict[str, set[str]]) -> float:
        if not all_tokens:
            return 0.0
        agent_ids = list(all_tokens.keys())
        if len(agent_ids) < 2:
            return 0.0

        total_pairs = 0
        shared_pairs = 0
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                set_i = all_tokens[agent_ids[i]]
                set_j = all_tokens[agent_ids[j]]
                union = set_i | set_j
                if not union:
                    continue
                total_pairs += 1
                intersection = set_i & set_j
                shared_pairs += len(intersection) / len(union)

        return shared_pairs / total_pairs if total_pairs > 0 else 0.0

    def _compute_overlap(self, all_tokens: dict[str, set[str]]) -> dict[str, int]:
        from collections import Counter
        token_counts: Counter = Counter()
        for tokens in all_tokens.values():
            token_counts.update(tokens)
        return dict(token_counts.most_common(10))
