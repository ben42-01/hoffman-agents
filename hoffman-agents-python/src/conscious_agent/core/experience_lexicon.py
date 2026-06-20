from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LexiconEntry:
    label: str
    output_token: str
    trace_signature: np.ndarray
    prediction_error_peak: float = 0.0
    integration_depth: float = 0.5
    encounter_count: int = 0
    generation_bound: int = 0
    step_bound: int = 0
    labeling_source: str = "proto"
    associated_labels: list[str] = field(default_factory=list)


class ExperienceLexicon:
    def __init__(self, embedding_dim: int = 64, association_threshold: float = 0.15) -> None:
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        self._embedding_dim = embedding_dim
        self._association_threshold = association_threshold
        self._entries: dict[str, LexiconEntry] = {}
        self._signatures: list[tuple[np.ndarray, str]] = []

    def bind(
        self,
        label: str,
        trace_signature: np.ndarray,
        prediction_error_peak: float = 0.0,
        source: str = "proto",
        generation: int = 0,
        step: int = 0,
        output_token: str | None = None,
    ) -> LexiconEntry:
        trace_signature = self._normalize_signature(trace_signature)
        associations = self._compute_associations(trace_signature)

        if output_token is None:
            output_token = label

        entry = LexiconEntry(
            label=label,
            output_token=output_token,
            trace_signature=trace_signature,
            prediction_error_peak=prediction_error_peak,
            generation_bound=generation,
            step_bound=step,
            labeling_source=source,
            associated_labels=associations,
        )

        self._entries[label] = entry
        self._signatures.append((trace_signature, label))
        return entry

    def lookup_by_label(self, label: str) -> LexiconEntry | None:
        return self._entries.get(label)

    def lookup_by_signature(
        self,
        trace_signature: np.ndarray,
        threshold: float = 0.2,
    ) -> LexiconEntry | None:
        trace_signature = self._normalize_signature(trace_signature)

        best_dist = float("inf")
        best_label: str | None = None

        for sig, label in self._signatures:
            dist = float(np.linalg.norm(sig - trace_signature))
            if dist < best_dist:
                best_dist = dist
                best_label = label

        if best_label is not None and best_dist < threshold:
            return self._entries[best_label]
        return None

    def update_integration(self, label: str, encountered: bool) -> None:
        entry = self._entries.get(label)
        if entry is None:
            return

        delta_up = 0.05
        delta_down = 0.01

        if encountered:
            entry.integration_depth = min(entry.integration_depth + delta_up, 1.0)
            entry.encounter_count += 1
        else:
            entry.integration_depth = max(entry.integration_depth - delta_down, 0.0)

    def nearest_label(self, trace_signature: np.ndarray) -> tuple[str, float]:
        trace_signature = self._normalize_signature(trace_signature)

        if not self._signatures:
            return ("", 1.0)

        best_label = ""
        best_dist = float("inf")

        for sig, label in self._signatures:
            dist = float(np.linalg.norm(sig - trace_signature))
            if dist < best_dist:
                best_dist = dist
                best_label = label

        return (best_label, best_dist)

    def vocabulary_size(self) -> int:
        return sum(1 for e in self._entries.values() if e.integration_depth > 0.1)

    def sorted_by_integration(self) -> list[LexiconEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: e.integration_depth,
            reverse=True,
        )

    def _normalize_signature(self, sig: np.ndarray) -> np.ndarray:
        if sig.ndim == 0:
            arr = np.full(self._embedding_dim, float(sig))
        elif sig.ndim == 1 and sig.shape[0] < self._embedding_dim:
            arr = np.pad(sig, (0, self._embedding_dim - sig.shape[0]))
        elif sig.ndim == 1:
            arr = sig[:self._embedding_dim]
        else:
            arr = sig.flatten()[:self._embedding_dim]

        if arr.shape[0] < self._embedding_dim:
            arr = np.pad(arr, (0, self._embedding_dim - arr.shape[0]))

        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float64)

    def _compute_associations(self, trace_signature: np.ndarray) -> list[str]:
        trace_signature = self._normalize_signature(trace_signature)
        associations: list[tuple[float, str]] = []
        for sig, label in self._signatures:
            dist = float(np.linalg.norm(sig - trace_signature))
            if dist < self._association_threshold:
                associations.append((dist, label))

        associations.sort()
        return [label for _, label in associations[:10]]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def relabel(self, old_label: str, new_label: str, new_output_token: str | None = None) -> LexiconEntry | None:
        entry = self._entries.get(old_label)
        if entry is None:
            return None
        if new_output_token is None:
            new_output_token = new_label
        self._signatures = [(sig, lbl) for sig, lbl in self._signatures if lbl != old_label]
        del self._entries[old_label]
        entry.label = new_label
        entry.output_token = new_output_token
        entry.labeling_source = "parent"
        self._entries[new_label] = entry
        self._signatures.append((entry.trace_signature, new_label))
        return entry

    def remove(self, label: str) -> bool:
        if label not in self._entries:
            return False
        self._signatures = [(sig, lbl) for sig, lbl in self._signatures if lbl != label]
        del self._entries[label]
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._signatures.clear()

    def to_dict(self) -> dict:
        entries = []
        for entry in self._entries.values():
            entries.append({
                "label": entry.label,
                "output_token": entry.output_token,
                "trace_signature": entry.trace_signature.tolist(),
                "prediction_error_peak": entry.prediction_error_peak,
                "integration_depth": entry.integration_depth,
                "encounter_count": entry.encounter_count,
                "generation_bound": entry.generation_bound,
                "step_bound": entry.step_bound,
                "labeling_source": entry.labeling_source,
                "associated_labels": list(entry.associated_labels),
            })
        return {
            "embedding_dim": self._embedding_dim,
            "association_threshold": self._association_threshold,
            "entries": entries,
        }

    @staticmethod
    def from_dict(data: dict) -> ExperienceLexicon:
        lex = ExperienceLexicon(
            embedding_dim=data.get("embedding_dim", 64),
            association_threshold=data.get("association_threshold", 0.15),
        )
        for ed in data.get("entries", []):
            sig = np.array(ed["trace_signature"], dtype=np.float64)
            entry = LexiconEntry(
                label=ed["label"],
                output_token=ed["output_token"],
                trace_signature=sig,
                prediction_error_peak=ed.get("prediction_error_peak", 0.0),
                integration_depth=ed.get("integration_depth", 0.5),
                encounter_count=ed.get("encounter_count", 0),
                generation_bound=ed.get("generation_bound", 0),
                step_bound=ed.get("step_bound", 0),
                labeling_source=ed.get("labeling_source", "proto"),
                associated_labels=list(ed.get("associated_labels", [])),
            )
            lex._entries[ed["label"]] = entry
            lex._signatures.append((sig, ed["label"]))
        return lex
