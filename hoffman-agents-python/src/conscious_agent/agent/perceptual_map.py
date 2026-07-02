from __future__ import annotations

import numpy as np

from ..core import TraceEvent, TraceBuffer, ExperienceLexicon, LexiconEntry, invent_token, is_invented_token
from .world_state import WorldState
from .experience_space import ExperienceSpace


def perceive(
    world: WorldState,
    experience: ExperienceSpace,
    step: int = 0,
    meta_observation_interval: int = 20,
    frozen: bool = False,
    ergodic_state: str = "idle",
) -> ExperienceSpace:
    if not world:
        return experience

    world_state_id = world.get_state_id()

    previous_id = experience.last_world_state_id
    if previous_id is not None:
        prediction = experience.trie.predict_next([previous_id])
        prediction_correct = prediction == world_state_id if prediction is not None else False
        prediction_error = _compute_prediction_error(prediction, world_state_id)
    else:
        prediction = None
        prediction_correct = False
        prediction_error = 0.5

    event = TraceEvent(
        from_state=previous_id if previous_id is not None else -1,
        to_state=world_state_id,
        timestamp=step,
        prediction=prediction if prediction is not None else -1,
        prediction_correct=prediction_correct,
        prediction_error=prediction_error,
        token=None,
    )

    experience.trace_buffer.append(event)

    if not frozen:
        experience.trie.insert([event.to_state], prediction_error)
        if event.from_state >= 0:
            experience.trie.insert([event.from_state, event.to_state], prediction_error)

        _update_lexicon(experience, world, world_state_id, prediction_error, step)

        if step > 0 and step % meta_observation_interval == 0:
            meta_id = experience.meta_trie.observe_self(
                experience.trace_buffer, timestamp=step,
                ergodic_state=ergodic_state, is_locked=experience.self_token.locked,
            )
            experience.self_token.update(experience.meta_trie, generation=step)

        if step > 0 and step % (meta_observation_interval * 3) == 0:
            _check_bind_proto_word(experience, world_state_id, step)

    experience.last_world_state_id = world_state_id
    return experience


def _compute_prediction_error(
    prediction: int | None,
    actual: int,
) -> float:
    if prediction is None:
        return 1.0
    return 0.0 if prediction == actual else 1.0


def _update_lexicon(
    experience: ExperienceSpace,
    world: WorldState,
    world_state_id: int,
    prediction_error: float,
    step: int,
) -> None:
    _decay_lexicon(experience)

    for agent_id, sequence in world.sequences.items():
        if agent_id == "world":
            continue
        for token in sequence:
            if not is_invented_token(token):
                continue
            existing = _lookup_by_output_token(experience, token)
            if existing is not None:
                existing.encounter_count += 1
                existing.integration_depth = min(existing.integration_depth + 0.05, 1.0)
                experience.lexicon.update_integration(existing.label, True)
            else:
                sig = _build_transition_signature(
                    experience.last_world_state_id, world_state_id,
                    experience.lexicon.embedding_dim,
                )
                label = f"adopted:{token}"
                entry = experience.lexicon.bind(
                    label=label,
                    output_token=token,
                    trace_signature=sig,
                    prediction_error_peak=prediction_error,
                    source="adopted",
                    generation=step,
                    step=step,
                )
                entry.integration_depth = 0.7
                entry.encounter_count = 1

    if prediction_error < 0.3:
        return

    label = f"p:{world_state_id:08x}"
    if experience.lexicon.lookup_by_label(label) is not None:
        return

    sig = _build_transition_signature(
        experience.last_world_state_id, world_state_id,
        experience.lexicon.embedding_dim,
    )

    output_token = invent_token()
    entry = experience.lexicon.bind(
        label=label,
        output_token=output_token,
        trace_signature=sig,
        prediction_error_peak=prediction_error,
        source="proto",
        generation=step,
        step=step,
    )
    entry.integration_depth = 0.3


def _decay_lexicon(experience: ExperienceSpace) -> None:
    for entry in experience.lexicon._entries.values():
        if entry.encounter_count == 0 and entry.labeling_source != "adopted":
            entry.integration_depth *= 0.999
        if entry.integration_depth < 0.01:
            entry.integration_depth = 0.01


def _build_transition_signature(
    prev_id: int | None,
    curr_id: int,
    embedding_dim: int,
) -> np.ndarray:
    sig = np.zeros(embedding_dim, dtype=np.float64)
    if prev_id is not None:
        combined = hash(f"{prev_id}->{curr_id}")
        sig[combined % embedding_dim] = 1.0
    sig[curr_id % embedding_dim] = 1.0
    norm = np.linalg.norm(sig)
    if norm > 0:
        sig = sig / norm
    return sig


def _lookup_by_output_token(experience: ExperienceSpace, token: str) -> LexiconEntry | None:
    for entry in experience.lexicon._entries.values():
        if entry.output_token == token:
            return entry
    return None


def _check_bind_proto_word(
    experience: ExperienceSpace,
    world_state_id: int,
    step: int,
) -> None:
    recent_errors = experience.trace_buffer.prediction_error_mean(window=20)
    if recent_errors > 0.6:
        label = f"p:{world_state_id:08x}"
        if experience.lexicon.lookup_by_label(label) is not None:
            return

        sig = _build_transition_signature(
            experience.last_world_state_id, world_state_id,
            experience.lexicon.embedding_dim,
        )

        output_token = invent_token()
        experience.lexicon.bind(
            label=label,
            output_token=output_token,
            trace_signature=sig,
            prediction_error_peak=recent_errors,
            source="proto",
            generation=step,
            step=step,
        )
