"""2.x I-lock rule, preserved verbatim for math_version='legacy'.

Locks when max stationary probability exceeds a fixed threshold for a few
observations; never unlocks.
"""
from __future__ import annotations


def update(st, meta_trie, generation: int) -> None:
    dist = meta_trie.stationary_distribution()
    if not dist:
        return

    dominant = max(dist, key=dist.get)
    prob = min(dist[dominant], 1.0)

    st.stationary_prob = prob
    st.stability_history.append(prob)
    if len(st.stability_history) > 20:
        st.stability_history.pop(0)

    if not st.locked:
        if prob > st.lock_threshold:
            st.consecutive_above_threshold += 1
            if st.consecutive_above_threshold >= st.lock_consecutive_required:
                st.locked = True
                st.referent_meta_state_id = dominant
                st.lock_generation = generation
        else:
            st.consecutive_above_threshold = 0
    else:
        st.referent_meta_state_id = dominant
