"""
Exchange Symmetry Test

Tests whether the combination operator produces exchange-symmetric or
exchange-antisymmetric results under swapping of constituent agents.

Phase A — Exchange invariance:
  Create identical twin agents via clone_agent(), combine as AB and BA,
  measure whether observable quantities (spectral gap, loop score, etc.)
  change under exchange.

Phase B — Occupation / redundancy:
  Combine identical twins vs genuinely distinct agents, measure the
  "redundancy ratio" of the merged experience space. If identical agents
  produce near-duplicate content (ratio -> 0.5) while distinct agents
  produce additive content (ratio -> 1.0), that is an emergent signature
  analogous to exclusion-like behavior in information space.

This is an exploratory computational analogy — NOT a physics claim.
"""

from __future__ import annotations

import time
import math
import random

import numpy as np

from conscious_agent import (
    ConsciousAgent,
    WorldState,
    ExperienceSpace,
    SelfTokenState,
    SimpleWorld,
    combine,
    fuse,
    experience_space_distance,
    clone_agent,
    ExperienceTrie,
    TraceBuffer,
    TraceEvent,
)


# ────────────── Helper utilities ──────────────


def spectral_gap(P: np.ndarray) -> dict:
    n = P.shape[0]
    if n < 2:
        return {"gap": 1.0, "convergence_error": 0.0}

    # Stationary distribution
    pi = np.ones(n) / n
    for _ in range(1000):
        pi_next = pi @ P
        diff = np.max(np.abs(pi_next - pi))
        pi = pi_next
        if diff < 1e-10:
            break

    # Deflate: B = P - 1 @ pi^T
    B = P - np.ones((n, 1)) @ pi.reshape(1, n)

    # Power iteration for dominant eigenvalue of B
    b = np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(500):
        b_next = B @ b
        norm = np.linalg.norm(b_next)
        if norm > 0:
            b_next = b_next / norm
        lam = (b_next @ b) / (b @ b)
        b = b_next

    return {"gap": max(0.0, 1.0 - abs(lam)), "lambda": abs(lam)}


def detailed_balance_error(P: np.ndarray) -> float:
    n = P.shape[0]
    if n < 2:
        return 0.0

    pi = np.ones(n) / n
    for _ in range(1000):
        pi_next = pi @ P
        diff = np.max(np.abs(pi_next - pi))
        pi = pi_next
        if diff < 1e-10:
            break

    err_sum = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            fwd = pi[i] * P[i, j]
            rev = pi[j] * P[j, i]
            denom = fwd + rev
            if denom > 1e-12:
                err_sum += abs(fwd - rev) / denom
                pair_count += 1
    return err_sum / pair_count if pair_count > 0 else 0.0


def extract_meta_matrix(agent: ConsciousAgent) -> tuple[np.ndarray, int]:
    mt = agent.experience.meta_trie
    n_registry = mt.registry_size
    if n_registry < 2:
        return np.array([[1.0]]), 1

    all_ids = list(mt._registry.keys())
    active_ids = [sid for sid in all_ids if mt.trie.lookup([sid]) is not None]
    if len(active_ids) < 2:
        active_ids = all_ids[:]

    sorted_ids = sorted(set(active_ids))
    idx = {sid: i for i, sid in enumerate(sorted_ids)}
    n = len(sorted_ids)
    P = np.zeros((n, n))

    for state_id in sorted_ids:
        node = mt.trie.lookup([state_id])
        if node is None:
            continue
        total = sum(child.visit_count for child in node.children.values())
        if total == 0:
            continue
        i = idx[state_id]
        for child_state_id, child in node.children.items():
            j = idx.get(child_state_id)
            if j is not None:
                P[i, j] = child.visit_count / total

    return P, n


def measure_agent(agent: ConsciousAgent) -> dict:
    P, state_count = extract_meta_matrix(agent)
    gap = spectral_gap(P)["gap"]
    db_err = detailed_balance_error(P) if state_count > 1 else 0.0
    return {
        "spectral_gap": gap,
        "detailed_balance_error": db_err,
        "loop_score": agent.loop_score,
        "mean_prediction_error": agent.mean_prediction_error,
        "is_i_locked": agent.is_i_locked,
        "cycle_level": agent.cycle_level,
    }


def redundancy_ratio(combined, parent1, parent2):
    comb_size = combined.experience.trie.size()
    p1_size = parent1.experience.trie.size()
    p2_size = parent2.experience.trie.size()
    sum_size = p1_size + p2_size
    return {
        "trie_ratio": comb_size / sum_size if sum_size > 0 else 0.0,
        "combined_trie_size": comb_size,
        "parent_sum_trie_size": sum_size,
    }


# ────────────── Phase A: Exchange invariance ──────────────


def phase_a(n_trials: int = 20):
    print("\n  Phase A: Exchange Invariance\n")

    results = []

    for trial in range(n_trials):
        seed = 42 + trial
        world = SimpleWorld(n_states=5, seed=seed)

        # Train parent agent
        A = ConsciousAgent(agent_id=f"A_{trial}", meta_observation_interval=10)
        for _ in range(300):
            A.step(world.step())

        # Clone to create identical twin B
        B = clone_agent(A, new_id=f"B_{trial}")

        # Combine both ways
        AB = combine(A, B)
        BA = combine(B, A)

        # Isolate order-effects of combine() itself from step()-time RNG
        # noise: give both agents identical, freshly-seeded random streams
        # so their post-combination decision-time token sampling is
        # directly comparable.
        step_seed = 10000 + trial
        AB._rng = random.Random(step_seed)
        BA._rng = random.Random(step_seed)

        # Run both for many steps, tracking running average loop score
        # (a single instantaneous loop_score read is a 1-sample stochastic
        # quantity — average over the run for a statistically meaningful value)
        n_run_steps = 100
        loop_sum_ab = 0.0
        loop_sum_ba = 0.0
        for _ in range(n_run_steps):
            ws = world.step()
            AB.step(ws)
            BA.step(ws)
            loop_sum_ab += AB.loop_score
            loop_sum_ba += BA.loop_score
        mean_loop_ab = loop_sum_ab / n_run_steps
        mean_loop_ba = loop_sum_ba / n_run_steps

        m_ab = measure_agent(AB)
        m_ba = measure_agent(BA)
        m_ab["loop_score"] = mean_loop_ab
        m_ba["loop_score"] = mean_loop_ba
        red_ab = redundancy_ratio(AB, A, B)
        red_ba = redundancy_ratio(BA, B, A)
        dist = experience_space_distance(AB.experience, BA.experience)

        results.append(
            {
                "trial": trial,
                "spectral_gap": {
                    "AB": m_ab["spectral_gap"],
                    "BA": m_ba["spectral_gap"],
                    "delta": abs(m_ab["spectral_gap"] - m_ba["spectral_gap"]),
                },
                "db_error": {
                    "AB": m_ab["detailed_balance_error"],
                    "BA": m_ba["detailed_balance_error"],
                    "delta": abs(m_ab["detailed_balance_error"] - m_ba["detailed_balance_error"]),
                },
                "loop_score": {
                    "AB": m_ab["loop_score"],
                    "BA": m_ba["loop_score"],
                    "delta": abs(m_ab["loop_score"] - m_ba["loop_score"]),
                },
                "pred_error": {
                    "AB": m_ab["mean_prediction_error"],
                    "BA": m_ba["mean_prediction_error"],
                    "delta": abs(m_ab["mean_prediction_error"] - m_ba["mean_prediction_error"]),
                },
                "locked": {
                    "AB": m_ab["is_i_locked"],
                    "BA": m_ba["is_i_locked"],
                },
                "experience_distance": dist,
            }
        )

    n = n_trials
    agg = {
        "spectral_gap": {
            "mean": sum(r["spectral_gap"]["delta"] for r in results) / n,
            "std": (sum((r["spectral_gap"]["delta"] - sum(r["spectral_gap"]["delta"] for r in results) / n) ** 2 for r in results) / n) ** 0.5,
        },
        "db_error": {
            "mean": sum(r["db_error"]["delta"] for r in results) / n,
            "std": (sum((r["db_error"]["delta"] - sum(r["db_error"]["delta"] for r in results) / n) ** 2 for r in results) / n) ** 0.5,
        },
        "loop_score": {
            "mean": sum(r["loop_score"]["delta"] for r in results) / n,
            "std": (sum((r["loop_score"]["delta"] - sum(r["loop_score"]["delta"] for r in results) / n) ** 2 for r in results) / n) ** 0.5,
        },
        "pred_error": {
            "mean": sum(r["pred_error"]["delta"] for r in results) / n,
            "std": (sum((r["pred_error"]["delta"] - sum(r["pred_error"]["delta"] for r in results) / n) ** 2 for r in results) / n) ** 0.5,
        },
        "experience_distance": sum(r["experience_distance"] for r in results) / n,
        "lock_agreement": sum(1 for r in results if r["locked"]["AB"] == r["locked"]["BA"]) / n,
    }

    print(f"  Trials: {n_trials}")
    print(f"\n    Observable                    Mean |d|  +/-1s     Interpretation")
    print(f"    {'─' * 75}")
    rows = [
        ("Spectral gap (1-|l2|)", agg["spectral_gap"], 0.05),
        ("Detailed balance error", agg["db_error"], 0.03),
        ("Loop score", agg["loop_score"], 0.05),
        ("Mean prediction error", agg["pred_error"], 0.03),
    ]
    for label, v, thr in rows:
        is_sym = v["mean"] < thr
        is_noisy = v["std"] > v["mean"] * 2 and v["mean"] > 0
        tag = "<- symmetric" if is_sym else ("<- noisy (increase steps)" if is_noisy else "<- biased")
        print(f"    {label:<32s} {v['mean']:.4f} +/- {v['std']:.4f}  {tag}")
    print(f"    Experience space distance       {agg['experience_distance']:.4f}")
    print(f"    Lock agreement (AB vs BA):      {agg['lock_agreement'] * 100:.0f}%")

    return agg


# ────────────── Phase B: Occupation / redundancy ──────────────


def phase_b():
    print("\n  Phase B: Redundancy Ratio — Identical vs Distinct\n")

    n_trials = 20
    ident_results = []
    distinct_results = []

    for trial in range(n_trials):
        seed1 = 42 + trial
        seed2 = 999 + trial

        world1 = SimpleWorld(n_states=5, seed=seed1)
        world2 = SimpleWorld(n_states=5, seed=seed2)

        A = ConsciousAgent(agent_id=f"A_{trial}")
        for _ in range(300):
            A.step(world1.step())

        A_twin = clone_agent(A, new_id=f"twin_{trial}")

        C = ConsciousAgent(agent_id=f"C_{trial}")
        for _ in range(300):
            C.step(world2.step())

        S = combine(A, A_twin)
        ratio_ident = redundancy_ratio(S, A, A_twin)
        ident_results.append(ratio_ident["trie_ratio"])

        D = combine(A, C)
        ratio_distinct = redundancy_ratio(D, A, C)
        distinct_results.append(ratio_distinct["trie_ratio"])

    ident_mean = sum(ident_results) / n_trials
    ident_std = (sum((v - ident_mean) ** 2 for v in ident_results) / n_trials) ** 0.5
    distinct_mean = sum(distinct_results) / n_trials
    distinct_std = (sum((v - distinct_mean) ** 2 for v in distinct_results) / n_trials) ** 0.5

    print(f"  Trials: {n_trials}")
    print(f"\n    Pair type      Mean ratio  +/-1s     Interpretation")
    print(f"    {'─' * 60}")
    print(f"    Identical twins   {ident_mean:.3f}     +/- {ident_std:.3f}  1.0 = additive, 0.5 = overlap")
    print(f"    Distinct agents  {distinct_mean:.3f}     +/- {distinct_std:.3f}  1.0 = additive, 0.5 = overlap")
    print(f"    Contrast:        {distinct_mean - ident_mean:.3f}  (higher = less redundancy)")

    if ident_mean < distinct_mean - 0.1:
        print("\n  > Identical twins produce more redundancy — content overlap detected")
        print("    (Analogous to exclusion-like behavior in information space)")
    elif abs(ident_mean - distinct_mean) < 0.05:
        print("\n  ~ No significant difference — redundancy is independent of constituent identity")
    else:
        print("\n  ~ Weak or noisy contrast")

    return {"identical": ident_mean, "distinct": distinct_mean}


# ────────────── Main ──────────────


def main():
    t0 = time.time()
    print("=" * 62)
    print("Exchange Symmetry Test")
    print("=" * 62)
    print("\nTesting whether the combination operator preserves or breaks")
    print("exchange symmetry when its constituent agents are swapped.\n")
    print("This is an exploratory computational analogy.")
    print("Results are directional signals for further formal investigation.")
    print()

    phase_a_result = phase_a(20)
    phase_b_result = phase_b()

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")
    print("=" * 62)
    print("Summary")
    print("=" * 62)
    print(f"\n  Exchange invariance: {'AB=BA across all observables' if phase_a_result['lock_agreement'] >= 0.5 else 'AB!=BA - order bias detected'}")
    contrast = phase_b_result["identical"] < phase_b_result["distinct"] - 0.05
    print(f"  Redundancy contrast: {'Identical agents are more redundant' if contrast else 'No contrast'}")
    print("\n  See README.md for interpretation and caveats.\n")


if __name__ == "__main__":
    main()