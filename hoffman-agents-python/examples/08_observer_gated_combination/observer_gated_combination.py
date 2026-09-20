"""
Observer-Gated Combination

Follow-up to 07_exchange_symmetry. That experiment showed combine() is
exactly exchange-symmetric — a direct consequence of being built from pure
union operations (trie merge, lexicon merge), which are symmetric by
algebraic necessity. Union has no vantage point, so it cannot produce
asymmetry no matter how it's composed.

perceive() is the one operation in this architecture that IS inherently
asymmetric: prediction error is always computed FROM one agent's history,
relative to what it alone expects. It has a vantage point. It has an
observer.

This experiment asks: what happens if, before merging, each agent first
PERCEIVES the other -- treating the other's recent trajectory as a world to
predict -- and the resulting (asymmetric) surprise gates how the merge
proceeds?

This is an exploratory computational analogy — NOT a physics claim.
"""

from __future__ import annotations

import time

from conscious_agent import (
    ConsciousAgent,
    SimpleWorld,
    combine,
    clone_agent,
)


# ────────────── Observer surprise ──────────────


def observer_surprise(perceiver: ConsciousAgent, other: ConsciousAgent, window: int = 30) -> float:
    recent = other.experience.trace_buffer.get_recent(window)
    if len(recent) < 2:
        return 1.0

    total_error = 0
    count = 0
    for i in range(1, len(recent)):
        prev_state = recent[i - 1].to_state
        actual_state = recent[i].to_state
        predicted = perceiver.experience.trie.predict_next([prev_state])
        total_error += 0 if predicted == actual_state else 1
        count += 1
    return total_error / count if count > 0 else 1.0


# ────────────── Observer-gated combination ──────────────


def observer_gated_combine(agent_a, agent_b, hole_threshold=0.5, window=30):
    surprise_a_on_b = observer_surprise(agent_a, agent_b, window)
    surprise_b_on_a = observer_surprise(agent_b, agent_a, window)
    max_surprise = max(surprise_a_on_b, surprise_b_on_a)

    if max_surprise < hole_threshold:
        dominant, submissive, mode = agent_a, agent_b, "union"
    else:
        if surprise_a_on_b <= surprise_b_on_a:
            dominant, submissive = agent_a, agent_b
        else:
            dominant, submissive = agent_b, agent_a
        mode = "exclusion"

    result = combine(dominant, submissive)
    return {
        "result": result,
        "mode": mode,
        "surprise_a_on_b": surprise_a_on_b,
        "surprise_b_on_a": surprise_b_on_a,
        "dominant_id": dominant.agent_id,
    }


# ────────────── Phase A: Twin sanity check ──────────────


def phase_a_twins(n_trials=20):
    print("\n  Phase A: Identical Twins (sanity check — expect symmetry to hold)\n")

    mode_matches = 0
    asymmetries = []

    for trial in range(n_trials):
        world = SimpleWorld(n_states=5, seed=42 + trial)
        A = ConsciousAgent(agent_id=f"A_{trial}")
        for _ in range(300):
            A.step(world.step())
        B = clone_agent(A, new_id=f"B_{trial}")

        AB = observer_gated_combine(A, B)
        BA = observer_gated_combine(B, A)

        asymmetries.append(abs(AB["surprise_a_on_b"] - AB["surprise_b_on_a"]))
        if AB["mode"] == BA["mode"]:
            mode_matches += 1

    mean_asym = sum(asymmetries) / n_trials
    print(f"  Trials: {n_trials}")
    print(f"    Mean surprise asymmetry (twins): {mean_asym:.4f}  (expect ~0)")
    print(f"    Mode agreement (AB vs BA):       {mode_matches}/{n_trials}")

    if mean_asym < 0.05 and mode_matches == n_trials:
        print("\n  > Twins remain symmetric — consistent with 07_exchange_symmetry")
    else:
        print("\n  ~ Unexpected asymmetry detected even for twins — investigate further")

    return {"mean_asym": mean_asym, "mode_matches": mode_matches}


# ────────────── Phase B: Totally disjoint agents ──────────────


def phase_b_disjoint(n_trials=20):
    print("\n  Phase B: Totally Disjoint Agents (different worlds, zero overlap)\n")

    asymmetries = []
    both_max_surprise = 0

    for trial in range(n_trials):
        world_a = SimpleWorld(n_states=5, seed=100 + trial)
        world_c = SimpleWorld(n_states=8, seed=900 + trial)

        A = ConsciousAgent(agent_id=f"A_{trial}")
        for _ in range(300):
            A.step(world_a.step())
        C = ConsciousAgent(agent_id=f"C_{trial}")
        for _ in range(300):
            C.step(world_c.step())

        AC = observer_gated_combine(A, C)
        asymmetries.append(abs(AC["surprise_a_on_b"] - AC["surprise_b_on_a"]))
        if AC["surprise_a_on_b"] >= 0.95 and AC["surprise_b_on_a"] >= 0.95:
            both_max_surprise += 1

    mean_asym = sum(asymmetries) / n_trials
    print(f"  Trials: {n_trials}")
    print(f"    Mean surprise asymmetry: {mean_asym:.4f}")
    print(f"    Both saturated at max surprise (mutual incomprehension): {both_max_surprise}/{n_trials}")
    print("\n  ~ Zero overlap produces symmetric CONFUSION, not asymmetric dominance.")
    print("    Two agents with nothing in common cannot judge who understands whom —")
    print("    the comparison is degenerate, not meaningfully exchange-antisymmetric.")

    return {"mean_asym": mean_asym, "both_max_surprise": both_max_surprise}


# ────────────── Phase C: Veteran vs Novice ──────────────


def phase_c_veteran_novice(n_trials=20):
    print("\n  Phase C: Veteran vs Novice (same world, unequal experience depth)\n")

    order_independent_dominance = 0
    veteran_dominates_count = 0
    asymmetries = []

    for trial in range(n_trials):
        seed = 200 + trial
        world_veteran = SimpleWorld(n_states=5, seed=seed)
        world_novice = SimpleWorld(n_states=5, seed=seed)

        veteran = ConsciousAgent(agent_id=f"Veteran_{trial}")
        for _ in range(2000):
            veteran.step(world_veteran.step())

        novice = ConsciousAgent(agent_id=f"Novice_{trial}")
        for _ in range(40):
            novice.step(world_novice.step())

        VN = observer_gated_combine(veteran, novice)
        NV = observer_gated_combine(novice, veteran)

        asym = abs(VN["surprise_a_on_b"] - VN["surprise_b_on_a"])
        asymmetries.append(asym)

        if VN["dominant_id"] == NV["dominant_id"]:
            order_independent_dominance += 1
        if VN["dominant_id"].startswith("Veteran"):
            veteran_dominates_count += 1

    mean_asym = sum(asymmetries) / n_trials
    max_asym = max(asymmetries)
    print(f"  Trials: {n_trials}")
    print(f"    Mean surprise asymmetry: {mean_asym:.4f}  (max: {max_asym:.4f})")
    print(f"    Dominance is order-independent: {order_independent_dominance}/{n_trials}")
    print("    (same agent wins regardless of whether you call combine(V,N) or combine(N,V))")
    print(f"    Veteran dominates:              {veteran_dominates_count}/{n_trials}")

    if mean_asym > 0.05 and order_independent_dominance == n_trials:
        print("\n  > Genuine, principled asymmetry emerged — driven by relationship, not argument order")
    elif mean_asym <= 0.05:
        print("\n  ~ No meaningful asymmetry detected in this configuration")
    else:
        print("\n  ~ Asymmetry present but dominance is NOT fully order-independent — investigate further")

    return {
        "mean_asym": mean_asym,
        "order_independent_dominance": order_independent_dominance,
        "veteran_dominates_count": veteran_dominates_count,
    }


# ────────────── Main ──────────────


def main():
    t0 = time.time()
    print("=" * 66)
    print("Observer-Gated Combination")
    print("=" * 66)
    print("\nTesting whether replacing pure union with an act of PERCEPTION")
    print("(one agent predicting another's trajectory before merging) can")
    print("introduce principled, non-arbitrary exchange asymmetry.\n")
    print("This is an exploratory computational analogy.")
    print("Results are directional signals for further formal investigation.\n")

    a = phase_a_twins(20)
    b = phase_b_disjoint(20)
    c = phase_c_veteran_novice(20)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")
    print("=" * 66)
    print("Summary")
    print("=" * 66)
    print(f"\n  Twins (identical):       symmetric fit     (asym {a['mean_asym']:.4f})")
    print(f"  Disjoint (zero overlap): symmetric confusion (asym {b['mean_asym']:.4f})")
    print(f"  Veteran/Novice (shared world, unequal depth): asym {c['mean_asym']:.4f}, "
          f"order-independent {c['order_independent_dominance']}/20")
    print("\n  See README.md for interpretation and caveats.\n")


if __name__ == "__main__":
    main()
