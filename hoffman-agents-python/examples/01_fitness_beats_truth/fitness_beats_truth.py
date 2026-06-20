"""
Fitness Beats Truth — Hoffman's Interface Theory, Experiment 1

Two agents experience the same hidden Markov world through different
interfaces. The "Interface" agent sees a compressed grouping (5 groups
of 4 states), while the "Truth" agent sees all 20 raw states.

Hoffman's claim: the Interface agent predicts BETTER because it discards
irrelevant within-group variation. Perception hides truth to optimize
for fitness.

Expected result:
  Interface CA improvement >> Truth CA improvement
"""
from conscious_agent import ConsciousAgent, WorldBuilder, WorldState
import numpy as np


def build_hidden_markov_world(n_groups=5, states_per_group=4):
    """Build a hidden Markov world with group structure.

    States within a group have high transition probability to each other.
    States across groups have low transition probability.
    """
    n_states = n_groups * states_per_group
    P = np.full((n_states, n_states), 0.01)
    for g in range(n_groups):
        start = g * states_per_group
        end = start + states_per_group
        for i in range(start, end):
            P[i, start:end] = 0.2
            P[i, i] = 0.3
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums

    labels = {i: f"s{i}" for i in range(n_states)}
    sequence = [np.random.choice(n_states, p=P[s]) for s in np.random.choice(n_states, size=2000)]
    return n_states, P, sequence, labels


def build_interface_world(states, n_groups=5, states_per_group=4):
    """Map N raw states into M groups — the interface projection."""
    interface_sequence = [s // states_per_group for s in states]
    unique = sorted(set(interface_sequence))
    n_states = len(unique)
    P = np.zeros((n_states, n_states))
    for i in range(len(interface_sequence) - 1):
        P[interface_sequence[i], interface_sequence[i + 1]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    P = P / row_sums

    labels = {i: f"group_{i}" for i in range(n_states)}
    return interface_sequence, P, labels


def run_experiment():
    np.random.seed(42)
    print("=" * 60)
    print("Experiment 1: Fitness Beats Truth")
    print("=" * 60)

    # Build hidden world: 5 groups × 4 states = 20 states
    n_groups, states_per_group = 5, 4
    _, _, seq, labels = build_hidden_markov_world(n_groups, states_per_group)

    # Build Interface world (5 groups)
    iface_seq, iface_P, iface_labels = build_interface_world(seq, n_groups, states_per_group)

    # Build Truth world (20 raw states)
    truth_seq = list(seq)
    unique = sorted(set(truth_seq))
    truth_P = np.zeros((len(unique), len(unique)))
    for i in range(len(truth_seq) - 1):
        truth_P[truth_seq[i], truth_seq[i + 1]] += 1
    row_sums = truth_P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    truth_P = truth_P / row_sums
    truth_labels = {i: f"s{i}" for i in unique}

    # Split into train/test
    split = int(len(seq) * 0.8)

    def create_agent_and_train(state_sequence, name):
        agent = ConsciousAgent(agent_id=name)

        # Train: store hash IDs, not raw state IDs
        prev_hash = None
        for s in state_sequence[:split]:
            ws = WorldState.from_sequence("world", [f"s{s}"])
            agent.step(ws)
            prev_hash = ws.get_state_id()

        # Evaluate prediction error on test set using hash IDs
        errors = []
        for s in state_sequence[split:]:
            ws = WorldState.from_sequence("world", [f"s{s}"])
            curr_hash = ws.get_state_id()
            if prev_hash is not None:
                pred = agent.experience.trie.predict_next([prev_hash])
                err = 0.0 if pred == curr_hash else 1.0
                errors.append(err)
            agent.step(ws)
            prev_hash = curr_hash

        mean_err = np.mean(errors) if errors else 1.0
        improvement = (1.0 - mean_err) * 100
        return agent, mean_err, improvement

    iface_agent, iface_err, iface_imp = create_agent_and_train(
        iface_seq, "Interface"
    )
    truth_agent, truth_err, truth_imp = create_agent_and_train(
        truth_seq, "Truth"
    )

    print(f"\n  {'Agent':<20s} {'Pred Error':<15s} {'Improvement':<15s} {'I-Locked':<10s}")
    print(f"  {'-'*60}")
    print(f"  {'Interface (5 groups)':<20s} {iface_err:.3f}         {iface_imp:+.1f}%         {iface_agent.is_i_locked}")
    print(f"  {'Truth (20 states)':<20s} {truth_err:.3f}         {truth_imp:+.1f}%         {truth_agent.is_i_locked}")

    if iface_imp > truth_imp:
        print(f"\n  ✓ RESULT: Interface dominates — less information → better prediction")
        print(f"    This is Hoffman's core claim: perception hides truth to optimize for fitness.")
    else:
        print(f"\n  ✗ RESULT: Expected Interface > Truth, got opposite")

    return iface_imp, truth_imp


if __name__ == "__main__":
    run_experiment()
