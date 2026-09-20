"""
Ergodic Diagnostics: when does "I" lock, and why?

Three worlds, same agent:
  noise     - i.i.d.-like random transitions: no experiential attractor
  attractor - the agent mostly stays "home": one dominant attractor
  switch    - attractor for 1500 steps, then noise: lock, then unlock

Prints the ergodic analysis of the meta-state chain, the lock criteria and
lock/unlock events, the exact analysis of the decision kernel D, and a
FormalConsciousAgent whose joint kernel Q is checked against simulation.

Run with --legacy to see the 2.x rule lock in every world at step 61.
"""
import sys

from conscious_agent import ConsciousAgent, FormalConsciousAgent, SimpleWorld, WorldState, mulberry32

MATH_VERSION = "legacy" if "--legacy" in sys.argv else "v3"


class HomeWorld:
    def __init__(self, seed, p_home=0.9, n=20):
        self.r, self.p, self.n = mulberry32(seed + 999), p_home, n

    def step(self):
        return WorldState.from_sequence("world", ["home" if self.r.random() < self.p else f"s{int(self.r.random() * self.n)}"])


class SwitchingWorld:
    def __init__(self, seed, at):
        self.home, self.noise, self.t, self.at = HomeWorld(seed, 0.95), SimpleWorld(n_states=50, seed=seed), 0, at

    def step(self):
        self.t += 1
        return self.home.step() if self.t < self.at else self.noise.step()


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def report(name, world, steps):
    agent = ConsciousAgent(agent_id=name, seed=7, world=world, math_version=MATH_VERSION)
    events = []
    for _ in range(steps):
        out = agent.step()
        if out.interrupt:
            events.append(f"{out.interrupt['event']}@step {out.step}")
        if MATH_VERSION == "legacy" and out.i_locked and not events:
            events.append(f"lock@step {out.step}")
    s = agent.ergodic_stats()
    m = s["meta"]
    print(f"\n  ── {name} ({steps} steps, math_version={MATH_VERSION}) ──")
    print(f"    meta-states observed      {agent.experience.meta_trie.registry_size}")
    print(f"    recurrent class size      {m['class_size']}   (transitions in class: {m['n_transitions']:.0f})")
    print(f"    dominant pi               {fmt(m['dominant_prob'])}   dominance pi_max - 1/n: {fmt(m['dominance'])}")
    print(f"    period / aperiodic        {m['period'] if m['period'] is not None else '-'} / {m['aperiodic']}")
    print(f"    |lambda_2|, t_mix         {fmt(m['lambda2'])}, {fmt(m['mixing_time'], 1)} observations")
    print(f"    occupancy                 {fmt(m['occupancy'], 2)}")
    if s["lock"]["criteria"]:
        print("    lock criteria             " + "  ".join(f"{k}:{'yes' if v else 'no'}" for k, v in s["lock"]["criteria"].items()))
    print(f"    events                    {', '.join(events) if events else 'none'}")
    print(f"    locked now                {agent.is_i_locked}")
    return agent


def main():
    print("=" * 66)
    print('Ergodic Diagnostics of the "I" attractor')
    print("=" * 66)

    report("noise", SimpleWorld(n_states=50, seed=3), 3000)
    locked = report("attractor", HomeWorld(3), 3000)
    report("switch", SwitchingWorld(3, 1500), 6000)

    D = locked.decision_kernel.diagnostics()
    print("\n  ── decision kernel D (exact) ──")
    print("    stationary  " + "  ".join(f"{k}={fmt(v)}" for k, v in D["stationary"].items()))
    print(f"    ergodic={D['ergodic']}  period={D['period']}  |lambda_2|={fmt(D['lambda2'])}  t_mix<={fmt(D['mixing_time'], 1)}")

    c = FormalConsciousAgent(
        ["calm", "alert"], ["stay", "move"], ["left", "right"],
        {"left": [[0.9, 0.1], [0.6, 0.4]], "right": [[0.3, 0.7], [0.1, 0.9]]},
        [[0.8, 0.2], [0.2, 0.8]],
        {"stay": [[1, 0], [0, 1]], "move": [[0, 1], [1, 0]]},
        rng=mulberry32(1),
    )
    q = c.diagnostics()
    counts, w, n = {}, "left", 100000
    for _ in range(n):
        o = c.step(w)
        w = o["w"]
        counts[f"{o['x']}|{w}"] = counts.get(f"{o['x']}|{w}", 0) + 1
    print("\n  ── FormalConsciousAgent (X, G, P, D, A, N): joint kernel Q on X x W ──")
    print(f"    ergodic={q['ergodic']}  |lambda_2|={fmt(q['lambda2'])}")
    for k, p in q["stationary"].items():
        print(f"    pi({k:<11}) exact {fmt(p)}   simulated {fmt(counts.get(k, 0) / n)}")
    print()


if __name__ == "__main__":
    main()
