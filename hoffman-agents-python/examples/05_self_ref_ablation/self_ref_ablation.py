"""
Self-Reference Ablation: does the "I" lock track structure?

Three conditions, 8 seeded agents each, 1000 steps:

  A. structured world, lock ON    - a repeating 10-state cycle
  B. structured world, lock OFF   - lock_margin = 2, so dominance (<= 1) can never qualify
  C. structureless world, lock ON - a fresh random state from 10 each step

B is an ablation by construction: without the lock the agent only ever says
"wait", so B cannot fail. It shows that the lock gates expression, nothing more.
The informative comparison is A vs C: if the lock also fired in C, it would not
be tracking anything about the agent's experience.

The v3 rule needs evidence (>= 20 meta-transitions, i.e. >= 400 steps at the
default observation interval) before it can lock. 2.x behaviour is available
with math_version="legacy"; under it condition C locks too (at step 61).

Port of the Node example; prints the same numbers.
"""
import time

from conscious_agent import ConsciousAgent, ExperienceSpace, SelfTokenState, WorldState, mulberry32

N_AGENTS = 8
N_STEPS = 1000


def run_condition(name, lock_margin, structured):
    agents = [ConsciousAgent(agent_id=f"Agent_{i:02d}", seed=i + 1,
                             experience=ExperienceSpace(self_token=SelfTokenState(lock_margin=lock_margin)))
              for i in range(N_AGENTS)]
    world_rng = mulberry32(99)
    locked = set()
    non_trivial = loop = counted = 0

    for step in range(N_STEPS):
        state = step % 10 if structured else int(world_rng.random() * 10)
        ws = WorldState.from_sequence("world", [f"state_{state}"])
        for agent in agents:
            out = agent.step(ws)
            if out.i_locked:
                locked.add(agent.agent_id)
            if step >= N_STEPS - 100:
                counted += 1
                non_trivial += out.sequence_str != "wait"
                loop += out.loop_depth

    r = {"lock_rate": len(locked) / N_AGENTS, "non_trivial": non_trivial / counted, "loop": loop / counted}
    print(f"\n  ── {name} ──")
    print(f"    lock rate:          {r['lock_rate'] * 100:.0f}%")
    print(f"    non-\"wait\" output:  {r['non_trivial'] * 100:.1f}% of the last 100 steps")
    print(f"    mean loop score:    {r['loop']:.3f}")
    return r


def main():
    t0 = time.time()
    print("=" * 66)
    print('Self-reference ablation: does the "I" lock track structure?')
    print("=" * 66)

    a = run_condition("A. structured world, lock ON", 0.15, True)
    b = run_condition("B. structured world, lock OFF (by construction)", 2, True)
    c = run_condition("C. structureless world, lock ON", 0.15, False)

    print("\n" + "=" * 66)
    print("Summary")
    print("=" * 66)
    print(f"  lock rate:  A {a['lock_rate'] * 100:.0f}%   B {b['lock_rate'] * 100:.0f}%   C {c['lock_rate'] * 100:.0f}%")
    if a["lock_rate"] == 1 and c["lock_rate"] == 0:
        print("\n  The lock fires where the agent's experience has a stable attractor (A) and not where it has")
        print("  none (C): it tracks structure. B confirms only that the lock gates output, which is true by")
        print("  construction and says nothing about whether self-reference matters for anything else.")
    else:
        print(f"\n  Mixed result: A {a['lock_rate'] * 100:.0f}% vs C {c['lock_rate'] * 100:.0f}%.")
    print(f"\n  Done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
