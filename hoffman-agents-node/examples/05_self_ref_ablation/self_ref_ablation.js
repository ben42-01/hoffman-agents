/**
 * Self-Reference Ablation: does the "I" lock track structure?
 *
 * Three conditions, 8 seeded agents each, 1000 steps:
 *
 *   A. structured world, lock ON    - a repeating 10-state cycle
 *   B. structured world, lock OFF   - lockMargin = 2, so dominance (<= 1) can never qualify
 *   C. structureless world, lock ON - a fresh random state from 10 each step
 *
 * B is an ablation by construction: without the lock the agent only ever says
 * "wait", so B cannot fail. It shows that the lock gates expression, nothing more.
 * The informative comparison is A vs C: if the lock also fired in C, it would
 * not be tracking anything about the agent's experience.
 *
 * The v3 rule needs evidence (>= 20 meta-transitions, i.e. >= 400 steps at the
 * default observation interval) before it can lock. 2.x behaviour is available
 * with { mathVersion: 'legacy' }; under it condition C locks too (at step 61).
 */
const { ConsciousAgent, WorldState, SelfTokenState, ExperienceSpace, mulberry32 } = require('../../src/index');

const N_AGENTS = 8;
const N_STEPS = 1000;

function runCondition(name, { lockMargin, structured }) {
  const agents = Array.from({ length: N_AGENTS }, (_, i) => new ConsciousAgent({
    agentId: `Agent_${String(i).padStart(2, '0')}`,
    seed: i + 1,
    experience: new ExperienceSpace({ selfToken: new SelfTokenState({ lockMargin }) }),
  }));
  const worldRng = mulberry32(99);
  const locked = new Set();
  let nonTrivial = 0, loop = 0, counted = 0;

  for (let step = 0; step < N_STEPS; step++) {
    const state = structured ? step % 10 : Math.floor(worldRng() * 10);
    const ws = WorldState.fromSequence('world', [`state_${state}`]);
    for (const agent of agents) {
      const out = agent.step(ws);
      if (out.iLocked) locked.add(agent.agentId);
      if (step >= N_STEPS - 100) {
        counted++;
        if (out.sequenceStr !== 'wait') nonTrivial++;
        loop += out.loopDepth;
      }
    }
  }

  const r = { name, lockRate: locked.size / N_AGENTS, nonTrivial: nonTrivial / counted, loop: loop / counted };
  console.log(`\n  ── ${name} ──`);
  console.log(`    lock rate:          ${(r.lockRate * 100).toFixed(0)}%`);
  console.log(`    non-"wait" output:  ${(r.nonTrivial * 100).toFixed(1)}% of the last 100 steps`);
  console.log(`    mean loop score:    ${r.loop.toFixed(3)}`);
  return r;
}

function main() {
  const t0 = Date.now();
  console.log('='.repeat(66));
  console.log('Self-reference ablation: does the "I" lock track structure?');
  console.log('='.repeat(66));

  const A = runCondition('A. structured world, lock ON', { lockMargin: 0.15, structured: true });
  const B = runCondition('B. structured world, lock OFF (by construction)', { lockMargin: 2, structured: true });
  const C = runCondition('C. structureless world, lock ON', { lockMargin: 0.15, structured: false });

  console.log('\n' + '='.repeat(66));
  console.log('Summary');
  console.log('='.repeat(66));
  console.log(`  lock rate:  A ${(A.lockRate * 100).toFixed(0)}%   B ${(B.lockRate * 100).toFixed(0)}%   C ${(C.lockRate * 100).toFixed(0)}%`);
  if (A.lockRate === 1 && C.lockRate === 0) {
    console.log('\n  The lock fires where the agent\'s experience has a stable attractor (A) and not where it has');
    console.log('  none (C): it tracks structure. B confirms only that the lock gates output, which is true by');
    console.log('  construction and says nothing about whether self-reference matters for anything else.');
  } else {
    console.log(`\n  Mixed result: A ${(A.lockRate * 100).toFixed(0)}% vs C ${(C.lockRate * 100).toFixed(0)}%.`);
  }
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

main();
