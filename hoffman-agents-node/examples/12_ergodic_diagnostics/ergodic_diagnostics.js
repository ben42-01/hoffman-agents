/**
 * Ergodic Diagnostics: when does "I" lock, and why?
 *
 * Three worlds, same agent:
 *   noise     - i.i.d.-like random transitions: no experiential attractor
 *   attractor - the agent mostly stays "home": one dominant attractor
 *   switch    - attractor for 1500 steps, then noise: lock, then unlock
 *
 * For each run it prints the ergodic analysis of the agent's meta-state chain
 * (recurrent class size, stationary dominance, period, |lambda_2|, mixing
 * time), the lock criteria, and the lock/unlock events. It also prints the
 * exact analysis of the decision kernel D and a FormalConsciousAgent whose
 * joint kernel Q can be checked against simulation.
 *
 * Try { mathVersion: 'legacy' } to see the 2.x rule lock in every world at step 61.
 */
const { ConsciousAgent, SimpleWorld, WorldState, FormalConsciousAgent, mulberry32 } = require('../../src/index');

const mathVersion = process.argv.includes('--legacy') ? 'legacy' : 'v3';

function homeWorld(seed, pHome = 0.9, n = 20) {
  const r = mulberry32(seed + 999);
  return { step: () => WorldState.fromSequence('world', [r() < pHome ? 'home' : `s${Math.floor(r() * n)}`]) };
}

function switchingWorld(seed, at) {
  const home = homeWorld(seed, 0.95), noise = new SimpleWorld({ nStates: 50, seed });
  let t = 0;
  return { step: () => (++t < at ? home.step() : noise.step()) };
}

const fmt = (x, d = 3) => (x === null || x === undefined ? '-' : Number.isFinite(x) ? x.toFixed(d) : String(x));

function report(name, world, steps) {
  const agent = new ConsciousAgent({ agentId: name, seed: 7, world, mathVersion });
  const events = [];
  for (let i = 0; i < steps; i++) {
    const out = agent.step();
    if (out.interrupt) events.push(`${out.interrupt.event}@step ${out.step}`);
    if (mathVersion === 'legacy' && out.iLocked && events.length === 0) events.push(`lock@step ${out.step}`);
  }
  const s = agent.ergodicStats();
  const m = s.meta;
  console.log(`\n  ── ${name} (${steps} steps, mathVersion=${mathVersion}) ──`);
  console.log(`    meta-states observed      ${agent.experience.metaTrie.registrySize}`);
  console.log(`    recurrent class size      ${m.classSize}   (transitions in class: ${m.nTransitions})`);
  console.log(`    dominant pi               ${fmt(m.dominantProb)}   dominance pi_max - 1/n: ${fmt(m.dominance)}`);
  console.log(`    period / aperiodic        ${m.period ?? '-'} / ${m.aperiodic}`);
  console.log(`    |lambda_2|, t_mix         ${fmt(m.lambda2)}, ${fmt(m.mixingTime, 1)} observations`);
  console.log(`    occupancy                 ${fmt(m.occupancy, 2)}`);
  if (s.lock.criteria) {
    console.log(`    lock criteria             ${Object.entries(s.lock.criteria).map(([k, v]) => `${k}:${v ? 'yes' : 'no'}`).join('  ')}`);
  }
  console.log(`    events                    ${events.length ? events.join(', ') : 'none'}`);
  console.log(`    locked now                ${agent.isILocked}`);
  return agent;
}

function main() {
  console.log('='.repeat(66));
  console.log('Ergodic Diagnostics of the "I" attractor');
  console.log('='.repeat(66));

  report('noise', new SimpleWorld({ nStates: 50, seed: 3 }), 3000);
  const locked = report('attractor', homeWorld(3), 3000);
  report('switch', switchingWorld(3, 1500), 6000);

  const D = locked.decisionKernel.diagnostics();
  console.log('\n  ── decision kernel D (exact) ──');
  console.log(`    stationary  ${Object.entries(D.stationary).map(([k, v]) => `${k}=${fmt(v)}`).join('  ')}`);
  console.log(`    ergodic=${D.ergodic}  period=${D.period}  |lambda_2|=${fmt(D.lambda2)}  t_mix<=${fmt(D.mixingTime, 1)}`);

  const c = new FormalConsciousAgent({
    X: ['calm', 'alert'], G: ['stay', 'move'], W: ['left', 'right'],
    P: { left: [[0.9, 0.1], [0.6, 0.4]], right: [[0.3, 0.7], [0.1, 0.9]] },
    D: [[0.8, 0.2], [0.2, 0.8]],
    A: { stay: [[1, 0], [0, 1]], move: [[0, 1], [1, 0]] },
    rng: mulberry32(1),
  });
  const q = c.diagnostics();
  const counts = {};
  let w = 'left';
  const n = 100000;
  for (let i = 0; i < n; i++) { const o = c.step(w); w = o.w; counts[`${o.x}|${w}`] = (counts[`${o.x}|${w}`] || 0) + 1; }
  console.log('\n  ── FormalConsciousAgent (X, G, P, D, A, N): joint kernel Q on X × W ──');
  console.log(`    ergodic=${q.ergodic}  |lambda_2|=${fmt(q.lambda2)}`);
  for (const [k, p] of Object.entries(q.stationary)) {
    console.log(`    pi(${k.padEnd(11)}) exact ${fmt(p)}   simulated ${fmt((counts[k] || 0) / n)}`);
  }
  console.log();
}

main();
