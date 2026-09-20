/**
 * Does Spacetime Emerge in the Headset?
 *
 * Granting Hoffman's premise that space and time are an interface built from
 * conscious-agent dynamics, this experiment asks when an observer's experience
 * carries the structure of space (a dimension) and of time (an arrow).
 *
 * Ruler: the spectral dimension d_s. A random walk returns to its start with
 * probability R(t) ~ t^(-d_s/2); d_s = 1, 2, 3 on lattices of that dimension,
 * it has no plateau on networks without geometry (expanders), and it is the
 * probe used for emergent dimension in quantum-gravity models (causal
 * dynamical triangulations). Arrow of time: irreversibility, the normalised
 * net probability flux (0 = detailed balance, no arrow).
 *
 *   1. Calibration: the ruler on lattices of known dimension and on an expander.
 *   2. Reconstruction: an agent living in a hidden lattice world sees only
 *      opaque symbols (hashes). Does the kernel it learns from experience
 *      carry the world's dimension?
 *   3. Combination: conscious agents whose experience is a 1-D ring are
 *      combined with ⊗. How does dimension behave under combination?
 *   4. Interaction: three ring agents whose worlds are each other (agent i
 *      perceives agent i+1 and drifts toward it). Does interaction keep,
 *      create or destroy dimensions, and does an arrow of time appear?
 */
const { ConsciousAgent, WorldState, FormalConsciousAgent, MarkovKernel, markov, mulberry32 } = require('../../src/index');

/* ────────────── kernels ────────────── */

// Simple random walk on a d-dimensional torus of side L (state = mixed-radix index).
function torus(L, d) {
  const n = L ** d;
  const P = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let k = 0, stride = 1; k < d; k++, stride *= L) {
      const c = Math.floor(i / stride) % L;
      for (const s of [1, -1]) P[i][i + (((c + s + L) % L) - c) * stride] += 1 / (2 * d);
    }
  }
  return P;
}

// Random 4-regular multigraph (two random perfect matchings): no geometry.
function expander(n, seed) {
  const r = mulberry32(seed);
  const P = Array.from({ length: n }, () => new Float64Array(n));
  for (let rep = 0; rep < 2; rep++) {
    const p = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
    for (let i = 0; i < n; i++) { P[i][p[i]] += 0.25; P[p[i]][i] += 0.25; }
  }
  return P;
}

// A conscious agent whose experience space X is a ring of m states and whose
// perception moves it one step either way; its world has a single state.
function ringAgent(m) {
  const ring = torus(m, 1).map(r => Array.from(r));
  return new FormalConsciousAgent({
    X: [...Array(m).keys()].map(String), G: ['·'], W: ['·'],
    P: { '·': ring }, D: ring.map(() => [1]), A: { '·': [[1]] },
  });
}

// Three ring agents, each perceiving the next: agent i steps toward agent i+1
// with probability (1 + c)/2 (lazy: it stays put half the time).
function coupledRings(m, c) {
  const n = m ** 3;
  const Q = Array.from({ length: n }, () => new Float64Array(n));
  const step = (x, w) => {
    const toward = ((w - x + m) % m) === 0 ? 0 : (((w - x + m) % m) <= m / 2 ? 1 : -1);
    const up = 0.25 * (1 + c * toward), down = 0.25 * (1 - c * toward);
    return [[x, 0.5], [(x + 1) % m, up], [(x - 1 + m) % m, down]];
  };
  for (let i = 0; i < n; i++) {
    const x = [i % m, Math.floor(i / m) % m, Math.floor(i / (m * m))];
    const moves = x.map((xi, k) => step(xi, x[(k + 1) % 3]));
    for (const [a, pa] of moves[0]) for (const [b, pb] of moves[1]) for (const [d, pd] of moves[2]) {
      Q[i][a + m * b + m * m * d] += pa * pb * pd;
    }
  }
  return Q;
}

/* ────────────── measurements ────────────── */

const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : String(x));

function measure(P, window, options = {}) {
  const s = markov.spectralDimension(P, { tMin: window[0], tMax: window[1], ...options });
  return { d: s.dimension, local: s.local, saturation: s.saturation };
}

const localStr = (loc) => loc.map(l => `${l.t}:${f(l.d)}`).join(' ');

/* ────────────── 2. reconstruction through the headset ────────────── */

// The agent perceives the world through a headset: each lattice site is shown
// only as an opaque token. With aliasing > 0 some pairs of sites share a token.
function learnedKernel(L, d, { steps, seed, aliasing = 0 }) {
  const P = torus(L, d);
  const n = P.length;
  const r = mulberry32(seed);
  const token = [...Array(n).keys()].map(i => `q${i * 7919 % 100003}`);
  if (aliasing > 0) {
    const order = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    const pairs = Math.floor(aliasing * n / 2);
    for (let k = 0; k < pairs; k++) token[order[2 * k + 1]] = token[order[2 * k]];
  }
  let site = 0;
  const world = {
    step() {
      const u = r();
      let acc = 0;
      for (let j = 0; j < n; j++) { acc += P[site][j]; if (u < acc) { site = j; break; } }
      return WorldState.fromSequence('world', [token[site]]);
    },
  };
  const agent = new ConsciousAgent({ agentId: `observer_${d}d`, seed, world });
  agent.run(steps);
  const K = agent.toFormal().P;
  return { kernel: K, states: K.states.length, truth: P };
}

function main() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Does spacetime emerge in the headset?');
  console.log('='.repeat(78));

  console.log('\n  1. Calibrating the ruler (spectral dimension; local values at t = 2, 4, 8, ...)');
  const calib = [
    ['ring, 200 sites (d = 1)', torus(200, 1), [4, 64]],
    ['torus 30×30 (d = 2)', torus(30, 2), [4, 64]],
    ['torus 16×16×16 (d = 3)', torus(16, 3), [4, 32]],
    ['random 4-regular graph, 400 nodes', expander(400, 1), [2, 8]],
  ];
  for (const [name, P, w] of calib) {
    const m = measure(P, w);
    console.log(`    ${name.padEnd(36)} d_s = ${f(m.d)}   local ${localStr(m.local)}`);
  }
  console.log('    → lattices read 1, 2, 3; the expander has no plateau (its local dimension keeps rising).');

  console.log('\n  2. Reconstruction: an agent sees only opaque tokens from a hidden lattice');
  console.log('    (small worlds read below their dimension; what matters is learned vs true)');
  console.log(`    ${'hidden world'.padEnd(32)} ${'steps'.padEnd(7)} ${'states learned'.padEnd(15)} d_s(true)  d_s(learned)`);
  const recon = [];
  for (const [L, d, w, aliasing] of [[64, 1, [2, 16], 0], [12, 2, [2, 12], 0], [8, 3, [2, 8], 0], [12, 2, [2, 12], 0.3]]) {
    const steps = 60 * L ** d;
    const { kernel, states, truth } = learnedKernel(L, d, { steps, seed: 10 + d, aliasing });
    const dt = measure(truth, w).d, dl = measure(kernel.matrix, w).d;
    recon.push({ d, aliasing, dt, dl });
    const label = `${d}-D torus, side ${L}${aliasing ? `, ${aliasing * 100}% aliased` : ''}`;
    console.log(`    ${label.padEnd(32)} ${String(steps).padEnd(7)} ${String(states).padEnd(15)} ${f(dt).padEnd(10)} ${f(dl)}`);
  }

  console.log('\n  3. Combination: ring agents (X = ring of 10 experiences) combined with ⊗');
  const comb = [];
  let agent = ringAgent(10);
  for (let count = 1; count <= 3; count++) {
    if (count > 1) agent = FormalConsciousAgent.combine(agent, ringAgent(10));
    const Q = agent.jointKernel().matrix;
    const m = measure(Q, [2, 10]);
    comb.push({ count, d: m.d, states: Q.length });
    console.log(`    ${`${count} agent${count > 1 ? 's' : ''} combined`.padEnd(22)} ${String(Q.length).padEnd(6)} experiences   d_s = ${f(m.d)}   local ${localStr(m.local)}`);
  }
  const mixed = new MarkovKernel({ states: [...Array(10).keys()].map(String), matrix: torus(10, 1) })
    .tensor(new MarkovKernel({ states: [...Array(100).keys()].map(String), matrix: expander(100, 3) }));
  const mm = measure(mixed.matrix, [2, 10]);
  console.log(`    ${'ring ⊗ geometry-free'.padEnd(22)} ${String(mixed.states.length).padEnd(6)} experiences   d_s = ${f(mm.d)}   local ${localStr(mm.local)}`);

  console.log('\n  4. Interaction: three ring agents, each perceiving the next (coupling c)');
  console.log('    (returns measured from the stationary distribution: coupled agents drift toward each other,');
  console.log('     and returns from transient states would measure that drift rather than geometry)');
  console.log('    c      d_s     local                              irreversibility (arrow of time)');
  const inter = [];
  for (const c of [0, 0.3, 0.6, 0.9]) {
    const Q = coupledRings(10, c);
    const pi = markov.stationary(Q).pi;
    const m = measure(Q, [2, 10], { startDistribution: 'stationary', pi, maxStarts: 128, lazy: false });
    const irr = markov.irreversibility(Q, pi);
    inter.push({ c, d: m.d, irr, local: m.local });
    console.log(`    ${f(c, 1).padEnd(6)} ${f(m.d).padEnd(7)} ${localStr(m.local).padEnd(34)} ${f(irr, 3)}`);
  }

  console.log('\n' + '─'.repeat(78));
  console.log('Findings');
  console.log('─'.repeat(78));
  const worst = Math.max(...recon.filter(r => !r.aliasing).map(r => Math.abs(r.dt - r.dl)));
  const aliased = recon.find(r => r.aliasing);
  console.log(`  1. From opaque symbols alone, the agent's learned kernel carries its world's dimension`);
  console.log(`     (largest |d_s(learned) − d_s(true)| = ${f(worst)}). Aliasing ${aliased.aliasing * 100}% of the sites shifts it`);
  console.log(`     from ${f(aliased.dt)} to ${f(aliased.dl)}: a coarser headset distorts perceived space.`);
  console.log(`  2. Under ⊗, dimension adds: ${comb.map(c => `${c.count} → ${f(c.d)}`).join(', ')}.`);
  console.log('     This is exact for independent combination (return probabilities multiply), so three');
  console.log('     combined 1-D agents are a 3-D space. Combining with a geometry-free agent gives no plateau.');
  const c0 = inter[0], cMax = inter[inter.length - 1];
  console.log(`  3. Interaction ${cMax.d < c0.d - 0.3 ? 'binds dimensions' : 'changes dimension little'}: d_s ${inter.map(x => `${f(x.d)} (c = ${x.c})`).join(' → ')}.`);
  console.log('     Strongly coupled agents move as one: only their shared centre diffuses, a 1-D space.');
  console.log(`     An arrow of time appears with the cyclic coupling: irreversibility ${f(c0.irr, 3)} → ${f(cMax.irr, 3)}.`);
  console.log('\n  Interpretation: in this model, space in the headset is inherited, not created. The observer');
  console.log('  recovers whatever geometry the agent dynamics has; ⊗ adds dimensions exactly; interaction');
  console.log('  binds them; networks without geometric structure give none. Getting 3+1 dimensions therefore');
  console.log('  requires specific agent dynamics (three independent 1-D directions, weak coupling). Nothing');
  console.log('  here selects that structure, which is the open question for the "spacetime is a headset" view.');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

if (require.main === module) main();

module.exports = { torus, expander, ringAgent, coupledRings, learnedKernel };
