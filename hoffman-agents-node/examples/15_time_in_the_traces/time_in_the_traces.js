/**
 * Time in the Traces
 *
 * Hoffman's trace logic: an observer who can only see part of a network of
 * conscious agents (its window S) does not see the network's dynamics. It
 * experiences the TRACE CHAIN on S:
 *
 *     P_S = P_SS + P_SC (I - P_CC)^(-1) P_CS
 *
 * (stay in the window, or leave it, spend any number of hidden steps outside,
 * and come back). This experiment asks what TIME looks like from inside a trace.
 *
 *   1. Every observer experiences its own Markov chain. We simulate the network,
 *      record only what each observer registers, and compare with the formula.
 *      Nested observers are consistent: a trace of a trace is the direct trace.
 *   2. Proper time. An observer's clock ticks only when the network is inside
 *      its window, so observers age at different rates: pi(S) per network step
 *      (Kac's lemma). Nested observers agree about each other's clocks.
 *   3. The arrow of time is observer-dependent.
 *      a. A reversible network (no arrow in the dynamics): every observer's
 *         uncertainty H(X_n | X_0) still grows. The arrow appears from projection,
 *         as Hoffman, Prakash & Prentner (2023) prove.
 *      b. An irreversible network: windows see different amounts of arrow. A
 *         window of two states never sees one (every two-state chain satisfies
 *         detailed balance), so perceiving an arrow of time needs >= 3 states.
 *
 * Not tested here: Hoffman's conjecture that Minkowski spacetime (Lorentz time
 * dilation) emerges from traces of n-cycle chains as n -> infinity. That needs
 * the exact construction from the 2025-26 trace-logic papers.
 */
const { FormalConsciousAgent, markov, mulberry32 } = require('../../src/index');
const trace = require('../../src/math/trace');

const STEPS = 200000;

/* ────────────── networks ────────────── */

function randomRow(n, r, i = null, stickiness = 0) {
  const u = Array.from({ length: n }, () => r() + 1e-3);
  const s = u.reduce((a, b) => a + b, 0);
  return u.map((v, j) => (1 - stickiness) * v / s + (j === i ? stickiness : 0));
}

function formalAgent(r, stickiness) {
  const X = ['calm', 'alert'], G = ['stay', 'move'], W = ['L', 'R'];
  return new FormalConsciousAgent({
    X, G, W,
    P: Object.fromEntries(W.map(w => [w, X.map((_, i) => randomRow(2, r, i, stickiness))])),
    D: X.map(() => randomRow(2, r)),
    A: Object.fromEntries(G.map(g => [g, W.map((_, i) => randomRow(2, r, i, stickiness))])),
    rng: r,
  });
}

// Two conscious agents combined, acting on a shared world: 8 joint states "x1|x2|w".
function agentNetwork(seed, stickiness = 0.3) {
  const r = mulberry32(seed);
  const K = FormalConsciousAgent.combine(formalAgent(r, stickiness), formalAgent(r, stickiness)).jointKernel();
  return { states: K.states, P: K.matrix };
}

// A reversible network: random symmetric conductances (detailed balance holds).
function reversibleNetwork(n, seed) {
  const r = mulberry32(seed);
  const Wt = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) Wt[i][j] = Wt[j][i] = r() < 0.5 ? r() : 0.02;
  return Wt.map(row => { const s = row.reduce((a, b) => a + b, 0); return Float64Array.from(row, v => v / s); });
}

/* ────────────── observers ────────────── */

function windowOf(states, pred) {
  return states.map((s, i) => [s, i]).filter(([s]) => pred(s.split('|'))).map(([, i]) => i);
}

function simulate(P, steps, seed, windows) {
  const r = mulberry32(seed);
  const n = P.length;
  const inWin = windows.map(S => { const m = new Map(); S.forEach((s, k) => m.set(s, k)); return m; });
  const counts = windows.map(S => S.map(() => new Float64Array(S.length)));
  const last = windows.map(() => null);
  const ticks = windows.map(() => 0);
  const gaps = windows.map(() => []);
  const lastTick = windows.map(() => null);
  let x = 0;
  for (let t = 0; t < steps; t++) {
    const u = r();
    let acc = 0;
    for (let j = 0; j < n; j++) { acc += P[x][j]; if (u < acc) { x = j; break; } }
    windows.forEach((_, w) => {
      const k = inWin[w].get(x);
      if (k === undefined) return;
      ticks[w]++;
      if (last[w] !== null) counts[w][last[w]][k]++;
      if (lastTick[w] !== null) gaps[w].push(t - lastTick[w]);
      last[w] = k; lastTick[w] = t;
    });
  }
  return windows.map((_, w) => ({ counts: counts[w], ticks: ticks[w], gaps: gaps[w] }));
}

const f = (x, d = 3) => x.toFixed(d);
const tiny = (x) => (x < 1e-9 ? 'machine precision' : x.toExponential(1));
const maxAbsDiff = (A, B) => Math.max(...A.map((row, i) => Math.max(...Array.from(row, (v, j) => Math.abs(v - B[i][j])))));

function main() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Time in the traces');
  console.log('='.repeat(78));

  const net = agentNetwork(15);
  const { P, states } = net;
  const pi = markov.stationary(P).pi;
  const observers = [
    { name: 'O_world  (sees only world = L)', S: windowOf(states, s => s[2] === 'L') },
    { name: 'O_agent  (sees only agent 1 alert)', S: windowOf(states, s => s[0] === 'alert') },
    { name: 'O_both   (both conditions)', S: windowOf(states, s => s[0] === 'alert' && s[2] === 'L') },
  ];
  console.log(`\n  Network: two combined conscious agents on a shared world, ${states.length} joint states.`);
  console.log(`  Simulated for ${STEPS.toLocaleString()} network steps; each observer records only its window.`);

  console.log('\n  1. Each observer experiences its own Markov chain (the trace)');
  console.log(`    ${'observer'.padEnd(38)} states  max |empirical − trace formula|`);
  const sims = simulate(P, STEPS, 1, observers.map(o => o.S));
  observers.forEach((o, w) => {
    o.T = trace.traceChain(P, o.S);
    const emp = markov.normalizeRows(sims[w].counts).P;
    o.err = maxAbsDiff(emp, o.T);
    console.log(`    ${o.name.padEnd(38)} ${String(o.S.length).padEnd(7)} ${f(o.err)}`);
  });
  const inner = observers[2].S.map(s => observers[0].S.indexOf(s));
  const viaWorld = trace.traceChain(observers[0].T, inner);
  const nestedErr = maxAbsDiff(viaWorld, observers[2].T);
  console.log(`    O_both as a trace of O_world's experience vs directly: max difference ${tiny(nestedErr)}`);

  console.log('\n  2. Proper time: each observer ages at its own rate');
  console.log(`    ${'observer'.padEnd(38)} ticks per network step   mean gap   Kac 1/π(S)   gap spread (sd)`);
  observers.forEach((o, w) => {
    const rate = trace.clockRate(pi, o.S);
    const measured = sims[w].ticks / STEPS;
    const g = sims[w].gaps;
    const mean = g.reduce((a, b) => a + b, 0) / g.length;
    const sd = Math.sqrt(g.reduce((a, b) => a + (b - mean) ** 2, 0) / g.length);
    o.rate = rate;
    o.kacErr = Math.abs(trace.meanReturnTime(P, o.S, pi) - 1 / rate);
    console.log(`    ${o.name.padEnd(38)} ${f(measured)} (π(S) = ${f(rate)})   ${f(mean, 2).padEnd(8)} ${f(1 / rate, 2).padEnd(12)} ${f(sd, 2)}`);
  });
  const [ow, , ob] = observers;
  const piWorldTrace = markov.stationary(ow.T).pi;
  const bothInWorldTime = inner.reduce((s, k) => s + piWorldTrace[k], 0);
  console.log(`    O_both's clock measured in O_world's own time: ${f(bothInWorldTime)} ticks per O_world tick;`);
  console.log(`    from the network: π(O_both)/π(O_world) = ${f(ob.rate / ow.rate)}. Nested clocks agree exactly.`);
  console.log('    Between two of its own moments an observer cannot tell how much hidden time passed: the gaps');
  console.log('    vary (sd above), but every observer experiences one step per event.');

  console.log('\n  3a. A reversible network: no arrow of time in the dynamics');
  const R = reversibleNetwork(8, 4);
  const piR = markov.stationary(R).pi;
  const SR = [0, 1, 2];
  const TR = trace.traceChain(R, SR);
  const piTR = markov.stationary(TR).pi;
  const hNet = trace.conditionalEntropyProfile(R, piR, 5);
  const hObs = trace.conditionalEntropyProfile(TR, piTR, 5);
  console.log(`    irreversibility: network ${f(markov.irreversibility(R, piR))}, observer's trace ${f(markov.irreversibility(TR, piTR))}`);
  console.log(`    H(X_n) stays ${f(markov.entropy(piR))} nats (network) and ${f(markov.entropy(piTR))} (observer) at every n, but`);
  console.log(`    H(X_n | X_0), network:  ${hNet.map(v => f(v)).join('  ')}`);
  console.log(`    H(X_n | X_0), observer: ${hObs.map(v => f(v)).join('  ')}`);
  const monotone = (h) => h.every((v, i) => i === 0 || v >= h[i - 1] - 1e-12);
  console.log(`    → uncertainty about the future grows (${monotone(hNet) && monotone(hObs) ? 'monotone in both' : 'NOT monotone'}), with no arrow in the dynamics.`);

  console.log('\n  3b. An irreversible network: is the arrow the same for every observer?');
  const irrNet = markov.irreversibility(P, pi);
  console.log(`    network irreversibility: ${f(irrNet)}`);
  for (const o of observers) {
    const piT = markov.stationary(o.T).pi;
    o.irr = markov.irreversibility(o.T, piT);
    console.log(`    ${o.name.padEnd(38)} ${f(o.irr)}${o.S.length === 2 ? '  (two states: always 0)' : ''}`);
  }
  const cycle = [[0, 1, 0], [0, 0, 1], [1, 0, 0]];
  const cycleTrace = trace.traceChain(cycle, [0, 1]);
  console.log(`    one-way 3-cycle (irreversibility ${f(markov.irreversibility(cycle))}), observer seeing 2 of 3 states: ` +
    `trace [[${Array.from(cycleTrace[0]).join(', ')}], [${Array.from(cycleTrace[1]).join(', ')}]], irreversibility ${f(markov.irreversibility(cycleTrace))}`);

  console.log('\n' + '─'.repeat(78));
  console.log('Findings');
  console.log('─'.repeat(78));
  const worst = Math.max(...observers.map(o => o.err));
  console.log(`  1. Observers experience exactly the trace chain of their window (simulation vs formula within`);
  console.log(`     ${f(worst)} after ${STEPS.toLocaleString()} steps), and nested observers are consistent (${tiny(nestedErr)}).`);
  console.log('  2. Time is per-observer: each observer\'s clock runs at π(S) ticks per network step (Kac\'s');
  console.log(`     formula confirmed to ${tiny(Math.max(...observers.map(o => o.kacErr)))}), clocks of nested observers compose exactly,`);
  console.log('     and no observer can perceive the hidden time between its own moments.');
  console.log('  3. The arrow of time depends on the observer. With reversible dynamics, every observer\'s');
  console.log('     uncertainty about its future still grows: the entropic arrow comes from observing, not from');
  console.log('     the dynamics. With irreversible dynamics, windows see different amounts of arrow');
  console.log(`     (network ${f(irrNet)}; observers ${observers.map(o => f(o.irr)).join(', ')}). An observer that sees only two`);
  console.log('     states never perceives an arrow, because every two-state chain satisfies detailed balance: even a');
  console.log('     one-way cycle looks time-symmetric through a two-state window. An arrow needs a window of >= 3 states.');
  console.log('\n  What this does not show: that these per-observer clocks obey Einstein\'s time dilation. That is');
  console.log('  Hoffman\'s open conjecture (Minkowski spacetime from traces of n-cycles as n → ∞), and it is the');
  console.log('  natural next experiment once the exact construction is available.');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

if (require.main === module) main();

module.exports = { agentNetwork, reversibleNetwork, windowOf, simulate };
