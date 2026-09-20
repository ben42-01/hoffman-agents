/**
 * Quantum Agents
 *
 * Experiment 13 found that classical conscious-agent networks behind a headset
 * give CHSH = 2 + 2δ, anywhere from 2 up to the PR box's 4, and signal unless an
 * extra rule forbids it. Nature stops at Tsirelson's bound 2√2 ≈ 2.83 and never
 * signals. Here the agents' kernels carry complex amplitudes instead:
 *
 *   experiences  -> density matrices (qubits)
 *   P, A         -> quantum channels (Kraus operators)
 *   D            -> measurements (Born rule)
 *   combination  -> tensor product, plus an interaction unitary
 *
 * Sharp question: does combining quantum agents reproduce the quantum limit
 * 2√2 exactly, and what does the work?
 *
 *   1. Markov agents are quantum agents that have fully decohered.
 *   2. Combination without interaction: CHSH <= 2.
 *   3. Combination with interaction: the maximum, random interactions,
 *      no-signalling, and the same Bell classifier as experiment 13.
 *   4. Network distance and decoherence: back to the classical bound.
 *   5. Interference: pairwise (I2 ≠ 0) but never three-way (I3 = 0).
 */
const { quantum: q, bell, markov, mulberry32 } = require('../../src/index');

const TSIRELSON = 2 * Math.SQRT2;
const f = (x, d = 3) => x.toFixed(d);
const tiny = (x) => (x < 1e-9 ? 'machine precision' : x.toExponential(1));

function randomStochastic(n, r) {
  return Array.from({ length: n }, () => {
    const u = Array.from({ length: n }, () => r() + 0.05);
    const s = u.reduce((a, b) => a + b, 0);
    return u.map(v => v / s);
  });
}

// Random pure qubit state (cos θ/2, e^{iφ} sin θ/2) as [re, im].
function randomQubit(r) {
  const th = Math.PI * r(), ph = 2 * Math.PI * r();
  return [[Math.cos(th / 2), Math.cos(ph) * Math.sin(th / 2)], [0, Math.sin(ph) * Math.sin(th / 2)]];
}

// Product of two single-qubit amplitude vectors, as a density matrix.
function productState(a, b) {
  const re = [], im = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    re.push(a[0][i] * b[0][j] - a[1][i] * b[1][j]);
    im.push(a[0][i] * b[1][j] + a[1][i] * b[0][j]);
  }
  return q.densityFromState(re, im);
}

const ZERO2 = q.densityFromState([1, 0, 0, 0]);
const XX = q.kron(q.PAULI.X, q.PAULI.X);
const interact = (rho, t) => q.conjugate(rho, q.pauliRotation(XX, t));
const local = (rho, kraus) => q.applyChannel(q.applyChannel(rho, q.onQubit(kraus, 0, 2)), q.onQubit(kraus, 1, 2));

// Measure with the optimal settings and run the measured statistics through
// experiment 13's Bell classifier.
function measured(rho) {
  const s = q.optimalSettings(rho);
  const p = q.behaviour(rho, s.alice, s.bob);
  return { chsh: bell.chsh(bell.correlators(p)), cls: bell.classify(p, 1e-9), signalling: bell.signalling(p) };
}

/* ────────────── 1. Markov agents inside quantum agents ────────────── */

function part1() {
  console.log('\n  1. Markov agents are quantum agents that have fully decohered');
  const r = mulberry32(1);
  let actErr = 0, krausErr = 0, coherence = 0;
  for (let t = 0; t < 50; t++) {
    const n = t % 2 ? 4 : 2;
    const P = randomStochastic(n, r);
    const K = q.markovChannel(P);
    krausErr = Math.max(krausErr, q.krausDeviation(K));
    const pn = randomStochastic(n, r)[0];
    const rho = q.cmat(pn.map((v, i) => pn.map((_, j) => (i === j ? v : 0))));
    const out = q.applyChannel(rho, K);
    const expected = markov.leftApply(pn, P);
    for (let i = 0; i < n; i++) actErr = Math.max(actErr, Math.abs(out.re[i][i] - expected[i]));
    const plus = q.densityFromState(Array(n).fill(1 / Math.sqrt(n)));
    const o2 = q.applyChannel(plus, K);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) coherence = Math.max(coherence, Math.hypot(o2.re[i][j], o2.im[i][j]));
  }
  console.log('    Each Markov kernel P becomes a quantum channel with Kraus operators √P_ij |j⟩⟨i| (50 random kernels):');
  console.log(`      valid channel (Σ K†K = I): error ${tiny(krausErr)}; on classical states it acts exactly as P: error ${tiny(actErr)};`);
  console.log(`      it erases all coherence (largest surviving off-diagonal term ${tiny(coherence)}).`);
  return { actErr, krausErr, coherence };
}

/* ────────────── 2. combination without interaction ────────────── */

function part2() {
  console.log('\n  2. Combining quantum agents without interaction (tensor product only)');
  const r = mulberry32(2);
  let best = 0;
  for (let t = 0; t < 1000; t++) {
    let rho = productState(randomQubit(r), randomQubit(r));
    if (t % 2) rho = local(rho, q.depolarizing(r()));
    best = Math.max(best, q.maxChsh(rho));
  }
  console.log(`    1000 random product agents (pure and noisy): best CHSH over all measurements ${f(best, 6)} (local bound 2).`);
  console.log('    Combination alone creates no Bell correlations: an interaction is needed.');
  return { best };
}

/* ────────────── 3. combination with interaction ────────────── */

function part3() {
  console.log('\n  3. Combining quantum agents with an interaction U(t) = exp(-i t X⊗X), starting from |00⟩');
  const rows = [];
  for (const [label, t] of [['0', 0], ['π/16', Math.PI / 16], ['π/8', Math.PI / 8], ['3π/16', 3 * Math.PI / 16], ['π/4', Math.PI / 4]]) {
    const rho = interact(ZERO2, t);
    const m = measured(rho), h = q.maxChsh(rho), formula = 2 * Math.sqrt(1 + Math.sin(2 * t) ** 2);
    rows.push({ label, t, h, m, formula });
    console.log(`      t = ${label.padEnd(6)} best CHSH ${f(h, 6)}   formula 2√(1 + sin²2t) = ${f(formula, 6)}   measured ${f(m.chsh, 6)}  (${m.cls})`);
  }
  const peak = rows[rows.length - 1];
  console.log(`    At t = π/4 the agents reach ${f(peak.h, 6)}; Tsirelson's bound is ${f(TSIRELSON, 6)} (difference ${tiny(Math.abs(peak.h - TSIRELSON))}).`);

  const r = mulberry32(3);
  let best = 0, over = 0, violate = 0, maxSig = 0, bad = 0, matchErr = 0;
  const N = 2000;
  for (let t = 0; t < N; t++) {
    let rho = q.conjugate(productState(randomQubit(r), randomQubit(r)), q.randomUnitary2(r));
    if (t % 4 === 3) rho = local(rho, q.depolarizing(0.3 * r()));
    const h = q.maxChsh(rho), m = measured(rho);
    best = Math.max(best, h);
    over += h > TSIRELSON + 1e-9;
    violate += h > 2 + 1e-9;
    maxSig = Math.max(maxSig, m.signalling);
    matchErr = Math.max(matchErr, Math.abs(m.chsh - h));
    if (m.cls === 'signalling' || m.cls === 'post-quantum') bad++;
  }
  console.log(`    ${N} random interactions (random unitaries on random product agents, a quarter with noise):`);
  console.log(`      best CHSH ${f(best, 6)}; above 2√2: ${over}; above 2: ${violate} (${f(100 * violate / N, 1)}%).`);
  console.log(`      measured statistics match the best value to ${tiny(matchErr)}; classified signalling or post-quantum: ${bad};`);
  console.log(`      largest signalling ${tiny(maxSig)} (experiment 13: 98% of classical networks signal without an extra rule).`);
  return { peak: peak.h, rows, best, over, violate, maxSig, bad, matchErr, N };
}

/* ────────────── 4. network distance and decoherence ────────────── */

function part4() {
  console.log('\n  4. Network distance and decoherence: each agent\'s experience passes through k local channel steps');
  const bellPair = interact(ZERO2, Math.PI / 4);
  const dep = [], deph = [];
  let rho = bellPair, rho2 = bellPair;
  for (let k = 0; k <= 5; k++) {
    dep.push({ k, h: q.maxChsh(rho), formula: TSIRELSON * 0.9 ** (2 * k) });
    deph.push({ k, h: q.maxChsh(rho2), formula: 2 * Math.sqrt(1 + 0.6 ** (4 * k)) });
    rho = local(rho, q.depolarizing(0.1));
    rho2 = local(rho2, q.dephasing(0.2));
  }
  const full = q.maxChsh(local(bellPair, q.dephasing(0.5)));
  const fmtRow = (xs) => xs.map(x => f(x.h)).join('  ');
  const err = Math.max(...dep.map(x => Math.abs(x.h - x.formula)), ...deph.map(x => Math.abs(x.h - x.formula)));
  const crossing = dep.find(x => x.h < 2);
  console.log(`      k =                               ${[0, 1, 2, 3, 4, 5].map(k => String(k).padEnd(5)).join('  ')}`);
  console.log(`      noise (depolarising, p = 0.1)     ${fmtRow(dep)}   = 2√2·0.9^(2k)`);
  console.log(`      decoherence (dephasing, p = 0.2)  ${fmtRow(deph)}   = 2√(1 + 0.6^(4k))`);
  console.log(`      closed forms match to ${tiny(err)}. Noise makes the correlations local from k = ${crossing.k} on.`);
  console.log(`    Full dephasing turns both agents into Markov agents: best CHSH ${f(full, 6)}, exactly the classical bound.`);
  console.log('    Correlations decay with network distance as in experiment 13, but from 2√2, not from 4.');
  return { err, full, crossing: crossing.k, dep, deph };
}

/* ────────────── 5. interference ────────────── */

function part5() {
  console.log('\n  5. Interference: an agent reaches an experience by three routes; block any subset (Sorkin 1994)');
  const r = mulberry32(5);
  const routes = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]];
  const sorkin = (P) => ({
    I2: P['0,1'] - P['0'] - P['1'],
    I3: P['0,1,2'] - P['0,1'] - P['0,2'] - P['1,2'] + P['0'] + P['1'] + P['2'],
  });
  // coherence c: 1 = pure quantum agent, 0 = fully dephased (a Markov agent over routes).
  const coherences = [1, 0.5, 0];
  const worst = coherences.map(() => ({ I2: 0, I3: 0 }));
  const classical = { I2: 0, I3: 0 };
  for (let t = 0; t < 200; t++) {
    const phase = [0, 1, 2].map(() => 2 * Math.PI * r());
    const hop = [0, 1, 2].map(() => r());
    coherences.forEach((c, ci) => {
      const P = {};
      for (const S of routes) {
        // detection probability (1/9) Σ_{k,l ∈ S} c^[k≠l] cos(φ_k − φ_l): amplitudes add, then square
        let v = 0;
        for (const k of S) for (const l of S) v += (k === l ? 1 : c) * Math.cos(phase[k] - phase[l]);
        P[S.join(',')] = v / 9;
      }
      const sq = sorkin(P);
      worst[ci].I2 = Math.max(worst[ci].I2, Math.abs(sq.I2));
      worst[ci].I3 = Math.max(worst[ci].I3, Math.abs(sq.I3));
    });
    const Pc = {};
    for (const S of routes) Pc[S.join(',')] = S.reduce((a, k) => a + hop[k] / 3, 0); // Markov: probabilities add
    const sc = sorkin(Pc);
    classical.I2 = Math.max(classical.I2, Math.abs(sc.I2));
    classical.I3 = Math.max(classical.I3, Math.abs(sc.I3));
  }
  const fmt = (x) => (x < 1e-9 ? tiny(x) : 'up to ' + f(x));
  console.log('    200 random agents; largest |I2| (two-route interference) and |I3| (three-route interference):');
  for (const [name, w] of [['quantum agents', worst[0]], ['half decohered', worst[1]], ['fully decohered', worst[2]], ['Markov agents', classical]]) {
    console.log(`      ${name.padEnd(17)}|I2| ${fmt(w.I2).padEnd(19)} |I3| ${fmt(w.I3)}`);
  }
  const qI2 = worst[0].I2, qI3 = Math.max(...worst.map(w => w.I3));
  console.log('    Quantum agents interfere in pairs but never three at a time: the signature of the Born rule, which');
  console.log('    photon experiments confirm (Sinha et al. 2010). Decoherence scales I2 down to 0, and Markov agents cannot');
  console.log('    interfere at all (experiment 09).');
  return { qI2, qI3, half: worst[1].I2, dephased: worst[2].I2, classical };
}

function main() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Quantum agents');
  console.log('='.repeat(78));
  const R1 = part1(), R2 = part2(), R3 = part3(), R4 = part4(), R5 = part5();

  console.log('\n' + '─'.repeat(78));
  console.log('Findings');
  console.log('─'.repeat(78));
  console.log(`  1. Yes: combined quantum agents reach the quantum limit exactly (${f(R3.peak, 6)} = 2√2) and never exceed it`);
  console.log(`     (0 of ${R3.N} random interactions). No-signalling holds automatically, and the measured statistics pass`);
  console.log('     experiment 13\'s classifier as quantum. Without interaction, combination stays at or below 2.');
  console.log('  2. What does the work is the quantum structure put in, not the agent formalism: the complex Hilbert');
  console.log('     space and the Born rule cap CHSH at 2√2 (Tsirelson\'s theorem), and the tensor-product combination with');
  console.log('     local measurements guarantees no-signalling. Classical kernels supply neither (experiment 13).');
  console.log('     So quantum agents answer experiment 13 by assumption. The open problem for Hoffman\'s program is to');
  console.log('     derive this structure from agent dynamics, or to take quantum agents as the starting point.');
  console.log('  3. The classical world is the decohered limit: a Markov agent is a quantum agent with all coherence');
  console.log(`     erased. Dephasing drives CHSH from 2√2 to exactly ${f(R4.full)} and removes interference; quantum agents`);
  console.log(`     interfere pairwise (|I2| up to ${f(R5.qI2)}) but never three-way (|I3| at ${tiny(R5.qI3)}).`);
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
  return { R1, R2, R3, R4, R5 };
}

if (require.main === module) main();

module.exports = { randomStochastic, randomQubit, productState, interact, local, measured, ZERO2 };
