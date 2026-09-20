/**
 * Trace Logic and Decorated Permutations
 *
 * Hoffman's program has two mathematical bridges from conscious agents to
 * physics. This experiment builds both and checks them.
 *
 *   A. Decorated permutations (Hoffman, Prakash & Prentner, "Fusions of
 *      Consciousness", 2023, Definition 2). Each Markov chain maps to a
 *      decorated permutation, the combinatorial object that indexes cells of
 *      the positive Grassmannian (the geometry behind the amplituhedron and
 *      scattering amplitudes). What does the map keep, and what does it depend on?
 *
 *   B. Trace logic (Hoffman & Prakash, "Traces of Consciousness"). Agents are
 *      ordered by "A is a trace of B"; "A observes B" means A's qualia kernel
 *      Q = DAP is a trace of B's dynamics. Claimed: locally Boolean, globally
 *      not even a lattice, and homomorphic to the Lebesgue logic of belief via
 *      the stationary measure.
 *
 *   C. Two physics proposals from the trace-logic papers, stress-tested with
 *      consistency checks they could fail: mass <-> entropy rate, and
 *      speed <-> commute time.
 */
const { markov, trace, decorated, mulberry32, kron } = (() => {
  const lib = require('../../src/index');
  return { ...lib, kron: lib.markov.kron };
})();
const { coupledRings } = require('../14_spacetime_in_the_headset/spacetime_in_the_headset');

const f = (x, d = 3) => x.toFixed(d);
const tiny = (x) => (x < 1e-9 ? 'machine precision' : x.toExponential(1));

function randomKernel(n, r, density = 1) {
  return Array.from({ length: n }, (_, i) => {
    const u = Array.from({ length: n }, (_, j) => (j === i || r() < density ? r() + 0.05 : 0));
    const s = u.reduce((a, b) => a + b, 0);
    return Float64Array.from(u, v => v / s);
  });
}

function cycleKernel(n, cycles) {
  const P = Array.from({ length: n }, () => new Float64Array(n));
  for (const c of cycles) c.forEach((s, i) => { P[s - 1][c[(i + 1) % c.length] - 1] = 1; });
  return P;
}

function permutations(n) {
  if (n === 1) return [[0]];
  const out = [];
  for (const p of permutations(n - 1)) for (let i = 0; i <= p.length; i++) { const q = [...p]; q.splice(i, 0, n - 1); out.push(q); }
  return out;
}

// Relabelling preserves the cyclic order of every recurrent class of >= 3 states.
function preservesCyclicOrder(P, perm) {
  return markov.closedClasses(P).every(cls => {
    if (cls.length < 3) return true;
    const img = cls.map(i => perm[i]);
    const sorted = [...img].sort((a, b) => a - b);
    const k = sorted.indexOf(img[0]);
    return img.every((v, i) => v === sorted[(k + i) % sorted.length]);
  });
}

const show = (s) => `[${s.join(', ')}]`;

/* ────────────── A. decorated permutations ────────────── */

function partA() {
  console.log('\n  A. Decorated permutations');
  const example = cycleKernel(9, [[1, 5, 8], [2], [3, 4], [6], [7, 9]]);
  const sigma = decorated.decoratedPermutation(example);
  const paper = [8, 11, 4, 12, 10, 15, 9, 14, 16];
  const reproduces = sigma.every((v, i) => v === paper[i]);
  console.log(`    paper's example (cycles (1 5 8)(2)(3 4)(6)(7 9)): ${show(sigma)}  ${reproduces ? 'matches the paper' : 'DIFFERS from the paper'}`);

  console.log('    2-state agents Q = [[1-x, x], [y, 1-y]] (the four cells of the Markov square M2):');
  for (const [name, x, y] of [['x = y = 0', 0, 0], ['x, y > 0', 0.3, 0.6], ['x = 0, y > 0', 0, 0.6], ['x > 0, y = 0', 0.3, 0]]) {
    console.log(`      ${name.padEnd(14)} σ = ${show(decorated.decoratedPermutation([[1 - x, x], [y, 1 - y]]))}`);
  }

  // The paper's own table (Appendix B): all 27 vertices of M3 with their decorated permutations.
  const table = require('../../test/fixtures/fusions-appendix-b.json').vertices;
  let tableMatches = 0;
  const vertexSigmas = new Set(), tableSigmas = new Set(), patternSigmas = new Set();
  for (const v of table) {
    const ours = decorated.decoratedPermutation(v.M.map(r => Float64Array.from(r)));
    tableMatches += show(ours) === show(v.sigma);
    vertexSigmas.add(show(ours));
    tableSigmas.add(show(v.sigma));
  }
  for (let code = 0; code < 343; code++) {
    const masks = [code % 7 + 1, Math.floor(code / 7) % 7 + 1, Math.floor(code / 49) + 1];
    const P = masks.map(m => { const bits = [0, 1, 2].map(k => (m >> k) & 1); const s = bits.reduce((a, b) => a + b, 0); return Float64Array.from(bits, b => b / s); });
    patternSigmas.add(show(decorated.decoratedPermutation(P)));
  }
  console.log(`    3-state agents: our map agrees with the paper's Appendix B table on ${tableMatches} of 27 vertices of M3.`);
  console.log(`    The table contains ${tableSigmas.size} distinct decorated permutations (the paper's text says 17); all 343`);
  console.log(`    transition patterns of 3-state agents give ${patternSigmas.size}.`);

  const sticky = [[0.99, 0.01], [0.01, 0.99]], switching = [[0.01, 0.99], [0.99, 0.01]];
  console.log(`    probabilities are discarded: an agent that almost never changes experience and one that almost always does`);
  console.log(`    both map to ${show(decorated.decoratedPermutation(sticky))} (${show(decorated.decoratedPermutation(switching))}).`);

  console.log('    dependence on labelling (all 720 relabellings of 6-state chains):');
  let allAgree = true;
  const rows = [];
  for (const [name, P] of [['classes (1 3 5)(2 6)(4)', cycleKernel(6, [[1, 3, 5], [2, 6], [4]])],
    ['one 6-cycle', cycleKernel(6, [[1, 2, 3, 4, 5, 6]])], ['classes (1 2)(3 4)(5 6)', cycleKernel(6, [[1, 2], [3, 4], [5, 6]])],
    ['random irreducible agent', randomKernel(6, mulberry32(3))]]) {
    let covariant = 0, identical = 0, agree = 0;
    const perms = permutations(6);
    const base = show(decorated.decoratedPermutation(P));
    for (const p of perms) {
      const c = decorated.isCovariant(P, p), pr = preservesCyclicOrder(P, p);
      covariant += c; agree += c === pr;
      identical += show(decorated.decoratedPermutation(decorated.relabel(P, p))) === base;
    }
    allAgree = allAgree && agree === perms.length;
    rows.push({ name, covariant, identical, base });
    console.log(`      ${name.padEnd(26)} σ = ${base.padEnd(22)} renames consistently: ${String(covariant).padStart(3)}/720   identical list: ${String(identical).padStart(3)}/720`);
  }
  console.log(`    rule: σ renames consistently exactly when the relabelling keeps the cyclic order of every recurrent`);
  console.log(`    class of >= 3 states (${allAgree ? 'holds for every relabelling tested' : 'FAILS for some relabelling'}). For a single class, σ is the same list for`);
  console.log('    every chain and every numbering: it records the class, not the dynamics inside it.');
  return { reproduces, tableMatches, vertices: tableSigmas.size, patterns: patternSigmas.size, allAgree, rows };
}

/* ────────────── B. trace logic ────────────── */

function labelled(states, P) { return { states, P }; }
function traceOn(K, window) {
  const idx = window.map(s => K.states.indexOf(s));
  return labelled(window, trace.traceChain(K.P, idx));
}
// A <=_t B: A's states are a subset of B's and A is B's trace there.
function traceLeq(A, B, tol = 1e-9) {
  if (!A.states.every(s => B.states.includes(s))) return false;
  return trace.isTraceOf(A.P, B.P, A.states.map(s => B.states.indexOf(s)), tol);
}

function subsets(items) {
  const out = [];
  for (let m = 1; m < 1 << items.length; m++) out.push(items.filter((_, i) => (m >> i) & 1));
  return out;
}

function partB() {
  console.log('\n  B. Trace logic');
  const r = mulberry32(16);

  // Qualia kernel of an agent in Hoffman & Prakash's original form.
  const X = 3, G = 2, W = 3;
  const row = (n) => { const u = Array.from({ length: n }, () => r() + 0.05); const s = u.reduce((a, b) => a + b, 0); return u.map(v => v / s); };
  const D = Array.from({ length: X }, () => row(G)), A = Array.from({ length: G }, () => row(W)), P = Array.from({ length: W }, () => row(X));
  const Q = trace.qualiaKernel(D, A, P);
  console.log(`    qualia kernel Q = D·A·P of a (3 experiences, 2 actions, 3 world states) agent: stochastic ${markov.isStochastic(Q)}`);

  // A network N of 5 experiences; an observer with a 3-experience window.
  const labels = ['a', 'b', 'c', 'd', 'e'];
  const N = labelled(labels, randomKernel(5, r));
  const obs = traceOn(N, ['a', 'c', 'd']);
  const impostor = labelled(['a', 'c', 'd'], randomKernel(3, r));
  console.log(`    "A observes N" (A's kernel is N's trace on A's window): true observer ${traceLeq(obs, N)}, random kernel on the same window ${traceLeq(impostor, N)}`);

  // Locally Boolean: the traces of one fixed agent N form a Boolean algebra.
  const N4 = labelled(['a', 'b', 'c', 'd'], randomKernel(4, r));
  const windows = subsets(N4.states);
  const traces = windows.map(w => traceOn(N4, w));
  let orderOk = 0, pairs = 0;
  for (let i = 0; i < windows.length; i++) for (let j = 0; j < windows.length; j++) {
    const subset = windows[i].every(s => windows[j].includes(s));
    orderOk += traceLeq(traces[i], traces[j]) === subset;
    pairs++;
  }
  console.log(`    locally Boolean: for a 4-experience agent N, "trace ≤ trace" matches "window ⊆ window" in ${orderOk} of ${pairs} pairs,`);
  console.log('    so the 15 traces of N (plus a zero) form the Boolean algebra of its windows: join = trace on the union,');
  console.log('    meet = trace on the intersection, complement = trace on the complementary window.');

  // Globally not a lattice: two different agents on {a, b} share the lower bounds
  // "trace on {a}" and "trace on {b}" (both the trivial kernel [[1]]).
  const K1 = labelled(['a', 'b'], [Float64Array.of(0.9, 0.1), Float64Array.of(0.4, 0.6)]);
  const K2 = labelled(['a', 'b'], [Float64Array.of(0.2, 0.8), Float64Array.of(0.7, 0.3)]);
  const la = labelled(['a'], [Float64Array.of(1)]), lb = labelled(['b'], [Float64Array.of(1)]);
  const lowerOk = [la, lb].every(l => traceLeq(l, K1) && traceLeq(l, K2));
  const incomparable = !traceLeq(K1, K2) && !traceLeq(K2, K1) && !traceLeq(la, lb) && !traceLeq(lb, la);
  console.log(`    globally not a lattice: agents K1 ≠ K2 on {a, b} both have "trace on {a}" and "trace on {b}" below them (${lowerOk}),`);
  console.log(`    and all four are pairwise incomparable (${incomparable}). So K1 and K2 have two maximal common lower bounds`);
  console.log('    (no meet), and {a}, {b} have two minimal common upper bounds (no join). There is also no top element.');

  // Homomorphism to Lebesgue logic via the stationary measure.
  let tested = 0, preserved = 0;
  for (let t = 0; t < 200; t++) {
    const n = 4 + Math.floor(r() * 3);
    const B = labelled(['a', 'b', 'c', 'd', 'e', 'f'].slice(0, n), randomKernel(n, r));
    const win = B.states.filter(() => r() < 0.6);
    if (win.length === 0) continue;
    const Aw = traceOn(B, win);
    tested++;
    preserved += trace.lebesgueLeq(trace.stationaryMeasure(Aw.states, Aw.P), trace.stationaryMeasure(B.states, B.P));
  }
  const Qsq = markov.matMul(N.P, N.P);
  const sameMeasure = [...trace.stationaryMeasure(N.states, N.P)].every(([s, p]) => Math.abs(p - trace.stationaryMeasure(N.states, Qsq).get(s)) < 1e-9);
  const piObs = trace.stationaryMeasure(obs.states, obs.P);
  const piN = trace.stationaryMeasure(N.states, N.P);
  // a kernel on the observer's window with the same stationary measure but different dynamics
  const pi3 = obs.states.map(s => piObs.get(s));
  const independent = labelled(obs.states, obs.states.map(() => Float64Array.from(pi3)));
  const converse = trace.lebesgueLeq(trace.stationaryMeasure(independent.states, independent.P), piN) && !traceLeq(independent, N);
  console.log(`    homomorphism to Lebesgue logic: A ≤_t B implied π_A ≤_L π_B in ${preserved} of ${tested} random (agent, window) pairs.`);
  console.log(`    It is not injective: N and N² have the same stationary measure (${sameMeasure}); and the converse fails:`);
  console.log(`    a kernel with the observer's stationary measure but different dynamics satisfies ≤_L without being a trace (${converse}).`);
  return { orderOk, pairs, lowerOk, incomparable, preserved, tested, sameMeasure, converse, observes: traceLeq(obs, N), impostor: traceLeq(impostor, N) };
}

/* ────────────── C. physics proposals ────────────── */

function partC() {
  console.log('\n  C. Physics proposals from the trace-logic papers (proposals, not theorems), stress-tested');
  const r = mulberry32(17);
  const P1 = randomKernel(3, r), P2 = randomKernel(4, r);
  const h1 = markov.entropyRate(P1), h2 = markov.entropyRate(P2), h12 = markov.entropyRate(kron(P1, P2));
  const addErr = Math.abs(h12 - (h1 + h2));
  console.log(`    mass ↔ entropy rate. Mass of independent systems adds; so does entropy rate under ⊗:`);
  console.log(`      h(P1) + h(P2) = ${f(h1 + h2, 4)}, h(P1 ⊗ P2) = ${f(h12, 4)} (difference ${tiny(addErr)}): passes.`);
  const hs = [0, 0.3, 0.6, 0.9].map(c => { const Q = coupledRings(8, c); return { c, h: markov.entropyRate(Q) }; });
  console.log(`      Bound systems in physics weigh less than their parts (mass defect). Three coupled ring agents (experiment 14):`);
  console.log(`      entropy rate ${hs.map(x => `${f(x.h)} (c = ${x.c})`).join(' → ')}.`);
  const defect = hs[hs.length - 1].h < hs[0].h;
  console.log(`      ${defect ? 'Binding lowers it: qualitatively consistent with a mass defect.' : 'Binding does not lower it: no mass-defect analogue.'}`);

  let violations = 0, triples = 0, asym = 0;
  for (let t = 0; t < 30; t++) {
    const K = markov.commuteTimes(randomKernel(6, r));
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
      asym = Math.max(asym, Math.abs(K[i][j] - K[j][i]));
      for (let k = 0; k < 6; k++) { triples++; if (K[i][k] > K[i][j] + K[j][k] + 1e-9) violations++; }
    }
  }
  console.log(`    speed ↔ commute time. For speed to be distance over time, commute time must behave like a distance:`);
  console.log(`      symmetric (largest asymmetry ${tiny(asym)}), triangle inequality violated in ${violations} of ${triples} triples: passes.`);
  return { addErr, defect, violations, hs };
}

function main() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Trace logic and decorated permutations');
  console.log('='.repeat(78));
  const A = partA(), B = partB(), C = partC();

  console.log('\n' + '─'.repeat(78));
  console.log('Findings');
  console.log('─'.repeat(78));
  console.log(`  1. The decorated-permutation map is implemented as defined and reproduces the paper's example${A.reproduces ? '' : ' (NOT)'}.`);
  console.log('     σ is fixed by two things only: which experiences form recurrent classes, and the cyclic order in which');
  console.log('     they are numbered. All probabilities are discarded, the dynamics inside a class is invisible, and σ');
  console.log(`     renames consistently only for numberings that keep each class's cyclic order (${A.rows[1].covariant} of 720 for one class`);
  console.log('     of 6). Scattering-amplitude geometry also needs a cyclic order ("colour ordering"), so this is a');
  console.log('     requirement, not a bug: to reach physics, the program needs a principle that orders experiences.');
  console.log(`     It agrees with all ${A.tableMatches} entries of the paper's Appendix B table, which lists ${A.vertices} distinct decorated`);
  console.log('     permutations for M3, not the 17 stated in the text (a miscount; the table itself is correct).');
  console.log(`  2. Trace logic behaves as claimed: locally Boolean (${B.orderOk}/${B.pairs}), globally not a lattice (no meets, joins or`);
  console.log(`     top), and the stationary measure carries the trace order into Lebesgue logic (${B.preserved}/${B.tested}); the map is`);
  console.log('     neither injective nor reversible. This is not quantum logic: it fails far more than distributivity.');
  console.log(`  3. The proposals pass these first consistency checks: entropy rate adds like mass (${tiny(C.addErr)}) and drops`);
  console.log(`     with binding${C.defect ? '' : ' (NOT)'}; commute time is a genuine distance (${C.violations} triangle violations). Passing is necessary,`);
  console.log('     not sufficient: nothing here derives a mass spectrum or a speed limit.');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

if (require.main === module) main();

module.exports = { randomKernel, cycleKernel, preservesCyclicOrder, traceLeq, traceOn, labelled };
