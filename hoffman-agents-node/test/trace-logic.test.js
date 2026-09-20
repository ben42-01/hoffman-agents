const test = require('node:test');
const assert = require('node:assert');
const { markov, trace, decorated, mulberry32 } = require('../src/index');
const {
  randomKernel, cycleKernel, preservesCyclicOrder, traceLeq, traceOn, labelled,
} = require('../examples/16_trace_logic_and_decorated_permutations/trace_logic');

const show = (s) => s.join(',');

test('decorated permutation reproduces the paper example', () => {
  const P = cycleKernel(9, [[1, 5, 8], [2], [3, 4], [6], [7, 9]]);
  assert.deepStrictEqual(decorated.decoratedPermutation(P), [8, 11, 4, 12, 10, 15, 9, 14, 16]);
});

test('decorated permutation matches all 27 entries of Fusions Appendix B; 14 distinct', () => {
  const { vertices } = require('./fixtures/fusions-appendix-b.json');
  assert.strictEqual(vertices.length, 27);
  const distinct = new Set();
  for (const v of vertices) {
    const sigma = decorated.decoratedPermutation(v.M.map(r => Float64Array.from(r)));
    assert.deepStrictEqual(sigma, v.sigma, `vertex ${JSON.stringify(v.M)}`);
    distinct.add(show(sigma));
  }
  assert.strictEqual(distinct.size, 14);
});

test('decorated permutation: transient fixed, absorbing a -> a+n, probabilities ignored', () => {
  assert.deepStrictEqual(decorated.decoratedPermutation([[1, 0], [0.5, 0.5]]), [3, 2]);
  assert.deepStrictEqual(decorated.decoratedPermutation([[0.99, 0.01], [0.01, 0.99]]),
    decorated.decoratedPermutation([[0.01, 0.99], [0.99, 0.01]]));
});

test('decorated permutation renames consistently iff cyclic order of classes >= 3 is kept', () => {
  const perms = [];
  const gen = (p, rest) => { if (!rest.length) perms.push(p); rest.forEach((x, i) => gen([...p, x], rest.filter((_, j) => j !== i))); };
  gen([], [0, 1, 2, 3, 4, 5]);
  for (const P of [cycleKernel(6, [[1, 3, 5], [2, 6], [4]]), cycleKernel(6, [[1, 2, 3, 4, 5, 6]]), randomKernel(6, mulberry32(3))]) {
    for (const p of perms) assert.strictEqual(decorated.isCovariant(P, p), preservesCyclicOrder(P, p));
  }
});

test('trace logic: locally Boolean (trace order = window inclusion)', () => {
  const N = labelled(['a', 'b', 'c', 'd'], randomKernel(4, mulberry32(5)));
  const windows = [];
  for (let m = 1; m < 16; m++) windows.push(N.states.filter((_, i) => (m >> i) & 1));
  const traces = windows.map(w => traceOn(N, w));
  for (let i = 0; i < 15; i++) for (let j = 0; j < 15; j++) {
    assert.strictEqual(traceLeq(traces[i], traces[j]), windows[i].every(s => windows[j].includes(s)));
  }
});

test('trace logic: globally not a lattice (two maximal common lower bounds)', () => {
  const K1 = labelled(['a', 'b'], [Float64Array.of(0.9, 0.1), Float64Array.of(0.4, 0.6)]);
  const K2 = labelled(['a', 'b'], [Float64Array.of(0.2, 0.8), Float64Array.of(0.7, 0.3)]);
  const la = labelled(['a'], [Float64Array.of(1)]), lb = labelled(['b'], [Float64Array.of(1)]);
  for (const l of [la, lb]) { assert.ok(traceLeq(l, K1)); assert.ok(traceLeq(l, K2)); }
  assert.ok(!traceLeq(K1, K2) && !traceLeq(K2, K1) && !traceLeq(la, lb) && !traceLeq(lb, la));
});

test('trace order maps into Lebesgue order via the stationary measure', () => {
  const r = mulberry32(11);
  for (let t = 0; t < 50; t++) {
    const n = 3 + Math.floor(r() * 4);
    const B = labelled(['a', 'b', 'c', 'd', 'e', 'f'].slice(0, n), randomKernel(n, r));
    const win = B.states.filter(() => r() < 0.6);
    if (!win.length) continue;
    const A = traceOn(B, win);
    assert.ok(traceLeq(A, B));
    assert.ok(trace.lebesgueLeq(trace.stationaryMeasure(A.states, A.P), trace.stationaryMeasure(B.states, B.P)));
  }
  // not the other way round: a measure-compatible kernel need not be a trace
  const nu = new Map([['a', 0.5], ['b', 0.5]]);
  assert.ok(trace.lebesgueLeq(nu, new Map([['a', 0.25], ['b', 0.25], ['c', 0.5]])));
  assert.ok(!trace.lebesgueLeq(nu, new Map([['a', 0.1], ['b', 0.3], ['c', 0.6]])));
});

test('qualia kernel D·A·P is stochastic', () => {
  const Q = trace.qualiaKernel([[0.3, 0.7], [1, 0]], [[0.5, 0.5, 0], [0, 0.2, 0.8]], [[1, 0], [0.4, 0.6], [0, 1]]);
  assert.ok(markov.isStochastic(Q));
});

test('entropy rate is additive under tensor product', () => {
  const r = mulberry32(17);
  const P1 = randomKernel(3, r), P2 = randomKernel(4, r);
  const h = markov.entropyRate(markov.kron(P1, P2));
  assert.ok(Math.abs(h - markov.entropyRate(P1) - markov.entropyRate(P2)) < 1e-10);
  assert.ok(Math.abs(markov.entropyRate([[0.5, 0.5], [0.5, 0.5]]) - Math.log(2)) < 1e-12);
});

test('hitting and commute times: two-state closed form, metric', () => {
  const H = markov.hittingTimes([[0.7, 0.3], [0.2, 0.8]]);
  assert.ok(Math.abs(H[0][1] - 1 / 0.3) < 1e-12 && Math.abs(H[1][0] - 1 / 0.2) < 1e-12);
  const K = markov.commuteTimes(randomKernel(6, mulberry32(2)));
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
    assert.ok(Math.abs(K[i][j] - K[j][i]) < 1e-9);
    for (let k = 0; k < 6; k++) assert.ok(K[i][k] <= K[i][j] + K[j][k] + 1e-9);
  }
});
