const test = require('node:test');
const assert = require('node:assert');
const { quantum: q, bell, markov, mulberry32 } = require('../src/index');
const {
  randomStochastic, randomQubit, productState, interact, local, measured, ZERO2,
} = require('../examples/17_quantum_agents/quantum_agents');

const TSIRELSON = 2 * Math.SQRT2;

test('Bell state reaches Tsirelson exactly; textbook angles agree', () => {
  const s = Math.SQRT1_2;
  const rho = q.densityFromState([s, 0, 0, s]);
  assert.ok(Math.abs(q.maxChsh(rho) - TSIRELSON) < 1e-12);
  const p = q.behaviour(rho, [[0, 0, 1], [1, 0, 0]], [[s, 0, s], [-s, 0, s]]);
  assert.ok(Math.abs(bell.chsh(bell.correlators(p)) - TSIRELSON) < 1e-12);
  assert.strictEqual(bell.classify(p), 'quantum');
});

test('XX interaction: CHSH = 2 sqrt(1 + sin^2 2t)', () => {
  for (const t of [0, 0.1, 0.4, Math.PI / 4, 1.2]) {
    assert.ok(Math.abs(q.maxChsh(interact(ZERO2, t)) - 2 * Math.sqrt(1 + Math.sin(2 * t) ** 2)) < 1e-12);
  }
});

test('product agents never exceed 2; random interactions never exceed 2 sqrt 2 and never signal', () => {
  const r = mulberry32(9);
  for (let t = 0; t < 200; t++) {
    assert.ok(q.maxChsh(productState(randomQubit(r), randomQubit(r))) <= 2 + 1e-9);
    const rho = q.conjugate(productState(randomQubit(r), randomQubit(r)), q.randomUnitary2(r));
    assert.ok(Math.abs(q.traceRe(rho) - 1) < 1e-12);
    const m = measured(rho);
    assert.ok(q.maxChsh(rho) <= TSIRELSON + 1e-9);
    assert.ok(Math.abs(m.chsh - q.maxChsh(rho)) < 1e-9);
    assert.ok(m.signalling < 1e-12);
    assert.ok(m.cls === 'local' || m.cls === 'quantum');
  }
});

test('Markov kernels embed as channels acting exactly as P on classical states', () => {
  const r = mulberry32(4);
  const P = randomStochastic(3, r), p = randomStochastic(3, r)[0];
  const K = q.markovChannel(P);
  assert.ok(q.krausDeviation(K) < 1e-12);
  const out = q.applyChannel(q.cmat(p.map((v, i) => p.map((_, j) => (i === j ? v : 0)))), K);
  const expected = markov.leftApply(p, P);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(out.re[i][i] - expected[i]) < 1e-12);
});

test('channels are trace preserving; depolarising and dephasing closed forms', () => {
  assert.ok(q.krausDeviation(q.depolarizing(0.37)) < 1e-12);
  assert.ok(q.krausDeviation(q.dephasing(0.2)) < 1e-12);
  let rho = interact(ZERO2, Math.PI / 4), rho2 = rho;
  for (let k = 1; k <= 3; k++) {
    rho = local(rho, q.depolarizing(0.1));
    rho2 = local(rho2, q.dephasing(0.2));
    assert.ok(Math.abs(q.maxChsh(rho) - TSIRELSON * 0.9 ** (2 * k)) < 1e-12);
    assert.ok(Math.abs(q.maxChsh(rho2) - 2 * Math.sqrt(1 + 0.6 ** (4 * k))) < 1e-12);
  }
  assert.ok(Math.abs(q.maxChsh(local(interact(ZERO2, Math.PI / 4), q.dephasing(0.5))) - 2) < 1e-12);
});

test('symmetricEigen returns orthonormal eigenvectors', () => {
  const M = [[2, 1, 0], [1, 3, 1], [0, 1, 4]];
  const { values, vectors } = q.symmetricEigen(M);
  vectors.forEach((v, k) => {
    const Mv = M.map(row => row.reduce((s, x, j) => s + x * v[j], 0));
    Mv.forEach((x, i) => assert.ok(Math.abs(x - values[k] * v[i]) < 1e-10));
  });
  assert.ok(values[0] >= values[1] && values[1] >= values[2]);
});
