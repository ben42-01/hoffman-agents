const { describe, it } = require('node:test');
const assert = require('node:assert');
const markov = require('../src/math/markov');
const { mulberry32, fnv1a32 } = require('../src/math/rng');
const { MarkovKernel, StochasticMatrix, FormalConsciousAgent, buildDecisionKernel, DECISION_STATES, ConsciousAgent } = require('../src');

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
const closeArr = (a, b, tol = 1e-9) => { assert.equal(a.length, b.length); a.forEach((v, i) => close(v, b[i], tol)); };

describe('rng', () => {
  it('mulberry32 is deterministic and matches reference values', () => {
    const r = mulberry32(42);
    const got = [r(), r(), r()];
    const again = mulberry32(42);
    assert.deepEqual(got, [again(), again(), again()]);
    close(got[0], 0.6011037519201636, 1e-15);
  });
  it('fnv1a32 matches the published test vectors', () => {
    assert.equal(fnv1a32(''), 0x811C9DC5);
    assert.equal(fnv1a32('a'), 0xE40C292C);
    assert.equal(fnv1a32('foobar'), 0xBF9CF968);
  });
});

describe('stationary distribution', () => {
  it('converges on a periodic 2-cycle (plain power iteration does not)', () => {
    const r = markov.stationary([[0, 1], [1, 0]]);
    assert.ok(r.converged);
    closeArr(r.pi, [0.5, 0.5]);
  });
  it('matches the analytic answer for a birth-death chain', () => {
    const P = [[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]];
    closeArr(markov.stationary(P).pi, [0.25, 0.5, 0.25]);
    closeArr(markov.solveStationary(P), [0.25, 0.5, 0.25]);
  });
  it('matches the two-state closed form pi = (b, a) / (a + b)', () => {
    const a = 0.3, b = 0.1;
    closeArr(markov.stationary([[1 - a, a], [b, 1 - b]]).pi, [b / (a + b), a / (a + b)]);
  });
  it('solveStationary returns null when pi is not unique', () => {
    assert.equal(markov.solveStationary([[1, 0], [0, 1]]), null);
  });
});

describe('chain structure', () => {
  const R = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0.3, 0, 0.2, 0.5], [0, 0, 0, 1]];
  it('finds communicating and closed classes', () => {
    assert.deepEqual(markov.communicatingClasses(R), [[0, 1], [2], [3]]);
    assert.deepEqual(markov.closedClasses(R), [[0, 1], [3]]);
    assert.equal(markov.isIrreducible(R), false);
  });
  it('computes the period', () => {
    assert.equal(markov.period([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), 3);
    assert.equal(markov.period([[0, 1], [1, 0]]), 2);
    assert.equal(markov.period([[0.5, 0.5], [1, 0]]), 1);
    assert.equal(markov.isErgodic([[0, 1], [1, 0]]), false);
    assert.equal(markov.isErgodic([[0.5, 0.5], [1, 0]]), true);
  });
  it('estimates |lambda_2|', () => {
    close(markov.secondEigenvalueModulus([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]]), 0.5, 1e-6);
    close(markov.secondEigenvalueModulus([[0.9, 0.1], [0.5, 0.5]]), 0.4, 1e-6);
    close(markov.secondEigenvalueModulus([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), 1, 1e-6);
  });
  it('prunes rows that were never observed instead of making them absorbing', () => {
    // 0 -> 1 -> 2, 2 has no outgoing data: everything prunes away
    assert.deepEqual(markov.pruneUnobservedRows([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), []);
    // 0 <-> 1, 1 -> 2 (unobserved): 2 pruned, 0 and 1 survive
    assert.deepEqual(markov.pruneUnobservedRows([[0, 1, 0], [1, 0, 1], [0, 0, 0]]), [0, 1]);
  });
});

describe('MarkovKernel', () => {
  const K = new MarkovKernel({ states: ['a', 'b'], matrix: [[0.9, 0.1], [0.5, 0.5]] });
  it('rejects non-stochastic matrices', () => {
    assert.throws(() => new MarkovKernel({ states: ['a'], matrix: [[0.5]] }), /sums to/);
    assert.throws(() => new MarkovKernel({ states: ['a', 'b'], matrix: [[1.5, -0.5], [0, 1]] }), /negative/);
  });
  it('tensor product has stationary distribution pi1 ⊗ pi2 and stochastic rows', () => {
    const T = K.tensor(K);
    assert.ok(markov.isStochastic(T.matrix));
    const pi = K.stationary().pi;
    const piT = T.stationary().distribution;
    close(piT['a|b'], pi[0] * pi[1], 1e-9);
    close(piT['b|b'], pi[1] * pi[1], 1e-9);
  });
  it('sampling frequencies match the kernel row', () => {
    const r = mulberry32(3);
    let a = 0;
    for (let i = 0; i < 100000; i++) if (K.sample('b', r) === 'a') a++;
    close(a / 100000, 0.5, 0.01);
  });
  it('compose and power', () => {
    const K2 = K.power(2);
    close(K2.prob('a', 'a'), 0.9 * 0.9 + 0.1 * 0.5);
  });
  it('rectangular stochastic matrices compose', () => {
    const D = new StochasticMatrix({ rows: ['x', 'y'], cols: ['a', 'b'], matrix: [[1, 0], [0.5, 0.5]] });
    close(D.compose(K).prob('y', 'a'), 0.5 * 0.9 + 0.5 * 0.5);
  });
});

describe('decision kernel D', () => {
  it('rejects parameters that are not a distribution', () => {
    assert.throws(() => buildDecisionKernel({ pStable: 0.9, pLexicon: 0.2, pExplore: 0.1 }), /<= 1/);
    assert.throws(() => new ConsciousAgent({ pStable: 0.9, pLexicon: 0.2 }), /<= 1/);
    assert.throws(() => buildDecisionKernel({ pStable: -0.1 }), /\[0, 1\]/);
  });
  it('is ergodic with the default parameters', () => {
    const d = buildDecisionKernel().diagnostics();
    assert.equal(d.ergodic, true);
    assert.deepEqual(d.states, DECISION_STATES);
  });
  it('empirical output-mode frequencies converge to its stationary distribution', () => {
    const K = buildDecisionKernel({ pStable: 0.6, pLexicon: 0.2, pExplore: 0.1 });
    const pi = K.stationary().distribution;
    const r = mulberry32(11);
    const counts = Object.fromEntries(DECISION_STATES.map(s => [s, 0]));
    let s = 'core';
    const n = 100000;
    for (let i = 0; i < n; i++) { s = K.sample(s, r); counts[s]++; }
    const tv = DECISION_STATES.reduce((acc, k) => acc + Math.abs(counts[k] / n - pi[k]), 0) / 2;
    assert.ok(tv < 0.01, `TV ${tv}`);
  });
});

describe('FormalConsciousAgent', () => {
  const make = (seed) => new FormalConsciousAgent({
    X: ['calm', 'alert'], G: ['stay', 'move'], W: ['left', 'right'],
    P: { left: [[0.9, 0.1], [0.6, 0.4]], right: [[0.3, 0.7], [0.1, 0.9]] },
    D: [[0.8, 0.2], [0.2, 0.8]],
    A: { stay: [[1, 0], [0, 1]], move: [[0, 1], [1, 0]] },
    rng: mulberry32(seed),
  });
  it('joint kernel Q is stochastic and ergodic', () => {
    const c = make(1);
    assert.ok(markov.isStochastic(c.jointKernel().matrix));
    assert.equal(c.diagnostics().ergodic, true);
  });
  it('simulation agrees with the stationary distribution of Q (ergodic theorem)', () => {
    const c = make(7);
    const pi = c.diagnostics().stationary;
    const counts = {};
    let w = 'left';
    const n = 100000;
    for (let i = 0; i < n; i++) { const o = c.step(w); w = o.w; counts[`${o.x}|${w}`] = (counts[`${o.x}|${w}`] || 0) + 1; }
    for (const k of Object.keys(pi)) close((counts[k] || 0) / n, pi[k], 0.01);
    assert.equal(c.N, n);
  });
  it('combination builds product perception/decision kernels', () => {
    const cc = FormalConsciousAgent.combine(make(1), make(2));
    assert.equal(cc.X.length, 4);
    assert.equal(cc.G.length, 4);
    assert.ok(markov.isStochastic(cc.jointKernel().matrix));
    const round = FormalConsciousAgent.fromJSON(JSON.parse(JSON.stringify(cc.toJSON())));
    assert.deepEqual(round.jointKernel().matrix.map(r => Array.from(r)), cc.jointKernel().matrix.map(r => Array.from(r)));
  });
});
