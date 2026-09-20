const { describe, it } = require('node:test');
const assert = require('node:assert');
const { MarkovKernel, markov, trace, mulberry32 } = require('../src');
const { agentNetwork, reversibleNetwork, windowOf, simulate } = require('../examples/15_time_in_the_traces/time_in_the_traces');

const near = (a, b, tol, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);
const maxDiff = (A, B) => Math.max(...A.map((r, i) => Math.max(...Array.from(r, (v, j) => Math.abs(v - B[i][j])))));

function randomKernel(n, seed) {
  const r = mulberry32(seed);
  return Array.from({ length: n }, () => { const u = Array.from({ length: n }, () => r()); const s = u.reduce((a, b) => a + b); return Float64Array.from(u, v => v / s); });
}

describe('trace chains', () => {
  const P = randomKernel(7, 3);
  const pi = markov.stationary(P).pi;

  it('are stochastic, and the full window returns the chain itself', () => {
    assert.ok(markov.isStochastic(trace.traceChain(P, [0, 3, 4])));
    assert.ok(maxDiff(trace.traceChain(P, [0, 1, 2, 3, 4, 5, 6]), P) < 1e-15);
  });

  it('are transitive: the trace of a trace is the trace on the smaller window', () => {
    const outer = [0, 2, 3, 5, 6], innerStates = [2, 6];
    const viaOuter = trace.traceChain(trace.traceChain(P, outer), innerStates.map(s => outer.indexOf(s)));
    assert.ok(maxDiff(viaOuter, trace.traceChain(P, innerStates)) < 1e-12);
  });

  it('have the restricted stationary distribution', () => {
    const S = [1, 4, 5];
    const piT = markov.stationary(trace.traceChain(P, S)).pi;
    const mass = trace.clockRate(pi, S);
    S.forEach((s, i) => near(piT[i], pi[s] / mass, 1e-10));
  });

  it('satisfy Kac: mean return time = 1 / pi(S)', () => {
    for (const S of [[0], [1, 2], [0, 3, 5, 6]]) near(trace.meanReturnTime(P, S, pi), 1 / trace.clockRate(pi, S), 1e-9);
  });

  it('match what a simulated observer registers', () => {
    const { P: Q, states } = agentNetwork(15);
    const S = windowOf(states, s => s[2] === 'L');
    const [sim] = simulate(Q, 100000, 1, [S]);
    assert.ok(maxDiff(markov.normalizeRows(sim.counts).P, trace.traceChain(Q, S)) < 0.02);
    near(sim.ticks / 100000, trace.clockRate(markov.stationary(Q).pi, S), 0.01);
  });

  it('can hide the arrow of time: a one-way cycle seen through two states is reversible', () => {
    const cycle = [[0, 1, 0], [0, 0, 1], [1, 0, 0]];
    assert.equal(markov.irreversibility(cycle), 1);
    assert.equal(markov.irreversibility(trace.traceChain(cycle, [0, 1])), 0);
  });

  it('preserve reversibility, and observer uncertainty grows monotonically', () => {
    const R = reversibleNetwork(8, 4);
    const T = trace.traceChain(R, [0, 1, 2]);
    near(markov.irreversibility(T), 0, 1e-9);
    const h = trace.conditionalEntropyProfile(T, markov.stationary(T).pi, 6);
    h.forEach((v, i) => { if (i) assert.ok(v >= h[i - 1] - 1e-12); });
  });

  it('reject windows the chain can avoid forever, and invalid windows', () => {
    const absorbing = [[1, 0, 0], [0, 1, 0], [0.5, 0, 0.5]];
    assert.throws(() => trace.traceChain(absorbing, [2]), /no trace/);
    assert.throws(() => trace.traceChain(P, []), /non-empty/);
    assert.throws(() => trace.traceChain(P, [0, 0]), /repeated/);
    assert.throws(() => trace.traceChain(P, [9]), /out of range/);
  });

  it('MarkovKernel.trace and hasTrace', () => {
    const K = new MarkovKernel({ states: ['a', 'b', 'c'], matrix: [[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]] });
    const T = K.trace(['a', 'c']);
    assert.deepEqual(T.states, ['a', 'c']);
    assert.ok(K.hasTrace(T));
    assert.ok(!K.hasTrace(new MarkovKernel({ states: ['a', 'c'], matrix: [[0.5, 0.5], [0.5, 0.5]] })));
  });
});
