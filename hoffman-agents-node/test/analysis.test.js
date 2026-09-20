const { describe, it } = require('node:test');
const assert = require('node:assert');
const markov = require('../src/math/markov');
const bell = require('../src/analysis/bell');
const { hiddenNetwork, optimiseChsh } = require('../examples/13_bell_through_the_headset/bell_through_the_headset');
const { torus, ringAgent, coupledRings, learnedKernel } = require('../examples/14_spacetime_in_the_headset/spacetime_in_the_headset');
const { FormalConsciousAgent } = require('../src');

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} vs ${b} (tol ${tol})`);

describe('markov: arrow of time and contraction', () => {
  it('irreversibility is 0 for reversible chains and 1 for a one-way cycle', () => {
    near(markov.irreversibility(torus(10, 1)), 0, 1e-12);
    near(markov.irreversibility([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), 1, 1e-12);
  });
  it('dobrushin coefficient', () => {
    near(markov.dobrushin([[0.9, 0.1], [0.5, 0.5]]), 0.4, 1e-12);
    near(markov.dobrushin([[1, 0], [0, 1]]), 1, 1e-12);
    near(markov.dobrushin([[0.3, 0.7], [0.3, 0.7]]), 0, 1e-12);
  });
});

describe('markov: spectral dimension', () => {
  it('reads the dimension of lattices', () => {
    near(markov.spectralDimension(torus(200, 1), { tMin: 4, tMax: 64 }).dimension, 1, 0.1, 'ring');
    near(markov.spectralDimension(torus(30, 2), { tMin: 4, tMax: 64 }).dimension, 2, 0.1, '2-D torus');
  });
  it('adds under independent products', () => {
    near(markov.spectralDimension(markov.kron(torus(40, 1), torus(40, 1)), { tMin: 4, tMax: 64 }).dimension, 2, 0.15);
  });
  it('dimension adds when ring agents are combined with ⊗', () => {
    let agent = ringAgent(10);
    const dims = [];
    for (let k = 1; k <= 3; k++) {
      if (k > 1) agent = FormalConsciousAgent.combine(agent, ringAgent(10));
      dims.push(markov.spectralDimension(agent.jointKernel().matrix, { tMin: 2, tMax: 10 }).dimension);
    }
    dims.forEach((d, i) => near(d, i + 1, 0.2, `${i + 1} agents`));
  });
  it('interaction binds dimensions (stationary starts)', () => {
    const d = (c) => {
      const Q = coupledRings(8, c);
      const pi = markov.stationary(Q).pi;
      return markov.spectralDimension(Q, { tMin: 2, tMax: 8, startDistribution: 'stationary', pi, maxStarts: 64, lazy: false }).dimension;
    };
    assert.ok(d(0) > 2.4 && d(0.9) < 1.5, `${d(0)} -> ${d(0.9)}`);
  });
  it('an agent reconstructs a hidden world\'s dimension from opaque tokens', () => {
    const { kernel, truth } = learnedKernel(12, 2, { steps: 60 * 144, seed: 12 });
    const w = { tMin: 2, tMax: 12 };
    near(markov.spectralDimension(kernel.matrix, w).dimension, markov.spectralDimension(truth, w).dimension, 0.1);
  });
  it('rejects unknown start distributions', () => {
    assert.throws(() => markov.returnProbabilities(torus(5, 1), { startDistribution: 'x' }), /unknown/);
  });
});

describe('bell', () => {
  const s = Math.SQRT1_2;
  it('classifies reference behaviours', () => {
    assert.equal(bell.classify(bell.fromCorrelators([[1, 1], [1, 1]])), 'local');
    assert.equal(bell.classify(bell.fromCorrelators([[s, s], [s, -s]])), 'quantum');
    assert.equal(bell.classify(bell.fromCorrelators([[1, 1], [1, -1]])), 'post-quantum');
    near(bell.chsh([[s, s], [s, -s]]), bell.TSIRELSON, 1e-12);
  });
  it('detects signalling', () => {
    // Bob's outcome copies Alice's setting.
    const p = [0, 1].map(a => [0, 1].map(() => [0, 1].map(() => [0, 1].map(y => (y === a ? 0.5 : 0)))));
    near(bell.signalling(p), 1, 1e-12);
    assert.equal(bell.classify(p), 'signalling');
  });
});

describe('experiment 13: Bell test through the headset', () => {
  it('spacetime-local agents never exceed the Bell bound', () => {
    for (const [seed, st] of [[100, 0], [101, 0.5]]) {
      const net = hiddenNetwork(seed, st);
      for (const k of [0, 1]) {
        assert.ok(optimiseChsh(markov.matPow(net.Q, k), net.pi, { actionEntersNetwork: false }).chsh <= 2 + 1e-9);
      }
    }
  });
  it('best CHSH equals 2 + 2·δ(Q^k) behind the headset', () => {
    for (const [seed, st] of [[100, 0], [102, 0.8], [7, 0.3]]) {
      const net = hiddenNetwork(seed, st);
      for (const k of [0, 1, 3, 6]) {
        const Mk = markov.matPow(net.Q, k);
        const bound = 2 + 2 * markov.dobrushin(Mk);
        const { chsh } = optimiseChsh(Mk, net.pi, { actionEntersNetwork: true, seed: 7 + k });
        assert.ok(chsh <= bound + 1e-9, `bound violated: ${chsh} > ${bound}`);
        near(chsh, bound, 1e-6, `not attained (seed ${seed}, k ${k})`);
      }
    }
  });
});
