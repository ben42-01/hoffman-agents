const { describe, it } = require('node:test');
const assert = require('node:assert');
const { ConsciousAgent, combine, fuse, productKernel, experienceSpaceDistance } = require('../src');
const { serialize, deserialize } = require('../src/io');
const markov = require('../src/math/markov');
const { homeWorld } = require('./helpers/worlds');
const fs = require('fs');
const os = require('os');
const path = require('path');

const trained = (id, seed, steps = 900) => {
  const a = new ConsciousAgent({ agentId: id, seed, world: homeWorld(seed, 0.9, 8) });
  a.run(steps);
  return a;
};
const metaIds = (a) => a.experience.metaTrie.transitionCounts().ids;
const metaCounts = (a) => a.experience.metaTrie.transitionCounts().counts.map(r => Array.from(r));

describe('combination ⊗ (v3)', () => {
  const A = trained('A', 1), B = trained('B', 2), C = trained('C', 3);

  it('identity is commutative and associative', () => {
    assert.equal(combine(A, B).agentId, combine(B, A).agentId);
    assert.equal(combine(combine(A, B), C).agentId, combine(A, combine(B, C)).agentId);
    assert.equal(combine(A, B, C).agentId, combine(combine(A, B), C).agentId);
  });

  it('is symmetric in its learned structure', () => {
    const AB = combine(A, B), BA = combine(B, A);
    assert.deepEqual([...AB.experience.metaTrie._registry.keys()].sort(), [...BA.experience.metaTrie._registry.keys()].sort());
    assert.deepEqual([...AB.experience.traceBuffer].map(e => e.toState), [...BA.experience.traceBuffer].map(e => e.toState));
    assert.equal(AB.pStable, BA.pStable);
  });

  it('nested combination has collision-free provenance', () => {
    const ABC = combine(combine(A, B), C);
    const prov = ABC.experience.metaTrie._provenance;
    const expected = ABC.experience.metaTrie._registry.size;
    assert.ok(prov.size >= expected);
    const keys = new Set([...prov.values()].map(p => `${p.constituentId}:${p.localId}`));
    assert.equal(keys.size, prov.size, 'every (constituent, local id) maps to a distinct joint id');
  });

  it('combined agent starts unlocked with an empty native meta-chain', () => {
    assert.ok(A.isILocked && B.isILocked);
    const AB = combine(A, B);
    assert.equal(AB.isILocked, false);
    assert.equal(AB.experience.metaTrie.ergodicDiagnostics().classSize, 0);
  });

  it('product kernel has stationary distribution pi_A ⊗ pi_B', () => {
    const K = productKernel(A, B);
    assert.ok(K);
    assert.ok(markov.isStochastic(K.matrix));
    const piA = A.experience.metaTrie.ergodicDiagnostics().pi;
    const piB = B.experience.metaTrie.ergodicDiagnostics().pi;
    const piK = K.stationary().distribution;
    const [a0] = piA.keys(), [b0] = piB.keys();
    assert.ok(Math.abs(piK[`${a0}|${b0}`] - piA.get(a0) * piB.get(b0)) < 1e-6);
  });

  it('weighted combination mixes decision kernels convexly', () => {
    const x = new ConsciousAgent({ agentId: 'x', pStable: 0.9, pLexicon: 0.05, pExplore: 0.05 });
    const y = new ConsciousAgent({ agentId: 'y', pStable: 0.5, pLexicon: 0.3, pExplore: 0.1 });
    const xy = combine(x, { other: y, weights: [3, 1] });
    assert.ok(Math.abs(xy.pStable - 0.8) < 1e-12);
    assert.ok(markov.isStochastic(xy.decisionKernel.matrix));
    assert.throws(() => combine(x, { other: y, weights: [-1, 1] }), /non-negative/);
  });
});

describe('fuse (v3)', () => {
  const A = trained('A', 1), B = trained('B', 2), C = trained('C', 3);

  it('fuse(A ⊗ B) restores both constituents exactly', () => {
    const parts = fuse(combine(A, B));
    assert.deepEqual(parts.map(p => p.agentId), ['A', 'B']);
    for (const [part, orig] of [[parts[0], A], [parts[1], B]]) {
      assert.deepEqual(metaIds(part), metaIds(orig));
      assert.deepEqual(metaCounts(part), metaCounts(orig));
      assert.equal(part.isILocked, orig.isILocked);
      assert.equal(part.experience.selfToken.referentMetaStateId, orig.experience.selfToken.referentMetaStateId);
      assert.equal(part.experience.metaTrie.lastMetaState, orig.experience.metaTrie.lastMetaState);
    }
    assert.notEqual(parts[0].experience.lexicon, parts[1].experience.lexicon);
  });

  it('fuse of a nested agent returns its direct constituents, recursively', () => {
    const AB = combine(A, B);
    const ABC = combine(AB, C);
    const parts = fuse(ABC);
    const ab = parts.find(p => p.agentId === AB.agentId);
    assert.ok(ab && parts.some(p => p.agentId === 'C'));
    assert.deepEqual([...ab.constituentIds].sort(), ['A', 'B']);
    const [a2, b2] = fuse(ab);
    assert.deepEqual(metaCounts(a2), metaCounts(A));
    assert.deepEqual(metaCounts(b2), metaCounts(B));
  });

  it('provenance survives serialization', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ca-fuse-'));
    try {
      const file = path.join(dir, 'ab.soul');
      serialize(combine(A, B), file);
      const [a2] = fuse(deserialize(file));
      assert.equal(a2.agentId, 'A');
      assert.deepEqual(metaCounts(a2), metaCounts(A));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('experienceSpaceDistance', () => {
  it('kernel mode is 0 for identical and in [0, 1] for different experience', () => {
    const A = trained('A', 1, 400), B = trained('B', 5, 400);
    assert.equal(experienceSpaceDistance(A.experience, A.experience, { mode: 'kernel' }), 0);
    const d = experienceSpaceDistance(A.experience, B.experience, { mode: 'kernel' });
    assert.ok(d >= 0 && d <= 1);
    assert.throws(() => experienceSpaceDistance(A.experience, B.experience, { mode: 'nope' }), /unknown mode/);
  });
});
