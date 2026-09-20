const { describe, it } = require('node:test');
const assert = require('node:assert');
const crypto = require('crypto');
const path = require('path');
const { ConsciousAgent, SimpleWorld, combine, fuse } = require('../src');
const { deserialize } = require('../src/io');
const { mulberry32 } = require('../src/math/rng');
const golden = require('./fixtures/legacy-2.1.2.json');

describe("mathVersion: 'legacy'", () => {
  for (const g of golden) {
    it(`reproduces 2.1.2 output exactly (seed ${g.seed})`, () => {
      const agent = new ConsciousAgent({
        agentId: `CA_legacy_${g.seed}`, rng: mulberry32(g.seed),
        world: new SimpleWorld({ nStates: 50, seed: g.seed }), mathVersion: 'legacy',
      });
      const h = crypto.createHash('sha256');
      let lockStep = null;
      for (let i = 0; i < 300; i++) {
        const out = agent.step();
        h.update(out.sequenceStr + '|');
        if (out.iLocked && lockStep === null) lockStep = out.step;
      }
      assert.equal(h.digest('hex'), g.outputHash);
      assert.equal(lockStep, g.lockStep);
      assert.equal(agent.experience.metaTrie.registrySize, g.registrySize);
      assert.equal(agent.experience.selfToken.stationaryProb, g.stationaryProb);
      assert.equal(agent.experience.lexicon.entryCount, g.lexiconSize);
    });
  }

  it('loads a 2.x .soul file as legacy', () => {
    const agent = deserialize(path.join(__dirname, 'fixtures', 'legacy-2.1.2.soul'));
    assert.equal(agent.mathVersion, 'legacy');
    assert.equal(agent.isILocked, true);
    assert.equal(agent.stepCount, 300);
    agent.run(20);
  });

  it('legacy combine/fuse keep the 2.x tagging scheme', () => {
    const mk = (id, seed) => {
      const a = new ConsciousAgent({ agentId: id, rng: mulberry32(seed), world: new SimpleWorld({ seed }), mathVersion: 'legacy' });
      a.run(100);
      return a;
    };
    const ab = combine(mk('A', 1), mk('B', 2));
    assert.equal(ab.mathVersion, 'legacy');
    assert.ok([...ab.experience.metaTrie._registry.keys()].every(id => (id >>> 28) === 1 || (id >>> 28) === 2));
    assert.equal(fuse(ab).length, 2);
  });

  it('rejects unknown math versions', () => {
    assert.throws(() => new ConsciousAgent({ mathVersion: 'v9' }), /Invalid mathVersion/);
  });
});
