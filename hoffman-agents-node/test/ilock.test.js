const { describe, it } = require('node:test');
const assert = require('node:assert');
const { ConsciousAgent, MetaTrie, SelfTokenState, WorldState } = require('../src');
const { homeWorld, noiseWorld, switchingWorld } = require('./helpers/worlds');

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);

function metaTrieFrom(edges, history) {
  const mt = new MetaTrie();
  for (const [a, b] of edges) mt._trie.insert([a, b]);
  mt._history = history;
  mt._lastMetaState = history[history.length - 1];
  return mt;
}

describe('meta-state chain (v3)', () => {
  it('the newest meta-state is not an absorbing sink', () => {
    // 2.x gave pi = [0, 0, 1] here: all mass on the unobserved newest state.
    const d = metaTrieFrom([[1, 2], [2, 3]], [1, 2, 3]).ergodicDiagnostics();
    assert.equal(d.classSize, 0);
    assert.equal(d.dominant, null);
  });

  it('periodic chains give the correct stationary distribution and report their period', () => {
    const d = metaTrieFrom([[1, 2], [2, 1], [3, 1]], [3, 1, 2, 1]).ergodicDiagnostics();
    assert.deepEqual(d.states, [1, 2]);
    close(d.pi.get(1), 0.5); close(d.pi.get(2), 0.5);
    assert.equal(d.period, 2);
    assert.equal(d.aperiodic, false);
  });

  it('self-transitions count as dwell time', () => {
    const d = metaTrieFrom([[1, 1], [1, 1], [1, 2], [2, 1]], [1, 1, 1, 2, 1]).ergodicDiagnostics();
    close(d.pi.get(1), 0.75, 1e-9);
    assert.equal(d.dominant, 1);
    assert.equal(d.nTransitions, 4);
  });

  it('meta-state ids do not depend on the lock flag', () => {
    const mt = new MetaTrie();
    assert.equal(mt._computeMetaStateId([1, 2, 3], 0.1, 'core', false), mt._computeMetaStateId([1, 2, 3], 0.1, 'core', true));
  });

  it('observeSelf records repeated meta-states as self-loops', () => {
    const agent = new ConsciousAgent({ agentId: 'const', seed: 1, world: { step: () => WorldState.fromSequence('world', ['x']) } });
    agent.run(200);
    const mt = agent.experience.metaTrie;
    const id = mt.lastMetaState;
    assert.ok(mt.trie.lookup([id, id]).visitCount > 0);
  });
});

describe('"I" lock (v3)', () => {
  it('does not lock in a structureless world (null test)', () => {
    let locks = 0;
    for (let seed = 1; seed <= 20; seed++) {
      const a = new ConsciousAgent({ agentId: `n${seed}`, seed, world: noiseWorld(seed) });
      a.run(3000);
      if (a.experience.selfToken.lockHistory.length > 0) locks++;
    }
    assert.ok(locks / 20 <= 0.05, `false lock rate ${locks}/20`);
  });

  it('locks onto a dominant experiential attractor (positive test)', () => {
    let locks = 0;
    for (let seed = 1; seed <= 20; seed++) {
      const a = new ConsciousAgent({ agentId: `h${seed}`, seed, world: homeWorld(seed, 0.9) });
      a.run(3000);
      if (a.isILocked) locks++;
    }
    assert.ok(locks / 20 >= 0.95, `lock rate ${locks}/20`);
  });

  it('never locks before the evidence threshold', () => {
    const a = new ConsciousAgent({ agentId: 'early', seed: 1, world: homeWorld(1, 0.99) });
    a.run(61); // 2.x locked here
    assert.equal(a.isILocked, false);
  });

  it('unlocks when the attractor disappears and emits lock/unlock interrupts', () => {
    let unlocked = 0;
    for (let seed = 1; seed <= 10; seed++) {
      const a = new ConsciousAgent({ agentId: `s${seed}`, seed, world: switchingWorld(seed, 1500) });
      const events = [];
      for (let i = 0; i < 6000; i++) {
        const out = a.step();
        if (out.interrupt) events.push(out.interrupt);
      }
      assert.equal(events[0].event, 'lock');
      assert.ok(events[0].generation < 75, 'first lock happens during the attractor phase');
      if (events.some(e => e.event === 'unlock')) unlocked++;
      assert.deepEqual(a.experience.selfToken.lockHistory, events);
    }
    assert.ok(unlocked >= 8, `unlocked ${unlocked}/10`);
  });

  it('records the generation, not the step, at lock time', () => {
    const a = new ConsciousAgent({ agentId: 'gen', seed: 2, world: homeWorld(2, 0.95) });
    a.run(2000);
    const lock = a.experience.selfToken.lockHistory[0];
    assert.ok(lock && lock.generation < 100, `lockGeneration ${lock && lock.generation}`);
    assert.equal(a.experience.selfToken.lockGeneration, lock.generation);
  });

  it('exposes the lock criteria through ergodicStats()', () => {
    const a = new ConsciousAgent({ agentId: 'stats', seed: 3, world: homeWorld(3, 0.95) });
    a.run(1500);
    const s = a.ergodicStats();
    assert.equal(s.mathVersion, 'v3');
    assert.equal(s.decision.ergodic, true);
    assert.ok(s.meta.classSize > 0);
    assert.deepEqual(Object.keys(s.lock.criteria).sort(), ['dominance', 'ergodic', 'evidence', 'occupancy', 'stable']);
  });

  it('lock options pass through fromConfig and serialize', () => {
    const a = ConsciousAgent.fromConfig('cfg', { agent: { selfToken: { lockMargin: 0.3, minTransitions: 40 }, seed: 5 } });
    assert.equal(a.experience.selfToken.lockMargin, 0.3);
    assert.equal(a.experience.selfToken.minTransitions, 40);
    const st = SelfTokenState.fromJSON(a.experience.selfToken.toJSON());
    assert.equal(st.lockMargin, 0.3);
    assert.equal(st.mathVersion, 'v3');
  });
});
