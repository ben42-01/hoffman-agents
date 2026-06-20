const { describe, it } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const os = require('node:os');
const fs = require('node:fs');

const {
  ConsciousAgent, StepOutput, SimpleWorld,
  TraceBuffer, TraceEvent, ExperienceTrie, MetaTrie,
  SelfTokenState, ExperienceLexicon,
  combine, AgentNetwork, World, WorldBuilder, CoinTossWorld,
  SharedMeaningTracker,
} = require('../src/index');

const { serialize, deserialize, clone, saveAgent, loadAgent, cloneAgent } = require('../src/io');

describe('Core Components', () => {
  it('TraceBuffer', () => {
    const buf = new TraceBuffer(5);
    assert.equal(buf.length, 0);
    buf.append(new TraceEvent(0, 1, 0, 1, true, 0));
    assert.equal(buf.length, 1);
    for (let i = 0; i < 10; i++) {
      buf.append(new TraceEvent(i, i + 1, i, i + 1, true, 0));
    }
    assert.equal(buf.isFull(), true);
    assert.equal(buf.length, 5);
  });

  it('ExperienceTrie', () => {
    const trie = new ExperienceTrie(5);
    trie.insert([1, 2, 3], 0.1);
    trie.insert([1, 2, 3], 0.2);
    const node = trie.lookup([1, 2, 3]);
    assert.notEqual(node, null);
    assert.equal(node.visitCount, 2);
    assert.equal(Math.abs(node.meanPredictionError - 0.15) < 0.001, true);
    assert.equal(trie.predictNext([1, 2]), 3);
  });

  it('MetaTrie', () => {
    const mt = new MetaTrie(3, 5);
    const buf = new TraceBuffer(10);
    for (let i = 0; i < 6; i++) {
      buf.append(new TraceEvent(i, i + 1, i, i + 1, true, 0.1));
    }
    const metaId = mt.observeSelf(buf, 5);
    assert.ok(metaId > 0);
    assert.ok(mt.registrySize > 0);
  });

  it('SelfTokenState', () => {
    const st = new SelfTokenState();
    assert.equal(st.locked, false);
    assert.equal(st.token, 'I');
    assert.equal(st.lockThreshold, 0.25);
  });

  it('ExperienceLexicon', () => {
    const lex = new ExperienceLexicon(16);
    const sig = new Float64Array(16).fill(0.25);
    lex.bind('test_word', sig, { outputToken: 'test' });
    assert.equal(lex.vocabularySize(), 1);
    assert.notEqual(lex.lookupByLabel('test_word'), null);
  });
});

describe('Agent', () => {
  it('SimpleWorld step', () => {
    const world = new SimpleWorld({ nStates: 5, seed: 42 });
    const state = world.step();
    assert.notEqual(state, null);
    assert.ok(state.sequences);
  });

  it('ConsciousAgent step', () => {
    const world = new SimpleWorld({ nStates: 5, seed: 42 });
    const agent = new ConsciousAgent({ agentId: 'step_test' });
    const output = agent.step(world.step());
    assert.ok(output instanceof StepOutput);
    assert.ok(output.sequence);
  });

  it('ConsciousAgent run', () => {
    const world = new SimpleWorld({ nStates: 5, seed: 42 });
    const agent = new ConsciousAgent({ agentId: 'run_test', world });
    const outputs = agent.run(100);
    assert.equal(outputs.length, 100);
  });
});

describe('Combination', () => {
  it('combine two agents', () => {
    const a = new ConsciousAgent({ agentId: 'A' });
    const b = new ConsciousAgent({ agentId: 'B' });
    const c = combine(a, b);
    assert.notEqual(c.agentId, 'A');
    assert.notEqual(c.agentId, 'B');
    assert.equal(c.cycleLevel, 1);
  });
});

describe('Network', () => {
  it('AgentNetwork', () => {
    const net = new AgentNetwork({ nAgents: 4, seed: 42 });
    const state = net.step();
    assert.equal(state.generation, 0);
    assert.ok(state.outputs);
  });
});

describe('World', () => {
  it('WorldBuilder', () => {
    const data = [];
    for (let i = 0; i < 100; i++) data.push([Math.random(), Math.random(), Math.random()]);
    const builder = new WorldBuilder();
    builder.addFeature('f1', 'minmax', 4);
    builder.addFeature('f2', 'minmax', 4);
    builder.addFeature('f3', 'minmax', 4);
    const w = builder.build(data);
    assert.ok(w.nStates > 0);
  });

  it('CoinTossWorld', () => {
    const w = new CoinTossWorld(3);
    assert.equal(w.nStates, 8);
    const state = w.step();
    assert.notEqual(state, null);
  });
});

describe('IO', () => {
  it('serialize/deserialize', () => {
    const agent = new ConsciousAgent({ agentId: 'save_test' });
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ca-test-'));
    const filePath = path.join(tmpDir, 'test.soul');
    try {
      serialize(agent, filePath);
      const loaded = deserialize(filePath);
      assert.equal(loaded.agentId, 'save_test');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('save/load/clone', () => {
    const agent = new ConsciousAgent({ agentId: 'save_test2' });
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ca-test-'));
    try {
      const p = saveAgent(agent, tmpDir);
      const loaded = loadAgent(p);
      assert.equal(loaded.agentId, 'save_test2');
      const cloned = cloneAgent(agent, 'cloned_test');
      assert.equal(cloned.agentId, 'cloned_test');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

describe('Shared Meaning', () => {
  it('SharedMeaningTracker', () => {
    const tracker = new SharedMeaningTracker();
    const lex1 = new ExperienceLexicon();
    const lex2 = new ExperienceLexicon();
    const sig = new Float64Array(64).fill(1 / Math.sqrt(64));
    lex1.bind('a', sig, { outputToken: 'foo' });
    lex2.bind('b', sig, { outputToken: 'foo' });
    const result = tracker.snapshot({ a1: lex1, a2: lex2 }, 1);
    assert.ok(result.sharedness > 0);
  });
});
