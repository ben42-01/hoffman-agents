// Cross-language parity trace. Mirrors hoffman-agents-python/tests/parity/run.py
// line for line; both print the same JSON for the same scenarios.
const path = require('path');
const { ConsciousAgent, WorldState, combine, fuse } = require(path.join(__dirname, '..', '..', 'src'));
const { mulberry32 } = require(path.join(__dirname, '..', '..', 'src', 'math', 'rng'));

function homeWorld(seed, pHome, n) {
  const r = mulberry32(seed + 999);
  return { step: () => WorldState.fromSequence('world', [r() < pHome ? 'home' : `s${Math.floor(r() * n)}`]) };
}

function noiseWorld(seed, n) {
  const r = mulberry32(seed + 5000);
  return { step: () => WorldState.fromSequence('world', [`t${Math.floor(r() * n)}`, `u${Math.floor(r() * 3)}`]) };
}

function trace(agent, steps) {
  const outputs = [];
  for (let i = 0; i < steps; i++) {
    const o = agent.step();
    outputs.push([o.sequenceStr, agent.experience.metaTrie.lastMetaState, o.iLocked, Math.round(o.predictionError * 1e9) / 1e9]);
  }
  const d = agent.experience.metaTrie.ergodicDiagnostics();
  return {
    outputs,
    pi: [...d.pi].map(([id, p]) => [id, p]),
    period: d.period,
    nTransitions: d.nTransitions,
    lockHistory: agent.experience.selfToken.lockHistory.map(e => [e.event, e.generation, e.referent]),
    registry: [...agent.experience.metaTrie._registry.keys()].sort((a, b) => a - b),
    lexicon: [...agent.experience.lexicon._entries.values()].map(e => [e.label, e.outputToken]),
    trieSize: agent.experience.trie.size(),
  };
}

const result = {};
for (const seed of [1, 2, 3]) {
  result[`home_${seed}`] = trace(new ConsciousAgent({ agentId: `H${seed}`, seed, world: homeWorld(seed, 0.9, 20) }), 1500);
  result[`noise_${seed}`] = trace(new ConsciousAgent({ agentId: `N${seed}`, seed, world: noiseWorld(seed, 40) }), 1500);
}

const a = new ConsciousAgent({ agentId: 'A', seed: 11, world: homeWorld(11, 0.9, 8) });
const b = new ConsciousAgent({ agentId: 'B', seed: 12, world: homeWorld(12, 0.95, 8) });
a.run(900); b.run(900);
const ab = combine(b, a);
result.combine = {
  id: ab.agentId,
  provenance: [...ab.experience.metaTrie._provenance].map(([id, p]) => [id, p.constituentId, p.localId]).sort((x, y) => x[0] - y[0]),
  params: [ab.pStable, ab.pLexicon, ab.pExplore],
  trace: [...ab.experience.traceBuffer].map(e => e.toState),
  fused: fuse(ab).map(p => [p.agentId, p.experience.metaTrie.transitionCounts().ids, p.isILocked]),
};

process.stdout.write(JSON.stringify(result));
