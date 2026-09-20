const crypto = require('crypto');
const { ExperienceTrie } = require('../core/experience-trie');
const { MetaTrie, ID_MASK } = require('../core/meta-trie');
const { SelfTokenState } = require('../core/self-token');
const { TraceBuffer } = require('../core/trace-buffer');
const { ExperienceLexicon } = require('../core/experience-lexicon');
const { ConsciousAgent } = require('../agent/conscious-agent');
const { ExperienceSpace } = require('../agent/experience-space');
const { MarkovKernel } = require('../kernels/markov-kernel');
const markov = require('../math/markov');
const { fnv1a32 } = require('../math/rng');
const legacy = require('../legacy/combination');

// Combination operator ⊗.
//
// v3 semantics
//  - Commutative and associative on identity: the combined id is a hash of the
//    sorted set of leaf constituents, so A⊗B = B⊗A and (A⊗B)⊗C = A⊗(B⊗C).
//  - The combined agent's own meta-chain starts empty. Each constituent's meta
//    chain is carried along under collision-free provenance ids (not 4-bit tags)
//    so fuse() can restore it exactly, at any nesting depth.
//  - Hoffman-Prakash product semantics: until the combined agent has its own
//    experience, its meta dynamics is the independent product kernel
//    M1 ⊗ M2 (exposed as combinationPrior / productKernel()).
//  - "I" starts unlocked and must re-lock on the joint chain: the constituents'
//    referents live in different state spaces, so "both locked" is not a lock.
//  - Decision parameters are a convex mixture (weights default to equal), which
//    keeps the decision kernel row-stochastic.
function combine(...agents) {
  if (agents.length === 0) return trivialAgent();
  if (agents.length === 1) return agents[0];

  const hasWeights = agents.length === 2 && agents[1] && typeof agents[1] === 'object'
    && !(agents[1] instanceof ConsciousAgent) && 'weights' in agents[1];
  if (hasWeights) {
    const [agent1, opts] = agents;
    const [w1, w2] = opts.weights;
    return _weightedCombine(agent1, opts.other || trivialAgent(), w1, w2);
  }

  if (agents.length === 2) return _combinePair(agents[0], agents[1]);

  const mid = Math.floor(agents.length / 2);
  return combine(combine(...agents.slice(0, mid)), combine(...agents.slice(mid)));
}

function _isLegacy(...agents) {
  return agents.some(a => a.mathVersion === 'legacy');
}

function _combinePair(a, b, weights = [0.5, 0.5]) {
  if (_isLegacy(a, b)) return legacy.combinePair(a, b);
  if (a.agentId === 'CA_0') return b;
  if (b.agentId === 'CA_0') return a;

  // Canonical order makes every downstream choice symmetric.
  let [agent1, agent2] = [a, b];
  let [w1, w2] = weights;
  if (b.agentId < a.agentId) { [agent1, agent2] = [b, a]; [w1, w2] = [w2, w1]; }

  const leafIds = new Set([..._leafIds(agent1), ..._leafIds(agent2)]);
  const combinedId = combinedAgentId(leafIds);

  const metaTrie = _buildJointMetaTrie([agent1, agent2]);
  const st1 = agent1.experience.selfToken, st2 = agent2.experience.selfToken;
  const selfToken = new SelfTokenState({
    ...SelfTokenState.lockOptions(st1),
    token: st1.token,
    lockConsecutiveRequired: Math.max(st1.lockConsecutiveRequired, st2.lockConsecutiveRequired),
    minTransitions: Math.max(st1.minTransitions, st2.minTransitions),
    mathVersion: 'v3',
  });

  const traceBuffer = _interleaveTraces(agent1.experience.traceBuffer, agent2.experience.traceBuffer);
  const recent = traceBuffer.getRecent(1);
  const exp = new ExperienceSpace({
    trie: agent1.experience.trie.merge(agent2.experience.trie),
    metaTrie,
    selfToken,
    lexicon: _mergeLexicons(agent1.experience.lexicon, agent2.experience.lexicon),
    traceBuffer,
    lastWorldStateId: recent.length > 0 ? recent[0].toState : null,
    mathVersion: 'v3',
  });

  const total = w1 + w2;
  const [n1, n2] = total > 0 ? [w1 / total, w2 / total] : [0.5, 0.5];
  const mix = (k) => n1 * agent1[k] + n2 * agent2[k];

  const combined = new ConsciousAgent({
    agentId: combinedId,
    experience: exp,
    generation: Math.max(agent1.generation, agent2.generation),
    metaObservationInterval: agent1.metaObservationInterval,
    constituentIds: new Set([agent1.agentId, agent2.agentId]),
    leafConstituentIds: leafIds,
    cycleLevel: Math.max(agent1.cycleLevel, agent2.cycleLevel) + 1,
    pStable: mix('pStable'),
    pLexicon: mix('pLexicon'),
    pExplore: mix('pExplore'),
    lexiconRow: agent1.lexiconRow.map((v, i) => n1 * v + n2 * agent2.lexiconRow[i]),
    rng: agent1._rng,
    mathVersion: 'v3',
  });
  combined._combined = true;

  const prior = productKernel(agent1, agent2);
  combined.combinationPrior = prior ? { states: prior.states.length, diagnostics: _priorSummary(prior) } : null;
  return combined;
}

function _weightedCombine(agent1, agent2, w1, w2) {
  if (!(w1 >= 0 && w2 >= 0)) throw new Error(`combination weights must be non-negative, got [${w1}, ${w2}]`);
  if (_isLegacy(agent1, agent2)) return legacy.weightedCombine(agent1, agent2, w1, w2);
  if (agent2.agentId === 'CA_0') return agent1;
  if (agent1.agentId === 'CA_0') return agent2;
  return _combinePair(agent1, agent2, [w1, w2]);
}

function trivialAgent() {
  return new ConsciousAgent({ agentId: 'CA_0' });
}

function _leafIds(agent) {
  return agent.leafConstituentIds.size > 0 ? agent.leafConstituentIds : new Set([agent.agentId]);
}

function combinedAgentId(leafIds) {
  const key = JSON.stringify([...leafIds].sort());
  return `CA_${crypto.createHash('sha256').update(key).digest('hex').slice(0, 12)}`;
}

// Joint id for a constituent's local meta-state id: deterministic, 28-bit, and
// guaranteed unique within the joint trie (linear re-hash on collision).
function _provenanceId(taken, constituentId, localId) {
  let id = fnv1a32(`${constituentId}:${localId}`) & ID_MASK;
  for (let salt = 1; taken.has(id); salt++) id = fnv1a32(`${constituentId}:${localId}#${salt}`) & ID_MASK;
  return id;
}

function _provenanceSnapshot(agent) {
  const mt = agent.experience.metaTrie;
  return {
    agentId: agent.agentId,
    provenance: [...mt._provenance].map(([id, p]) => [id, { ...p }]),
    tree: mt._provenanceTree,
    history: [...mt._history],
    lastMetaState: mt.lastMetaState,
    selfToken: agent.experience.selfToken.toJSON(),
    constituentIds: [...agent.constituentIds].sort(),
    leafConstituentIds: [...agent.leafConstituentIds].sort(),
    cycleLevel: agent.cycleLevel,
    params: { pStable: agent.pStable, pLexicon: agent.pLexicon, pExplore: agent.pExplore, lexiconRow: [...agent.lexiconRow] },
    combinationPrior: agent.combinationPrior,
  };
}

function _buildJointMetaTrie(agents) {
  const joint = new MetaTrie(
    Math.max(...agents.map(a => a.experience.metaTrie._snapshotWindow)),
    Math.max(...agents.map(a => a.experience.metaTrie.trie.maxDepth)),
    { mathVersion: 'v3' },
  );
  const taken = new Set();

  for (const agent of agents) {
    const mt = agent.experience.metaTrie;
    const cid = agent.agentId;
    const localIds = new Set([...mt._registry.keys(), ...mt._tokenRegistry.keys()]);
    for (const [from, node] of Object.entries(mt.trie.root.children)) {
      localIds.add(parseInt(from));
      for (const to of Object.keys(node.children)) localIds.add(parseInt(to));
    }
    const map = new Map();
    for (const local of [...localIds].sort((x, y) => x - y)) {
      const id = _provenanceId(taken, cid, local);
      taken.add(id);
      map.set(local, id);
      joint._provenance.set(id, { constituentId: cid, localId: local });
    }

    for (const [local, snap] of mt._registry) joint._registry.set(map.get(local), snap);
    for (const [local, counts] of mt._tokenRegistry) joint._tokenRegistry.set(map.get(local), new Map(counts));
    for (const [from, node] of Object.entries(mt.trie.root.children)) {
      for (const [to, child] of Object.entries(node.children)) {
        const path = [map.get(parseInt(from)), map.get(parseInt(to))];
        joint._trie.insert(path);
        const dst = joint._trie.lookup(path);
        dst.visitCount = child.visitCount;
        dst.predictionErrors = [...child.predictionErrors];
        dst.meanPredictionError = child.meanPredictionError;
      }
    }
    joint._provenanceTree[cid] = _provenanceSnapshot(agent);
  }
  return joint;
}

function _interleaveTraces(buf1, buf2) {
  const tagged = [
    ...[...buf1].map((e, i) => [e, 0, i]),
    ...[...buf2].map((e, i) => [e, 1, i]),
  ];
  tagged.sort((x, y) => (x[0].timestamp - y[0].timestamp) || (x[1] - y[1]) || (x[2] - y[2]));
  const out = new TraceBuffer(Math.max(buf1.maxlen, buf2.maxlen));
  for (const [e] of tagged) out.append(e);
  return out;
}

// Same label in both lexicons: keep the more integrated entry, pool encounters.
function _mergeLexicons(lex1, lex2) {
  const merged = new ExperienceLexicon(
    Math.max(lex1._embeddingDim, lex2._embeddingDim),
    Math.min(lex1._associationThreshold, lex2._associationThreshold),
  );
  const byLabel = new Map();
  for (const entry of [...lex1._entries.values(), ...lex2._entries.values()]) {
    const prev = byLabel.get(entry.label);
    if (!prev) { byLabel.set(entry.label, { entry, encounters: entry.encounterCount }); continue; }
    const encounters = prev.encounters + entry.encounterCount;
    byLabel.set(entry.label, { entry: entry.integrationDepth > prev.entry.integrationDepth ? entry : prev.entry, encounters });
  }
  for (const [label, { entry, encounters }] of byLabel) {
    merged.bind(label, entry.traceSignature, {
      predictionErrorPeak: entry.predictionErrorPeak,
      source: entry.labelingSource,
      generation: entry.generationBound,
      step: entry.stepBound,
      outputToken: entry.outputToken,
    });
    const e = merged._entries.get(label);
    e.integrationDepth = entry.integrationDepth;
    e.encounterCount = encounters;
  }
  return merged;
}

// The agent's empirical meta-state kernel on its recurrent class, or null.
function metaKernel(agent) {
  const mt = agent.experience.metaTrie;
  const d = mt.ergodicDiagnostics();
  if (d.classSize === 0) return null;
  const { ids, counts } = mt.transitionCounts();
  const pos = new Map(ids.map((id, i) => [id, i]));
  const idx = d.states.map(s => pos.get(s));
  return new MarkovKernel({ states: d.states, matrix: markov.normalizeRows(markov.subMatrix(counts, idx)).P });
}

// Hoffman-Prakash combination of two independent agents' dynamics:
// (M1 ⊗ M2)((x1,x2),(y1,y2)) = M1(x1,y1) M2(x2,y2). Its stationary
// distribution is pi1 ⊗ pi2, and it is ergodic iff both factors are.
function productKernel(a, b) {
  const k1 = metaKernel(a), k2 = metaKernel(b);
  if (!k1 || !k2) return null;
  if (k1.states.length * k2.states.length > 4096) return null;
  return k1.tensor(k2);
}

function _priorSummary(kernel) {
  const d = kernel.diagnostics();
  const pi = Object.values(d.stationary);
  return { ergodic: d.ergodic, period: d.period, lambda2: d.lambda2, entropy: d.entropy, dominantProb: Math.max(...pi) };
}

// fuse: inverse of combine. Splits an agent into its direct constituents,
// restoring each one's meta-chain, provenance, lock state and parameters.
// Joint experience acquired after combination lives in the world trie and
// lexicon (copied into each part); the joint meta-chain is not attributable to
// either constituent and is dropped.
function fuse(agent) {
  if (agent.constituentIds.size === 0) return [agent];
  if (agent.mathVersion === 'legacy') return legacy.fuse(agent);

  const joint = agent.experience.metaTrie;
  return [...agent.constituentIds].sort().map((cid) => {
    const snap = joint._provenanceTree[cid];
    if (!snap) throw new Error(`fuse: no provenance recorded for constituent ${cid}`);

    const mt = new MetaTrie(joint._snapshotWindow, joint.trie.maxDepth, { mathVersion: 'v3' });
    const toLocal = new Map();
    for (const [id, p] of joint._provenance) if (p.constituentId === cid) toLocal.set(id, p.localId);

    for (const [id, s] of joint._registry) if (toLocal.has(id)) mt._registry.set(toLocal.get(id), s);
    for (const [id, counts] of joint._tokenRegistry) if (toLocal.has(id)) mt._tokenRegistry.set(toLocal.get(id), new Map(counts));
    for (const [from, node] of Object.entries(joint.trie.root.children)) {
      const f = parseInt(from);
      if (!toLocal.has(f)) continue;
      for (const [to, child] of Object.entries(node.children)) {
        const t = parseInt(to);
        if (!toLocal.has(t)) continue;
        const path = [toLocal.get(f), toLocal.get(t)];
        mt._trie.insert(path);
        const dst = mt._trie.lookup(path);
        dst.visitCount = child.visitCount;
        dst.predictionErrors = [...child.predictionErrors];
        dst.meanPredictionError = child.meanPredictionError;
      }
    }
    mt._provenance = new Map(snap.provenance.map(([id, p]) => [id, { ...p }]));
    mt._provenanceTree = snap.tree;
    mt._history = [...snap.history];
    mt._lastMetaState = snap.lastMetaState;

    const exp = new ExperienceSpace({
      trie: agent.experience.trie.merge(new ExperienceTrie(agent.experience.trie.maxDepth)),
      metaTrie: mt,
      selfToken: SelfTokenState.fromJSON(snap.selfToken),
      lexicon: agent.experience.lexicon.clone(),
      traceBuffer: TraceBuffer.fromEvents([...agent.experience.traceBuffer], agent.experience.traceBuffer.maxlen),
      lastWorldStateId: agent.experience.lastWorldStateId,
      mathVersion: 'v3',
    });

    const part = new ConsciousAgent({
      agentId: cid,
      experience: exp,
      generation: agent.generation,
      metaObservationInterval: agent.metaObservationInterval,
      constituentIds: new Set(snap.constituentIds),
      leafConstituentIds: new Set(snap.leafConstituentIds),
      cycleLevel: snap.cycleLevel,
      ...snap.params,
      rng: agent._rng,
      mathVersion: 'v3',
    });
    part.combinationPrior = snap.combinationPrior;
    part._combined = snap.constituentIds.length > 0;
    return part;
  });
}

// Distance between two experience spaces.
//  mode 'jaccard' (2.x default): 0.6 * path-set Jaccard + 0.4 * lexicon-label Jaccard.
//  mode 'kernel': weights.kernel * mean total-variation distance between the two
//    empirical world-transition kernels on shared source states (1 if none are
//    shared) + weights.lexicon * lexicon-label Jaccard.
function experienceSpaceDistance(exp1, exp2, { mode = 'jaccard', weights = null } = {}) {
  const l1 = new Set(exp1.lexicon._entries.keys());
  const l2 = new Set(exp2.lexicon._entries.keys());
  const unionL = new Set([...l1, ...l2]);
  const lexiconDist = unionL.size === 0 ? 0 : 1 - [...l1].filter(x => l2.has(x)).length / unionL.size;

  if (mode === 'kernel') {
    const w = { kernel: 0.6, lexicon: 0.4, ...(weights || {}) };
    return w.kernel * kernelDistance(exp1.trie, exp2.trie) + w.lexicon * lexiconDist;
  }
  if (mode !== 'jaccard') throw new Error(`experienceSpaceDistance: unknown mode "${mode}"`);

  const w = { trie: 0.6, lexicon: 0.4, ...(weights || {}) };
  const p1 = new Set(exp1.trie.getAllPaths(0).map(p => p.join(',')));
  const p2 = new Set(exp2.trie.getAllPaths(0).map(p => p.join(',')));
  const union = new Set([...p1, ...p2]);
  if (union.size === 0) return 0;
  const intersection = [...p1].filter(x => p2.has(x)).length;
  return (1 - intersection / union.size) * w.trie + lexiconDist * w.lexicon;
}

function kernelDistance(trie1, trie2) {
  const rowOf = (trie, from) => {
    const node = trie.root.children[from];
    if (!node) return null;
    const row = new Map();
    let total = 0;
    for (const [to, child] of Object.entries(node.children)) {
      if (child.visitCount > 0) { row.set(to, child.visitCount); total += child.visitCount; }
    }
    if (total === 0) return null;
    for (const [k, v] of row) row.set(k, v / total);
    return row;
  };
  let sum = 0, shared = 0;
  for (const from of Object.keys(trie1.root.children)) {
    const r1 = rowOf(trie1, from), r2 = rowOf(trie2, from);
    if (!r1 || !r2) continue;
    const keys = new Set([...r1.keys(), ...r2.keys()]);
    let tv = 0;
    for (const k of keys) tv += Math.abs((r1.get(k) || 0) - (r2.get(k) || 0));
    sum += tv / 2;
    shared++;
  }
  return shared === 0 ? 1 : sum / shared;
}

module.exports = {
  combine, trivialAgent, experienceSpaceDistance, fuse,
  productKernel, metaKernel, combinedAgentId, kernelDistance,
  _splitMetaTrie: legacy._splitMetaTrie,
};
