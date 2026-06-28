const crypto = require('../crypto');
const { ExperienceTrie } = require('../core/experience-trie');
const { MetaTrie } = require('../core/meta-trie');
const { SelfTokenState } = require('../core/self-token');
const { TraceBuffer, TraceEvent } = require('../core/trace-buffer');
const { ExperienceLexicon } = require('../core/experience-lexicon');
const { ConsciousAgent } = require('../agent/conscious-agent');
const { ExperienceSpace } = require('../agent/experience-space');

function combine(...agents) {
  if (agents.length === 0) return trivialAgent();
  if (agents.length === 1) return agents[0];

  const hasWeights = agents.length === 2 && agents[1] && typeof agents[1] === 'object' && !(agents[1] instanceof ConsciousAgent) && 'weights' in agents[1];
  
  if (hasWeights) {
    const [agent1, opts] = agents;
    const agent2 = opts.other;
    const w1 = opts.weights[0];
    const w2 = opts.weights[1];
    return _weightedCombine(agent1, agent2 || opts.other || trivialAgent(), w1, w2);
  }

  if (agents.length === 2) {
    const [agent1, agent2] = agents;
    if (agent1.agentId === 'CA_0') return agent2;
    if (agent2.agentId === 'CA_0') return agent1;

    const combinedId = _hashAgentIds(agent1.agentId, agent2.agentId);
    const mergedTrie = agent1.experience.trie.merge(agent2.experience.trie);

    const leafIds1 = agent1.leafConstituentIds.size > 0 ? agent1.leafConstituentIds : new Set([agent1.agentId]);
    const leafIds2 = agent2.leafConstituentIds.size > 0 ? agent2.leafConstituentIds : new Set([agent2.agentId]);
    const leafConstituentIds = new Set([...leafIds1, ...leafIds2]);
    const constituentIds = new Set([agent1.agentId, agent2.agentId]);

    const jointMt = _buildJointMetaTrie(agent1.experience.metaTrie, agent2.experience.metaTrie);
    const combinedSelf = _combineAttractors(agent1.experience.selfToken, agent2.experience.selfToken);
    const mergedLexicon = _mergeLexicons(agent1.experience.lexicon, agent2.experience.lexicon);

    let initState = null;
    const traceBuf = new TraceBuffer(50);
    if (agent1.experience.lastWorldStateId !== null || agent2.experience.lastWorldStateId !== null) {
      initState = _canonicalHash(
        agent1.experience.lastWorldStateId || -1,
        agent2.experience.lastWorldStateId || -1,
      );
      traceBuf.append(new TraceEvent(-1, initState, 0, -1, false, 0.5));
    }

    const exp = new ExperienceSpace({
      trie: mergedTrie,
      metaTrie: jointMt,
      selfToken: combinedSelf,
      lexicon: mergedLexicon,
      traceBuffer: traceBuf,
      lastWorldStateId: initState,
    });

    return new ConsciousAgent({
      agentId: combinedId,
      experience: exp,
      generation: Math.max(agent1.generation, agent2.generation),
      constituentIds,
      leafConstituentIds,
      cycleLevel: Math.max(agent1.cycleLevel, agent2.cycleLevel) + 1,
    });
  }

  const mid = Math.floor(agents.length / 2);
  return combine(combine(...agents.slice(0, mid)), combine(...agents.slice(mid)));
}

function _weightedCombine(agent1, agent2, w1, w2) {
  const base = combine(agent1, agent2);
  const total = w1 + w2;
  if (total === 0) return base;
  const normW1 = w1 / total;
  const normW2 = w2 / total;
  const weightedStable = agent1.pStable * normW1 + agent2.pStable * normW2;
  const weightedLexicon = agent1.pLexicon * normW1 + agent2.pLexicon * normW2;
  const weightedExplore = agent1.pExplore * normW1 + agent2.pExplore * normW2;
  base.pStable = weightedStable;
  base.pLexicon = weightedLexicon;
  base.pExplore = weightedExplore;
  return base;
}

function trivialAgent() {
  return new ConsciousAgent({ agentId: 'CA_0' });
}

function experienceSpaceDistance(exp1, exp2) {
  const p1 = new Set(exp1.trie.getAllPaths(0).map(p => p.join(',')));
  const p2 = new Set(exp2.trie.getAllPaths(0).map(p => p.join(',')));
  const union = new Set([...p1, ...p2]);
  if (union.size === 0) return 0;
  const intersection = new Set([...p1].filter(x => p2.has(x)));
  const trieDist = 1 - intersection.size / union.size;

  const l1 = new Set(exp1.lexicon._entries.keys());
  const l2 = new Set(exp2.lexicon._entries.keys());
  const unionL = new Set([...l1, ...l2]);
  const lexiconDist = unionL.size === 0 ? 0 : 1 - new Set([...l1].filter(x => l2.has(x))).size / unionL.size;

  return trieDist * 0.6 + lexiconDist * 0.4;
}

function _hashAgentIds(id1, id2) {
  return `CA_${crypto.createHash('sha256').update(`${id1}->${id2}`).digest('hex').slice(0, 12)}`;
}

function _canonicalHash(s1, s2) {
  const sorted = [s1, s2].sort((a, b) => a - b);
  const h = crypto.createHash('sha256').update(`ws:${sorted[0]}:${sorted[1]}`).digest();
  return h.readUInt32BE(0);
}

function _buildJointMetaTrie(mt1, mt2) {
  const joint = new MetaTrie(
    Math.max(mt1._snapshotWindow, mt2._snapshotWindow),
    Math.max(mt1.trie.maxDepth, mt2.trie.maxDepth)
  );

  for (const [mid, snap] of mt1._registry) {
    joint._registry.set(mid | 0x10000000, snap);
  }
  for (const [mid, snap] of mt2._registry) {
    joint._registry.set(mid | 0x20000000, snap);
  }

  for (const [mid, counts] of mt1._tokenRegistry) {
    joint._tokenRegistry.set(mid | 0x10000000, new Map(counts));
  }
  for (const [mid, counts] of mt2._tokenRegistry) {
    joint._tokenRegistry.set(mid | 0x20000000, new Map(counts));
  }

  return joint;
}

function _combineAttractors(st1, st2) {
  return new SelfTokenState({
    token: st1.token,
    lockThreshold: Math.min(st1.lockThreshold, st2.lockThreshold),
    lockConsecutiveRequired: Math.max(st1.lockConsecutiveRequired, st2.lockConsecutiveRequired),
    stationaryProb: (st1.stationaryProb + st2.stationaryProb) / 2,
    locked: st1.locked && st2.locked,
    referentMetaStateId: st1.locked && st2.locked ? st1.referentMetaStateId : null,
    lockGeneration: st1.locked && st2.locked ? Math.max(st1.lockGeneration || 0, st2.lockGeneration || 0) : null,
  });
}

function _mergeLexicons(lex1, lex2) {
  const merged = new ExperienceLexicon(
    Math.max(lex1._embeddingDim, lex2._embeddingDim),
    Math.min(lex1._associationThreshold, lex2._associationThreshold)
  );

  for (const entry of [...lex1._entries.values(), ...lex2._entries.values()]) {
    merged.bind(entry.label, entry.traceSignature, {
      predictionErrorPeak: entry.predictionErrorPeak,
      source: entry.labelingSource,
      generation: entry.generationBound,
      step: entry.stepBound,
      outputToken: entry.outputToken,
    });
    const e = merged._entries.get(entry.label);
    if (e) {
      e.integrationDepth = entry.integrationDepth;
      e.encounterCount = entry.encounterCount;
    }
  }
  return merged;
}

module.exports = { combine, trivialAgent, experienceSpaceDistance };
