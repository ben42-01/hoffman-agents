const { MarkovKernel } = require('../kernels/markov-kernel');
const { buildTransitionSignature } = require('../math/signature');
const legacy = require('../legacy/decision');

const CORE_TOKENS = ['I', 'notice', 'familiar', 'different', 'wait'];
const SIGNATURE_MATCH_THRESHOLD = 0.3;

// Output (decision) states of the agent. Their dynamics is the Markov kernel D.
const DECISION_STATES = ['core', 'lexicon', 'explore', 'idle'];
const DEFAULT_LEXICON_ROW = [0.70, 0.15, 0.10, 0.05];

const _random = () => Math.random();

// Build the decision kernel D over DECISION_STATES.
//
//   from core / explore / idle : [pStable, pLexicon, pExplore, 1 - pStable - pLexicon - pExplore]
//   from lexicon               : lexiconRow (after speaking a word, return to the core
//                                utterance with high probability)
//
// This is the 2.x transition structure written down explicitly, so it can be
// validated and analysed (stationary distribution, period, mixing time).
function buildDecisionKernel({ pStable = 0.8, pLexicon = 0.1, pExplore = 0.05, lexiconRow = DEFAULT_LEXICON_ROW } = {}) {
  for (const [name, v] of Object.entries({ pStable, pLexicon, pExplore })) {
    if (!(v >= 0 && v <= 1)) throw new Error(`${name} must be in [0, 1], got ${v}`);
  }
  const pIdle = 1 - pStable - pLexicon - pExplore;
  if (pIdle < -1e-9) {
    throw new Error(`pStable + pLexicon + pExplore must be <= 1, got ${pStable + pLexicon + pExplore}`);
  }
  const base = [pStable, pLexicon, pExplore, Math.max(0, pIdle)];
  return new MarkovKernel({
    states: DECISION_STATES,
    matrix: [base, [...lexiconRow], base, base],
  });
}

const _kernelCache = new Map();
function _cachedKernel(params) {
  const key = `${params.pStable}|${params.pLexicon}|${params.pExplore}|${(params.lexiconRow || DEFAULT_LEXICON_ROW).join(',')}`;
  let k = _kernelCache.get(key);
  if (!k) {
    k = buildDecisionKernel(params);
    if (_kernelCache.size > 256) _kernelCache.clear();
    _kernelCache.set(key, k);
  }
  return k;
}

function _sampleLexiconLabel(experience, vocabSize = 5, rng = _random) {
  const entries = experience.lexicon.sortedByIntegration();
  if (entries.length === 0) return null;

  const metaId = experience.metaTrie.lastMetaState;
  if (metaId !== null) {
    const predicted = experience.metaTrie.predictToken(metaId, 3);
    if (predicted) {
      for (const e of entries) {
        if (e.outputToken === predicted) {
          e.encounterCount++;
          e.integrationDepth = Math.min(e.integrationDepth + 0.02, 1);
          return predicted;
        }
      }
    }
  }

  const currId = experience.lastWorldStateId;
  if (currId !== null) {
    let prevId = null;
    const recent = experience.traceBuffer.getRecent(2);
    if (recent.length >= 2) prevId = recent[recent.length - 2].toState;
    const querySig = buildTransitionSignature(prevId, currId, experience.lexicon._embeddingDim, experience.mathVersion);
    const [bestLabel, bestDist] = experience.lexicon.nearestLabel(querySig);
    if (bestLabel && bestDist < SIGNATURE_MATCH_THRESHOLD) {
      const entry = experience.lexicon.lookupByLabel(bestLabel);
      if (entry && entry.integrationDepth > 0.01) {
        entry.encounterCount++;
        entry.integrationDepth = Math.min(entry.integrationDepth + 0.02, 1);
        return entry.outputToken;
      }
    }
  }

  const weighted = entries.map(e => {
    let w = Math.max(e.integrationDepth, 0.01);
    w *= (1 + 0.1 * e.encounterCount);
    if (e.labelingSource === 'adopted') w *= 3;
    return [w, e];
  });
  weighted.sort((a, b) => b[0] - a[0]);
  const topN = weighted.slice(0, vocabSize);
  const total = topN.reduce((s, [w]) => s + w, 0);
  let r = rng() * total, cumulative = 0;
  for (const [w, entry] of topN) {
    cumulative += w;
    if (r <= cumulative) return entry.outputToken;
  }
  return topN.length > 0 ? topN[topN.length - 1][1].outputToken : null;
}

function decide(experience, {
  maxTokens = 8, pStable = 0.8, pLexicon = 0.1, pExplore = 0.05, ergodicState = null, rng = _random,
  lexiconRow = DEFAULT_LEXICON_ROW, kernel = null,
} = {}) {
  if (!experience.selfToken.isLocked()) {
    return [['wait'], 'idle'];
  }

  const state = ergodicState || 'core';
  const next = experience.mathVersion === 'legacy'
    ? legacy.nextState(state, pStable, pLexicon, pExplore, rng)
    : (kernel || _cachedKernel({ pStable, pLexicon, pExplore, lexiconRow })).sample(state, rng);

  let tokens;
  if (next === 'core') {
    tokens = [experience.selfToken.token, 'notice'];
    const snapshot = experience.metaTrie.lastMetaState !== null
      ? experience.metaTrie.getMetaStateSnapshot(experience.metaTrie.lastMetaState) : null;
    tokens.push(experience.selfToken.token);
    tokens.push(snapshot && snapshot.meanPredictionError > 0.3 ? 'different' : 'familiar');
  } else if (next === 'lexicon') {
    const label = _sampleLexiconLabel(experience, 5, rng);
    tokens = [experience.selfToken.token, 'notice', label || 'familiar'];
  } else if (next === 'explore') {
    tokens = [CORE_TOKENS[Math.floor(rng() * CORE_TOKENS.length)]];
  } else {
    tokens = ['wait'];
  }

  return [tokens.slice(0, maxTokens), next];
}

module.exports = {
  decide, _sampleLexiconLabel, buildDecisionKernel,
  CORE_TOKENS, DECISION_STATES, DEFAULT_LEXICON_ROW,
};
