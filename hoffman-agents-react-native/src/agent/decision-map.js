const CORE_TOKENS = ['I', 'notice', 'familiar', 'different', 'wait'];
const SIGNATURE_MATCH_THRESHOLD = 0.3;

function _buildTransitionSignature(prevId, currId, embeddingDim) {
  const sig = new Float64Array(embeddingDim);
  if (prevId !== null && prevId !== undefined) {
    const h = Math.abs(hashCode(`${prevId}->${currId}`));
    sig[h % embeddingDim] = 1;
  }
  sig[currId % embeddingDim] = 1;
  let norm = 0;
  for (let i = 0; i < sig.length; i++) norm += sig[i] * sig[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < sig.length; i++) sig[i] /= norm;
  return sig;
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

const _random = () => Math.random();

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
    const querySig = _buildTransitionSignature(prevId, currId, experience.lexicon._embeddingDim);
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

function _nextState(current, pStable, pLexicon, pExplore, rng = _random) {
  const r = rng();
  if (current === 'lexicon') {
    if (r < 0.7) return 'core';
    if (r < 0.85) return 'lexicon';
    if (r < 0.95) return 'explore';
    return 'idle';
  } else {
    if (r < pStable) return 'core';
    if (r < pStable + pLexicon) return 'lexicon';
    if (r < pStable + pLexicon + pExplore) return 'explore';
    return 'idle';
  }
}

function decide(experience, {
  maxTokens = 8, pStable = 0.8, pLexicon = 0.1, pExplore = 0.05, ergodicState = null, rng = _random
} = {}) {
  if (!experience.selfToken.isLocked()) {
    return [['wait'], 'idle'];
  }

  const state = ergodicState || 'core';
  const next = _nextState(state, pStable, pLexicon, pExplore, rng);

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

module.exports = { decide, _sampleLexiconLabel, CORE_TOKENS };
