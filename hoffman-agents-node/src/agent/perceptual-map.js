const crypto = require('crypto');
const { TraceEvent } = require('../core/trace-buffer');
const { inventToken, isInventedToken } = require('../core/token-inventor');

function _compositeId(mid1, mid2) {
  const h = crypto.createHash('sha256').update(`cp:${mid1}:${mid2}`).digest();
  return h.readUInt32BE(0);
}

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
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0;
  }
  return hash;
}

function _lookupByOutputToken(experience, token) {
  for (const entry of experience.lexicon._entries.values()) {
    if (entry.outputToken === token) return entry;
  }
  return null;
}

function _decayLexicon(experience) {
  for (const entry of experience.lexicon._entries.values()) {
    if (entry.encounterCount === 0 && entry.labelingSource !== 'adopted') {
      entry.integrationDepth *= 0.999;
    }
    if (entry.integrationDepth < 0.01) entry.integrationDepth = 0.01;
  }
}

function perceive(world, experience, step = 0, metaObservationInterval = 20, frozen = false, ergodicState = 'idle') {
  if (!world || Object.keys(world.sequences).length === 0) return experience;

  const worldStateId = world.getStateId();
  const previousId = experience.lastWorldStateId;

  let prediction = null, predictionCorrect = false, predictionError = 0.5;
  if (previousId !== null) {
    const probPred = experience.trie.predictNextProbabilistic([previousId]);
    prediction = experience.trie.predictNext([previousId]);
    predictionCorrect = prediction === worldStateId;
    if (probPred) {
      if (probPred.state === worldStateId) {
        predictionError = 1 - probPred.probability;
      } else {
        const node = experience.trie.lookup([previousId]);
        let actualProb = 0;
        if (node && node.children[worldStateId]) actualProb = node.children[worldStateId].visitCount;
        const total = node ? Object.values(node.children).reduce((s, c) => s + c.visitCount, 0) : 0;
        predictionError = total > 0 ? 1 - actualProb / total : 1;
      }
    }
  }

  const event = new TraceEvent(
    previousId !== null ? previousId : -1,
    worldStateId,
    step,
    prediction !== null ? prediction : -1,
    predictionCorrect,
    predictionError,
    null
  );

  experience.traceBuffer.append(event);

  if (!frozen) {
    experience.trie.insert([event.toState], predictionError);
    if (event.fromState >= 0) {
      experience.trie.insert([event.fromState, event.toState], predictionError);
    }
  }

  _decayLexicon(experience);

  for (const [agentId, sequence] of Object.entries(world.sequences)) {
    if (agentId === 'world') continue;
    for (const token of sequence) {
      if (!isInventedToken(token)) continue;
      const existing = _lookupByOutputToken(experience, token);
      if (existing) {
        existing.encounterCount++;
        existing.integrationDepth = Math.min(existing.integrationDepth + 0.05, 1);
        experience.lexicon.updateIntegration(existing.label, true);
      } else {
        const sig = _buildTransitionSignature(experience.lastWorldStateId, worldStateId, experience.lexicon._embeddingDim);
        const label = `adopted:${token}`;
        const entry = experience.lexicon.bind(label, sig, {
          predictionErrorPeak: predictionError,
          source: 'adopted',
          step,
          outputToken: token,
        });
        entry.integrationDepth = 0.7;
        entry.encounterCount = 1;
      }
    }
  }

  if (predictionError >= 0.3) {
    const label = `p:${worldStateId.toString(16).padStart(8, '0')}`;
    if (!experience.lexicon.lookupByLabel(label)) {
      const sig = _buildTransitionSignature(experience.lastWorldStateId, worldStateId, experience.lexicon._embeddingDim);
      const tok = inventToken();
      const entry = experience.lexicon.bind(label, sig, {
        predictionErrorPeak: predictionError,
        source: 'proto',
        step,
        outputToken: tok,
      });
      entry.integrationDepth = 0.3;
    }
  }

  if (!frozen && step > 0 && step % metaObservationInterval === 0) {
    if (experience.parentMetaTries) {
      const [pmt1, pmt2] = experience.parentMetaTries;
      const [pt1, pt2] = experience.parentTries || [];
      const buf = experience.traceBuffer;
      const recent = buf.getRecent(10);
      const stateIds = recent.map(e => e.toState);
      const prevId = experience.lastWorldStateId;

      const computeErr = (trie) => {
        if (prevId === null || !trie) return 0.5;
        const probPred = trie.predictNextProbabilistic([prevId]);
        if (!probPred) return 1;
        if (probPred.state === worldStateId) return 1 - probPred.probability;
        const node = trie.lookup([prevId]);
        let actualProb = 0;
        if (node && node.children[worldStateId]) actualProb = node.children[worldStateId].visitCount;
        const total = node ? Object.values(node.children).reduce((s, c) => s + c.visitCount, 0) : 0;
        return total > 0 ? 1 - actualProb / total : 1;
      };

      const err1 = computeErr(pt1);
      const err2 = computeErr(pt2);

      const meta1 = pmt1.observeSelf(buf, step, ergodicState, experience.selfToken.locked, err1);
      const meta2 = pmt2.observeSelf(buf, step, ergodicState, experience.selfToken.locked, err2);

      const jointId = _compositeId(meta1, meta2);

      if (!experience.metaTrie._registry.has(jointId)) {
        const snap1 = pmt1.getMetaStateSnapshot(meta1) || { stateIds, meanPredictionError: err1, timestamp: step };
        const snap2 = pmt2.getMetaStateSnapshot(meta2) || { stateIds, meanPredictionError: err2, timestamp: step };
        experience.metaTrie._registry.set(jointId, {
          stateIds: [...(snap1.stateIds || []), ...(snap2.stateIds || [])],
          meanPredictionError: (snap1.meanPredictionError + snap2.meanPredictionError) / 2,
          timestamp: step,
          _parentPerspective: [meta1, meta2],
        });
      }

      const last = experience.metaTrie.lastMetaState;
      if (last !== null && last !== jointId) {
        experience.metaTrie._trie.insert([last, jointId]);
      }
      experience.metaTrie._lastMetaState = jointId;
      experience.selfToken.update(experience.metaTrie, step);
    } else {
      const metaId = experience.metaTrie.observeSelf(experience.traceBuffer, step, ergodicState, experience.selfToken.locked);
      experience.selfToken.update(experience.metaTrie, step);
    }
  }

  if (!frozen && step > 0 && step % (metaObservationInterval * 3) === 0) {
    const recentErrors = experience.traceBuffer.predictionErrorMean(20);
    if (recentErrors > 0.6) {
      const label = `p:${worldStateId.toString(16).padStart(8, '0')}`;
      if (!experience.lexicon.lookupByLabel(label)) {
        const sig = _buildTransitionSignature(experience.lastWorldStateId, worldStateId, experience.lexicon._embeddingDim);
        const tok = inventToken();
        experience.lexicon.bind(label, sig, {
          predictionErrorPeak: recentErrors,
          source: 'proto',
          step,
          outputToken: tok,
        });
      }
    }
  }

  experience.lastWorldStateId = worldStateId;
  return experience;
}

module.exports = { perceive };
