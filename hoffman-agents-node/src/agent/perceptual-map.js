const { TraceEvent } = require('../core/trace-buffer');
const { inventToken, isInventedToken } = require('../core/token-inventor');
const { buildTransitionSignature } = require('../math/signature');

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

// Perception kernel P: fold one world observation into the experience space.
//
// Prediction error (v3) is 1 - p(actual | previous), with p the Witten-Bell
// transition estimate from the experience trie, so it is a graded surprise in
// [0, 1] rather than an argmax hit/miss. The surprisal -ln p is recorded too.
function perceive(world, experience, step = 0, metaObservationInterval = 20, frozen = false, ergodicState = 'idle', rng = Math.random, generation = null) {
  if (!world || Object.keys(world.sequences).length === 0) return experience;

  const worldStateId = world.getStateId();
  const previousId = experience.lastWorldStateId;

  const v3 = experience.mathVersion !== 'legacy';
  const signatureOf = (prev, curr) => buildTransitionSignature(prev, curr, experience.lexicon._embeddingDim, experience.mathVersion);

  let prediction = null, predictionCorrect = false, predictionError = 0.5, surprisal = null;
  if (previousId !== null) {
    prediction = experience.trie.predictNext([previousId]);
    predictionCorrect = prediction === worldStateId;
    if (v3) {
      const p = experience.trie.transitionProbability(previousId, worldStateId);
      predictionError = Math.min(1, Math.max(0, 1 - p));
      surprisal = -Math.log(p);
    } else {
      predictionError = prediction === worldStateId ? 0 : 1;
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
  if (v3) event.surprisal = surprisal;

  experience.traceBuffer.append(event);

  if (!frozen) {
    experience.trie.insert([event.toState], predictionError);
    if (event.fromState >= 0) {
      if (v3) experience.trie.insertTransition(event.fromState, event.toState, predictionError);
      else experience.trie.insert([event.fromState, event.toState], predictionError);
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
          const sig = signatureOf(experience.lastWorldStateId, worldStateId);
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
        const sig = signatureOf(experience.lastWorldStateId, worldStateId);
        const tok = inventToken(rng);
        const entry = experience.lexicon.bind(label, sig, {
          predictionErrorPeak: predictionError,
          source: 'proto',
          step,
          outputToken: tok,
        });
        entry.integrationDepth = 0.3;
      }
    }

    if (step > 0 && step % metaObservationInterval === 0) {
      const metaId = experience.metaTrie.observeSelf(experience.traceBuffer, step, ergodicState, experience.selfToken.locked);
      experience.selfToken.update(experience.metaTrie, v3 && generation !== null ? generation : step);
    }

    if (step > 0 && step % (metaObservationInterval * 3) === 0) {
      const recentErrors = experience.traceBuffer.predictionErrorMean(20);
      if (recentErrors > 0.6) {
        const label = `p:${worldStateId.toString(16).padStart(8, '0')}`;
        if (!experience.lexicon.lookupByLabel(label)) {
          const sig = signatureOf(experience.lastWorldStateId, worldStateId);
          const tok = inventToken(rng);
          experience.lexicon.bind(label, sig, {
            predictionErrorPeak: recentErrors,
            source: 'proto',
            step,
            outputToken: tok,
          });
        }
      }
    }
  }

  experience.lastWorldStateId = worldStateId;
  return experience;
}

module.exports = { perceive };
