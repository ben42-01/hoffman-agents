const { fnv1a32 } = require('./rng');

// Java-style 31x string hash used by 2.x. Kept only for mathVersion 'legacy';
// it has no Python equivalent (Python 2.x used the per-process salted hash()).
function legacyHashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

// One-hot-pair embedding of a transition prev -> curr, L2-normalised.
function buildTransitionSignature(prevId, currId, embeddingDim, mathVersion = 'v3') {
  const sig = new Float64Array(embeddingDim);
  if (prevId !== null && prevId !== undefined) {
    const key = `${prevId}->${currId}`;
    const h = mathVersion === 'legacy' ? Math.abs(legacyHashCode(key)) : fnv1a32(key);
    sig[h % embeddingDim] = 1;
  }
  sig[currId % embeddingDim] = 1;
  let norm = 0;
  for (let i = 0; i < sig.length; i++) norm += sig[i] * sig[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < sig.length; i++) sig[i] /= norm;
  return sig;
}

module.exports = { buildTransitionSignature, legacyHashCode };
