// 2.x meta-trie math, preserved verbatim for mathVersion: 'legacy'.
// Known defects (fixed in v3): the newest meta-state is made absorbing,
// self-transitions are never recorded, plain power iteration does not
// converge on periodic chains, and the lock flag is hashed into the id.
const crypto = require('crypto');

function computeMetaStateId(stateIds, meanPredictionError, ergodicState, isLocked) {
  const errorBucket =
    meanPredictionError < 0.05 ? 0 :
    meanPredictionError < 0.15 ? 1 :
    meanPredictionError < 0.35 ? 2 :
    meanPredictionError < 0.65 ? 3 : 4;
  const coarseStates = stateIds.slice(-2).map(id => id % 8);
  const data = JSON.stringify([coarseStates, errorBucket, ergodicState || 'idle', !!isLocked]);
  const hash = crypto.createHash('sha256').update(data).digest();
  return hash.readUInt32BE(0) & 0x0FFFFFFF;
}

function stationaryDistribution(mt) {
  const allIds = [...mt._registry.keys()];
  if (allIds.length === 0) return new Map();

  const active = new Set();
  for (const sid of allIds) {
    const node = mt._trie.lookup([sid]);
    if (node && Object.keys(node.children).length > 0) active.add(sid);
  }
  if (mt._lastMetaState !== null) active.add(mt._lastMetaState);
  if (active.size === 0) return new Map();

  const sortedIds = [...active].sort((a, b) => a - b);
  const idx = new Map(sortedIds.map((id, i) => [id, i]));
  const n = sortedIds.length;

  const P = Array.from({ length: n }, () => new Float64Array(n));
  for (const stateId of sortedIds) {
    const node = mt._trie.lookup([stateId]);
    if (node && Object.keys(node.children).length > 0) {
      let total = 0;
      for (const child of Object.values(node.children)) total += child.visitCount;
      if (total > 0) {
        for (const [childState, childNode] of Object.entries(node.children)) {
          const ci = idx.get(parseInt(childState));
          if (ci !== undefined) P[idx.get(stateId)][ci] = childNode.visitCount / total;
        }
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (Array.from(P[i]).reduce((a, b) => a + b, 0) === 0) P[i][i] = 1.0;
  }

  let pi = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 1000; iter++) {
    const piNew = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        piNew[j] += pi[i] * P[i][j];
      }
    }
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(piNew[i] - pi[i]);
    pi = piNew;
    if (diff < 1e-8) break;
  }

  const result = new Map();
  for (let i = 0; i < n; i++) result.set(sortedIds[i], pi[i]);
  return result;
}

module.exports = { computeMetaStateId, stationaryDistribution };
