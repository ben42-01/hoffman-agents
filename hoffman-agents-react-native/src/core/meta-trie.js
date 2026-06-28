const crypto = require('../crypto');
const { ExperienceTrie } = require('./experience-trie');

class MetaStateSnapshot {
  constructor(stateIds, meanPredictionError, timestamp) {
    this.stateIds = stateIds;
    this.meanPredictionError = meanPredictionError;
    this.timestamp = timestamp;
  }
}

class MetaTrie {
  constructor(snapshotWindow = 10, maxDepth = 10) {
    this._trie = new ExperienceTrie(maxDepth);
    this._snapshotWindow = snapshotWindow;
    this._registry = new Map();
    this._lastMetaState = null;
    this._tokenRegistry = new Map();
  }

  _computeMetaStateId(stateIds, meanPredictionError) {
    const roundedError = Math.round(meanPredictionError * 100) / 100;
    const data = JSON.stringify([...stateIds, roundedError]);
    const hash = crypto.createHash('sha256').update(data).digest();
    return hash.readUInt32BE(0);
  }

  observeSelf(traceBuffer, timestamp = 0) {
    const recent = traceBuffer.getRecent(this._snapshotWindow);
    if (recent.length === 0) return 0;

    const stateIds = recent.map(e => e.toState);
    const meanError = traceBuffer.predictionErrorMean(this._snapshotWindow);
    const metaId = this._computeMetaStateId(stateIds, meanError);

    if (!this._registry.has(metaId)) {
      this._registry.set(metaId, new MetaStateSnapshot(
        [...stateIds], meanError, timestamp
      ));
    }

    if (this._lastMetaState !== null && metaId !== this._lastMetaState) {
      this._trie.insert([this._lastMetaState, metaId]);
    }

    this._lastMetaState = metaId;
    return metaId;
  }

  getMetaStateSnapshot(metaStateId) {
    return this._registry.get(metaStateId);
  }

  stationaryDistribution() {
    const allIds = [...this._registry.keys()];
    if (allIds.length === 0) return new Map();

    const active = new Set();
    for (const sid of allIds) {
      const node = this._trie.lookup([sid]);
      if (node && Object.keys(node.children).length > 0) active.add(sid);
    }
    if (this._lastMetaState !== null) active.add(this._lastMetaState);
    if (active.size === 0) return new Map();

    const sortedIds = [...active].sort((a, b) => a - b);
    const idx = new Map(sortedIds.map((id, i) => [id, i]));
    const n = sortedIds.length;

    // Build transition matrix
    const P = Array.from({ length: n }, () => new Float64Array(n));
    for (const stateId of sortedIds) {
      const node = this._trie.lookup([stateId]);
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

    // Handle absorbing states
    for (let i = 0; i < n; i++) {
      if (Array.from(P[i]).reduce((a, b) => a + b, 0) === 0) P[i][i] = 1.0;
    }

    // Power iteration
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

  dominantMetaState() {
    const dist = this.stationaryDistribution();
    if (dist.size === 0) return null;
    let bestId = null, bestProb = -1;
    for (const [id, prob] of dist) {
      if (prob > bestProb) { bestProb = prob; bestId = id; }
    }
    return bestId;
  }

  recordToken(metaStateId, token) {
    if (!this._tokenRegistry.has(metaStateId)) {
      this._tokenRegistry.set(metaStateId, new Map());
    }
    const counts = this._tokenRegistry.get(metaStateId);
    counts.set(token, (counts.get(token) || 0) + 1);
  }

  predictToken(metaStateId, minObservations = 3) {
    const counts = this._tokenRegistry.get(metaStateId);
    if (!counts || counts.size === 0) return null;
    let bestToken = null, bestCount = 0;
    for (const [token, count] of counts) {
      if (count > bestCount) { bestCount = count; bestToken = token; }
    }
    return bestCount >= minObservations ? bestToken : null;
  }

  get trie() { return this._trie; }
  get lastMetaState() { return this._lastMetaState; }
  get registrySize() { return this._registry.size; }

  clear() {
    this._trie.clear();
    this._registry.clear();
    this._lastMetaState = null;
    this._tokenRegistry.clear();
  }
}

module.exports = { MetaTrie, MetaStateSnapshot };
