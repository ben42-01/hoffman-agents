const { ExperienceTrie } = require('./experience-trie');
const markov = require('../math/markov');
const { fnv1a32 } = require('../math/rng');
const legacy = require('../legacy/meta-trie');

const ID_MASK = 0x0FFFFFFF;

class MetaStateSnapshot {
  constructor(stateIds, meanPredictionError, timestamp) {
    this.stateIds = stateIds;
    this.meanPredictionError = meanPredictionError;
    this.timestamp = timestamp;
  }
}

// The agent's model of its own experiential dynamics: a Markov chain over
// meta-states, estimated from the sequence of self-observations.
//
// Transitions are stored as depth-2 paths [from, to] in `_trie`; the child
// visitCount of `to` under `from` is the transition count N(from -> to).
class MetaTrie {
  constructor(snapshotWindow = 10, maxDepth = 10, { mathVersion = 'v3', historyLimit = 256 } = {}) {
    this._trie = new ExperienceTrie(maxDepth);
    this._snapshotWindow = snapshotWindow;
    this._registry = new Map();
    this._lastMetaState = null;
    this._tokenRegistry = new Map();
    this.mathVersion = mathVersion;
    // Recent meta-state trajectory (v3); used to pick the attractor the agent
    // currently occupies.
    this._history = [];
    this._historyLimit = historyLimit;
    // Combination provenance (v3). Ids inherited from constituents map to
    // { constituentId, localId }; `_provenanceTree` keeps each constituent's
    // own provenance, lock state and parameters so fuse() can restore it.
    this._provenance = new Map();
    this._provenanceTree = {};
  }

  _computeMetaStateId(stateIds, meanPredictionError, ergodicState, isLocked) {
    if (this.mathVersion === 'legacy') {
      return legacy.computeMetaStateId(stateIds, meanPredictionError, ergodicState, isLocked);
    }
    // The lock flag is deliberately NOT part of the id: hashing it in split
    // the chain into disconnected pre-lock and post-lock components.
    const data = JSON.stringify([
      stateIds.slice(-2).map(id => id % 8),
      errorBucket(meanPredictionError),
      ergodicState || 'idle',
    ]);
    let id = fnv1a32(data) & ID_MASK;
    for (let salt = 1; this._provenance.has(id); salt++) id = fnv1a32(`${data}#${salt}`) & ID_MASK;
    return id;
  }

  observeSelf(traceBuffer, timestamp = 0, ergodicState = 'idle', isLocked = false) {
    const recent = traceBuffer.getRecent(this._snapshotWindow);
    if (recent.length === 0) return 0;

    const stateIds = recent.map(e => e.toState);
    const meanError = traceBuffer.predictionErrorMean(this._snapshotWindow);
    const metaId = this._computeMetaStateId(stateIds, meanError, ergodicState, isLocked);

    if (!this._registry.has(metaId)) {
      this._registry.set(metaId, new MetaStateSnapshot([...stateIds], meanError, timestamp));
    }

    if (this.mathVersion === 'legacy') {
      if (this._lastMetaState !== null && metaId !== this._lastMetaState) {
        this._trie.insert([this._lastMetaState, metaId]);
      }
    } else {
      // Self-transitions are real dwell time in an attractor and must count.
      if (this._lastMetaState !== null) this._trie.insert([this._lastMetaState, metaId]);
      this._history.push(metaId);
      if (this._history.length > this._historyLimit) this._history.shift();
    }

    this._lastMetaState = metaId;
    return metaId;
  }

  getMetaStateSnapshot(metaStateId) {
    return this._registry.get(metaStateId);
  }

  // Transition counts between meta-states, optionally excluding ids inherited
  // through combination. Returns { ids, counts } with ids sorted ascending.
  transitionCounts({ includeInherited = false } = {}) {
    const keep = (id) => includeInherited || !this._provenance.has(id);
    const edges = [];
    const ids = new Set();
    for (const [fromStr, node] of Object.entries(this._trie.root.children)) {
      const from = parseInt(fromStr);
      if (!keep(from)) continue;
      for (const [toStr, child] of Object.entries(node.children)) {
        const to = parseInt(toStr);
        if (!keep(to) || child.visitCount <= 0) continue;
        edges.push([from, to, child.visitCount]);
        ids.add(from); ids.add(to);
      }
    }
    const sorted = [...ids].sort((a, b) => a - b);
    const idx = new Map(sorted.map((id, i) => [id, i]));
    const counts = markov.zeros(sorted.length);
    for (const [from, to, c] of edges) counts[idx.get(from)][idx.get(to)] += c;
    return { ids: sorted, counts };
  }

  // Ergodic analysis of the empirical meta-state chain.
  //
  //  1. States with no observed outgoing transition (typically the newest
  //     meta-state) have an unknown kernel row. They are pruned, repeatedly,
  //     instead of being made absorbing: an absorbing sink would soak up all
  //     stationary mass and fake an attractor.
  //  2. The remaining chain is split into communicating classes. The closed
  //     (recurrent) class the agent most recently occupied is selected.
  //  3. On that irreducible class the kernel is estimated (optionally with a
  //     Dirichlet pseudo-count alpha) and pi, the period and mixing are computed.
  ergodicDiagnostics({ alpha = 0, occupancyWindow = 20, includeInherited = false } = {}) {
    const { ids, counts } = this.transitionCounts({ includeInherited });
    const totalTransitions = counts.reduce((s, r) => s + markov.rowSum(r), 0);

    // 1. prune rows with no outgoing mass
    const keptIdx = markov.pruneUnobservedRows(counts);
    const empty = {
      states: [], pi: new Map(), piArray: new Float64Array(0), converged: false,
      period: null, aperiodic: false, ergodic: false, nClasses: 0, nClosedClasses: 0,
      classSize: 0, nTransitions: 0, totalTransitions, prunedStates: ids.length,
      lambda2: null, relaxationTime: null, mixingTime: null,
      entropy: 0, klFromUniform: 0, normalizedKL: 0,
      dominant: null, dominantProb: 0, dominance: 0, occupancy: 0,
    };
    if (keptIdx.length === 0) return empty;

    const keptIds = keptIdx.map(i => ids[i]);
    const C = markov.subMatrix(counts, keptIdx);

    // 2. pick the closed class the agent is in (or most recently was in)
    const classes = markov.communicatingClasses(C);
    const closed = markov.closedClasses(C);
    const classOf = new Map();
    closed.forEach((cls, k) => cls.forEach(i => classOf.set(keptIds[i], k)));
    let chosen = null;
    for (let h = this._history.length - 1; h >= 0 && chosen === null; h--) {
      if (classOf.has(this._history[h])) chosen = classOf.get(this._history[h]);
    }
    if (chosen === null) {
      let best = -1;
      closed.forEach((cls, k) => {
        const mass = cls.reduce((s, i) => s + cls.reduce((t, j) => t + C[i][j], 0), 0);
        if (mass > best) { best = mass; chosen = k; }
      });
    }
    const cls = closed[chosen];
    const states = cls.map(i => keptIds[i]);

    // 3. analyse the irreducible class
    const classCounts = markov.subMatrix(C, cls);
    const nTransitions = classCounts.reduce((s, r) => s + markov.rowSum(r), 0);
    const { P } = markov.normalizeRows(classCounts, alpha);
    const { pi, converged } = markov.stationary(P);
    const per = markov.period(P, 0);
    const mixing = markov.mixingTimeEstimate(P, { pi });
    const n = states.length;

    const dominantIdx = markov.argmaxStable(pi);
    const kl = markov.klFromUniform(pi);
    const inClass = new Set(states);
    const recent = this._history.slice(-occupancyWindow);
    const occupancy = recent.length === 0 ? 0 : recent.filter(id => inClass.has(id)).length / recent.length;

    return {
      states,
      pi: new Map(states.map((s, i) => [s, pi[i]])),
      piArray: pi,
      converged,
      period: per,
      aperiodic: per === 1,
      ergodic: per === 1,
      nClasses: classes.length,
      nClosedClasses: closed.length,
      classSize: n,
      nTransitions,
      totalTransitions,
      prunedStates: ids.length - keptIds.length,
      ...mixing,
      entropy: markov.entropy(pi),
      klFromUniform: kl,
      normalizedKL: n === 1 ? 1 : kl / Math.log(n),
      dominant: states[dominantIdx],
      dominantProb: pi[dominantIdx],
      // How far the dominant state sits above a uniform distribution on the
      // class. A single self-looping state is maximally dominant.
      dominance: n === 1 ? 1 : pi[dominantIdx] - 1 / n,
      occupancy,
    };
  }

  stationaryDistribution() {
    if (this.mathVersion === 'legacy') return legacy.stationaryDistribution(this);
    return this.ergodicDiagnostics().pi;
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
  get history() { return [...this._history]; }

  clear() {
    this._trie.clear();
    this._registry.clear();
    this._lastMetaState = null;
    this._tokenRegistry.clear();
    this._history = [];
    this._provenance.clear();
    this._provenanceTree = {};
  }
}

function errorBucket(e) {
  return e < 0.05 ? 0 : e < 0.15 ? 1 : e < 0.35 ? 2 : e < 0.65 ? 3 : 4;
}

module.exports = { MetaTrie, MetaStateSnapshot, errorBucket, ID_MASK };
