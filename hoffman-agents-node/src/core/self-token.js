const legacy = require('../legacy/self-token');

// The "I" attractor.
//
// v3 lock rule. "I" locks onto meta-state m* when, for `lockConsecutiveRequired`
// consecutive self-observations, all of the following hold on the closed
// recurrent class of the meta-state chain (see MetaTrie.ergodicDiagnostics):
//
//   1. evidence   - at least `minTransitions` observed transitions inside the class, and
//                   at least `minTransitionsPerState` per class state (each kernel row is
//                   estimated from real data, not one or two samples)
//   2. ergodicity - the class kernel is aperiodic and pi has converged
//   3. dominance  - pi(m*) - 1/n >= lockMargin   (or normalised KL(pi || U) >= klThreshold)
//   4. occupancy  - the agent spent >= minOccupancy of its recent observations in the class
//   5. stability  - the dominant state is the same one as at the previous observation
//
// Once locked, "I" unlocks after `unlockConsecutiveRequired` consecutive
// observations with dominance below `unlockMargin` (hysteresis: unlockMargin <
// lockMargin) or occupancy below minOccupancy / 2 - but only when the evidence
// criterion holds. Locking changes the agent's output mode, which creates new
// meta-states; a lock is not revised until the new chain has enough data.
class SelfTokenState {
  constructor({
    token = 'I',
    referentMetaStateId = null,
    stationaryProb = 0,
    locked = false,
    lockGeneration = null,
    lockThreshold = 0.25,
    consecutiveAboveThreshold = 0,
    lockConsecutiveRequired = 3,
    protectionRadius = 2,
    mathVersion = 'v3',
    minTransitions = 20,
    minTransitionsPerState = 2,
    lockMargin = 0.15,
    unlockMargin = 0.05,
    klThreshold = null,
    minOccupancy = 0.6,
    unlockConsecutiveRequired = 3,
    consecutiveBelowThreshold = 0,
    lockHistory = [],
  } = {}) {
    this.token = token;
    this.referentMetaStateId = referentMetaStateId;
    this.stationaryProb = stationaryProb;
    this.locked = locked;
    this.lockGeneration = lockGeneration;
    // lockThreshold is the 2.x rule (max pi > threshold); used by 'legacy' only.
    this.lockThreshold = lockThreshold;
    this.consecutiveAboveThreshold = consecutiveAboveThreshold;
    this.lockConsecutiveRequired = lockConsecutiveRequired;
    this.stabilityHistory = [];
    this.protectionRadius = protectionRadius;
    this.mathVersion = mathVersion;
    this.minTransitions = minTransitions;
    this.minTransitionsPerState = minTransitionsPerState;
    this.lockMargin = lockMargin;
    this.unlockMargin = unlockMargin;
    this.klThreshold = klThreshold;
    this.minOccupancy = minOccupancy;
    this.unlockConsecutiveRequired = unlockConsecutiveRequired;
    this.consecutiveBelowThreshold = consecutiveBelowThreshold;
    this.lockHistory = [...lockHistory];
    this.lastDiagnostics = null;
    this._lastDominant = null;
    this._pendingEvent = null;
  }

  update(metaTrie, generation) {
    if (this.mathVersion === 'legacy') return legacy.update(this, metaTrie, generation);

    const d = metaTrie.ergodicDiagnostics();
    this.lastDiagnostics = d;
    const previousDominant = this._lastDominant;
    this._lastDominant = d.dominant;

    if (d.classSize === 0) {
      this.consecutiveAboveThreshold = 0;
      return;
    }

    this.stationaryProb = d.dominantProb;
    this.stabilityHistory.push(d.dominantProb);
    if (this.stabilityHistory.length > 20) this.stabilityHistory.shift();

    if (!this.locked) {
      if (this.lockCriteria(d, previousDominant).met) {
        this.consecutiveAboveThreshold++;
        if (this.consecutiveAboveThreshold >= this.lockConsecutiveRequired) {
          this._lock(d.dominant, generation, d);
        }
      } else {
        this.consecutiveAboveThreshold = 0;
      }
      return;
    }

    const holds = !this._hasEvidence(d)
      || (d.dominance >= this.unlockMargin && d.occupancy >= this.minOccupancy / 2);
    if (holds) {
      this.consecutiveBelowThreshold = 0;
      if (this._hasEvidence(d)) this.referentMetaStateId = d.dominant;
    } else if (++this.consecutiveBelowThreshold >= this.unlockConsecutiveRequired) {
      this._unlock(generation, d);
    }
  }

  _hasEvidence(d) {
    return d.nTransitions >= this.minTransitions
      && d.nTransitions >= this.minTransitionsPerState * d.classSize;
  }

  lockCriteria(d, previousDominant = this._lastDominant) {
    const concentrated = d.dominance >= this.lockMargin
      || (this.klThreshold !== null && d.normalizedKL >= this.klThreshold);
    const checks = {
      evidence: this._hasEvidence(d),
      ergodic: d.converged && d.aperiodic,
      dominance: concentrated,
      occupancy: d.occupancy >= this.minOccupancy,
      stable: previousDominant !== null && previousDominant === d.dominant,
    };
    return { met: Object.values(checks).every(Boolean), checks };
  }

  _lock(metaStateId, generation, d = null) {
    this.locked = true;
    this.referentMetaStateId = metaStateId;
    this.lockGeneration = generation;
    this.consecutiveBelowThreshold = 0;
    const event = { event: 'lock', generation, referent: metaStateId, stationaryProb: d ? d.dominantProb : this.stationaryProb };
    this.lockHistory.push(event);
    this._pendingEvent = event;
  }

  _unlock(generation, d = null) {
    const event = { event: 'unlock', generation, referent: this.referentMetaStateId, stationaryProb: d ? d.dominantProb : this.stationaryProb };
    this.locked = false;
    this.referentMetaStateId = null;
    this.consecutiveAboveThreshold = 0;
    this.consecutiveBelowThreshold = 0;
    this.lockHistory.push(event);
    this._pendingEvent = event;
  }

  // Returns and clears the most recent lock/unlock event, if any.
  consumeEvent() {
    const e = this._pendingEvent;
    this._pendingEvent = null;
    return e;
  }

  isStable() { return this.locked; }
  isLocked() { return this.locked; }

  // 1 - std(recent dominant stationary probabilities). 1 = perfectly steady.
  stabilityScore() {
    if (this.stabilityHistory.length < 2) return 0;
    const mean = this.stabilityHistory.reduce((a, b) => a + b, 0) / this.stabilityHistory.length;
    const variance = this.stabilityHistory.reduce((a, b) => a + (b - mean) ** 2, 0) / this.stabilityHistory.length;
    return 1 - Math.min(Math.sqrt(variance), 1);
  }

  // Deprecated alias: despite the name this returns a stability score, not a variance.
  stationaryVariance() { return this.stabilityScore(); }

  protectedNodes(metaTrie) {
    if (this.referentMetaStateId === null) return new Set();
    const protected_ = new Set([this.referentMetaStateId]);

    const collectRadius = (stateId, depth) => {
      if (depth > this.protectionRadius) return;
      const node = metaTrie.trie.lookup([stateId]);
      if (!node) return;
      for (const cs of Object.keys(node.children)) {
        const id = parseInt(cs);
        protected_.add(id);
        collectRadius(id, depth + 1);
      }
    };

    collectRadius(this.referentMetaStateId, 0);
    return protected_;
  }

  toJSON() {
    return {
      token: this.token,
      referentMetaStateId: this.referentMetaStateId,
      stationaryProb: this.stationaryProb,
      locked: this.locked,
      lockGeneration: this.lockGeneration,
      lockThreshold: this.lockThreshold,
      consecutiveAboveThreshold: this.consecutiveAboveThreshold,
      lockConsecutiveRequired: this.lockConsecutiveRequired,
      stabilityHistory: [...this.stabilityHistory],
      protectionRadius: this.protectionRadius,
      mathVersion: this.mathVersion,
      minTransitions: this.minTransitions,
      minTransitionsPerState: this.minTransitionsPerState,
      lockMargin: this.lockMargin,
      unlockMargin: this.unlockMargin,
      klThreshold: this.klThreshold,
      minOccupancy: this.minOccupancy,
      unlockConsecutiveRequired: this.unlockConsecutiveRequired,
      consecutiveBelowThreshold: this.consecutiveBelowThreshold,
      lockHistory: this.lockHistory.map(e => ({ ...e })),
      lastDominant: this._lastDominant,
    };
  }

  static fromJSON(data = {}) {
    const st = new SelfTokenState({
      ...data,
      token: data.token || 'I',
      lockThreshold: data.lockThreshold ?? 0.25,
      lockConsecutiveRequired: data.lockConsecutiveRequired ?? 3,
      protectionRadius: data.protectionRadius ?? 2,
      referentMetaStateId: data.referentMetaStateId ?? null,
      lockGeneration: data.lockGeneration ?? null,
      lockHistory: data.lockHistory || [],
      mathVersion: data.mathVersion || 'legacy',
    });
    st.stabilityHistory = [...(data.stabilityHistory || [])];
    st._lastDominant = data.lastDominant ?? null;
    return st;
  }

  // Options accepted by the constructor that configure the lock rule.
  static lockOptions(src = {}) {
    const keys = ['lockThreshold', 'lockConsecutiveRequired', 'minTransitions', 'minTransitionsPerState', 'lockMargin', 'unlockMargin',
      'klThreshold', 'minOccupancy', 'unlockConsecutiveRequired', 'protectionRadius'];
    const out = {};
    for (const k of keys) if (src[k] !== undefined) out[k] = src[k];
    return out;
  }
}

module.exports = { SelfTokenState };
