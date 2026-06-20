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
  } = {}) {
    this.token = token;
    this.referentMetaStateId = referentMetaStateId;
    this.stationaryProb = stationaryProb;
    this.locked = locked;
    this.lockGeneration = lockGeneration;
    this.lockThreshold = lockThreshold;
    this.consecutiveAboveThreshold = consecutiveAboveThreshold;
    this.lockConsecutiveRequired = lockConsecutiveRequired;
    this.stabilityHistory = [];
    this.protectionRadius = protectionRadius;
  }

  update(metaTrie, generation) {
    const dist = metaTrie.stationaryDistribution();
    if (dist.size === 0) return;

    let dominant = null, prob = 0;
    for (const [id, p] of dist) {
      if (p > prob) { prob = p; dominant = id; }
    }
    prob = Math.min(prob, 1.0);

    this.stationaryProb = prob;
    this.stabilityHistory.push(prob);
    if (this.stabilityHistory.length > 20) this.stabilityHistory.shift();

    if (!this.locked) {
      if (prob > this.lockThreshold) {
        this.consecutiveAboveThreshold++;
        if (this.consecutiveAboveThreshold >= this.lockConsecutiveRequired) {
          this._lock(dominant, generation);
        }
      } else {
        this.consecutiveAboveThreshold = 0;
      }
    } else {
      this.referentMetaStateId = dominant;
    }
  }

  _lock(metaStateId, generation) {
    this.locked = true;
    this.referentMetaStateId = metaStateId;
    this.lockGeneration = generation;
  }

  isStable() { return this.locked; }
  isLocked() { return this.locked; }

  stationaryVariance() {
    if (this.stabilityHistory.length < 2) return 0;
    const mean = this.stabilityHistory.reduce((a, b) => a + b, 0) / this.stabilityHistory.length;
    const variance = this.stabilityHistory.reduce((a, b) => a + (b - mean) ** 2, 0) / this.stabilityHistory.length;
    return 1 - Math.min(Math.sqrt(variance), 1);
  }

  stabilityScore() { return this.stationaryVariance(); }

  protectedNodes(metaTrie) {
    if (this.referentMetaStateId === null) return new Set();
    const protected_ = new Set([this.referentMetaStateId]);

    const collectRadius = (stateId, depth) => {
      if (depth > this.protectionRadius) return;
      const node = metaTrie.trie.lookup([stateId]);
      if (!node) return;
      for (const cs of Object.keys(node.children)) {
        protected_.add(parseInt(cs));
        collectRadius(parseInt(cs), depth + 1);
      }
    };

    collectRadius(this.referentMetaStateId, 0);
    return protected_;
  }
}

module.exports = { SelfTokenState };
