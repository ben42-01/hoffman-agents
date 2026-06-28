const crypto = require('../crypto');

const DIMENSION_BINS = {
  temperature: 5, intensity: 5, valence: 5, rhythm: 4,
};

function sequenceToStateId(sequence) {
  const data = sequence.join('|');
  return crypto.createHash('sha256').update(data).digest().readUInt32BE(0);
}

class WorldState {
  constructor(sequences = {}, dimensions = {}) {
    this.sequences = sequences;
    this.dimensions = dimensions;
  }

  getStateId() {
    const flat = [];
    for (const [aid, seq] of Object.entries(this.sequences).sort()) {
      flat.push(aid, ...seq);
    }
    for (const [k, v] of Object.entries(this.dimensions).sort()) {
      const nBins = DIMENSION_BINS[k] || 5;
      const normalized = Math.max(0, Math.min(1, v));
      const binIdx = Math.min(Math.floor(normalized * nBins), nBins - 1);
      flat.push(`${k}:${binIdx}`);
    }
    return sequenceToStateId(flat);
  }

  dimensionDelta(other) {
    if (!other) return {};
    const delta = {};
    for (const k of Object.keys(this.dimensions)) {
      delta[k] = this.dimensions[k] - (other.dimensions[k] || 0);
    }
    return delta;
  }

  static fromSequence(agentId, sequence) {
    return new WorldState({ [agentId]: sequence });
  }
}

// WorldState is truthy if it has any sequences
WorldState.prototype[Symbol.toPrimitive] = function (hint) {
  if (hint === 'boolean') return Object.keys(this.sequences).length > 0;
  return null;
};

module.exports = { WorldState, EnvironmentState: WorldState, sequenceToStateId };
