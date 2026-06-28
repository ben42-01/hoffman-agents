const crypto = require('../crypto');
const { WorldState } = require('../agent/world-state');

class FeatureSpec {
  constructor(name, normalization = 'minmax', nBins = 4, params = {}) {
    this.name = name;
    this.normalization = normalization;
    this.nBins = nBins;
    this.params = params;
  }
}

class Normalizer {
  constructor(feature) {
    this.feature = feature;
    this.minVal = null;
    this.maxVal = null;
    this.percentiles = null;
    this.binEdges = null;
  }

  fit(values) {
    if (this.feature.normalization === 'minmax') {
      this.minVal = Math.min(...values);
      this.maxVal = Math.max(...values);
      if (this.maxVal === this.minVal) this.maxVal = this.minVal + 1;
    } else if (this.feature.normalization === 'percentile') {
      const sorted = [...values].sort((a, b) => a - b);
      const n = sorted.length;
      this.percentiles = [
        sorted[Math.floor(n * 0.25)],
        sorted[Math.floor(n * 0.50)],
        sorted[Math.floor(n * 0.75)],
      ];
    } else if (this.feature.normalization === 'tanh') {
      const scale = this.feature.params.tanhScale || 0.01;
      this.minVal = -scale;
      this.maxVal = scale;
    }

    const normalized = values.map(v => this._normalizeValue(v));
    this.binEdges = [];
    for (let i = 0; i <= this.feature.nBins; i++) this.binEdges.push(i / this.feature.nBins);
  }

  transform(value) {
    const normalized = this._normalizeValue(value);
    for (let i = 0; i < this.binEdges.length - 1; i++) {
      if (normalized >= this.binEdges[i] && normalized <= this.binEdges[i + 1]) return i;
    }
    return this.feature.nBins - 1;
  }

  _normalizeValue(value) {
    if (this.feature.normalization === 'minmax') {
      return (value - this.minVal) / (this.maxVal - this.minVal);
    } else if (this.feature.normalization === 'percentile') {
      return Math.max(0, Math.min(1, (value - this.percentiles[0]) / (this.percentiles[2] - this.percentiles[0] + 1e-8)));
    } else if (this.feature.normalization === 'tanh') {
      const scale = this.feature.params.tanhScale || 0.01;
      return Math.tanh(value * scale) * 0.5 + 0.5;
    }
    return value;
  }
}

class WorldBuilder {
  constructor() {
    this._features = [];
    this._smoothing = 0.001;
    this._metadata = {};
  }

  addFeature(name, normalization = 'minmax', nBins = 4, params = {}) {
    this._features.push(new FeatureSpec(name, normalization, nBins, params));
    return this;
  }

  setSmoothing(smoothing) {
    this._smoothing = smoothing;
    return this;
  }

  setMetadata(metadata) {
    Object.assign(this._metadata, metadata);
    return this;
  }

  build(data) {
    if (!Array.isArray(data)) throw new Error('data must be an array');
    const nRows = data.length;
    const nCols = this._features.length;
    if (nRows === 0) throw new Error('data must not be empty');

    // Fit normalizers
    const normalizers = this._features.map((feat, i) => {
      const col = data.map(row => typeof row === 'number' ? row : (row[feat.name] || row[i] || 0));
      const norm = new Normalizer(feat);
      norm.fit(col);
      return norm;
    });

    // Discretize
    const stateIds = data.map((row, rowIdx) => {
      const bins = normalizers.map((norm, i) => {
        const val = typeof row === 'number' ? row : (row[norm.feature.name] || row[i] || 0);
        return norm.transform(val);
      });
      return Math.abs(hashCode(JSON.stringify(bins))) >>> 0;
    });

    // Unique states
    const unique = [...new Set(stateIds)].sort((a, b) => a - b);
    const idToIdx = new Map(unique.map((id, i) => [id, i]));
    const nStates = unique.length;

    // Build transition matrix
    const P = Array.from({ length: nStates }, () => new Float64Array(nStates).fill(this._smoothing));
    for (let i = 0; i < stateIds.length - 1; i++) {
      const fromIdx = idToIdx.get(stateIds[i]);
      const toIdx = idToIdx.get(stateIds[i + 1]);
      P[fromIdx][toIdx] += 1;
    }

    // Row normalize
    for (let i = 0; i < nStates; i++) {
      let sum = 0;
      for (let j = 0; j < nStates; j++) sum += P[i][j];
      if (sum > 0) for (let j = 0; j < nStates; j++) P[i][j] /= sum;
    }

    const labels = {};
    for (const id of unique) labels[id] = `s${id % 1000}`;

    return new World({
      nStates,
      transitionMatrix: P,
      stateIds: [...stateIds],
      stateLabels: labels,
      normalizers,
      initialState: stateIds[0] || 0,
    });
  }
}

class World {
  constructor({ nStates, transitionMatrix, stateIds, stateLabels, normalizers = [], initialState = 0 } = {}) {
    this.nStates = nStates;
    this.transitionMatrix = transitionMatrix;
    this.stateIds = stateIds;
    this.stateLabels = stateLabels;
    this.normalizers = normalizers;
    this._currentState = initialState;
  }

  step() {
    const row = this.transitionMatrix[this._currentState];
    let r = Math.random(), cumulative = 0;
    let nextIdx = 0;
    for (let j = 0; j < row.length; j++) {
      cumulative += row[j];
      if (r <= cumulative) { nextIdx = j; break; }
    }
    this._currentState = nextIdx;
    return WorldState.fromSequence('world', [this.stateLabels[nextIdx] || String(nextIdx)]);
  }

  stepFromData(stateId) {
    this._currentState = stateId;
    return stateId;
  }

  stateFromNewData(data) {
    const bins = this.normalizers.map((norm, i) => {
      const featName = norm.feature.name;
      return norm.transform(data[featName] || 0);
    });
    return Math.abs(hashCode(JSON.stringify(bins))) >>> 0;
  }

  getLabel(stateId) { return this.stateLabels[stateId] || String(stateId); }
  get initialState() { return this._currentState; }

  static fromComponents({ stateIds, transitionMatrix, stateLabels = null }) {
    const nStates = transitionMatrix.length;
    if (!stateLabels) {
      stateLabels = {};
      for (let i = 0; i < nStates; i++) stateLabels[i] = `s${i}`;
    }
    return new World({ nStates, transitionMatrix, stateIds: [...stateIds], stateLabels, initialState: stateIds[0] || 0 });
  }
}

class CoinTossWorld {
  constructor(nCoins = 4) {
    this.nCoins = nCoins;
    this.nStates = 2 ** nCoins;
    this._state = 0;
  }

  step() {
    this._state = Math.floor(Math.random() * this.nStates);
    return WorldState.fromSequence('world', [String(this._state)]);
  }
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

module.exports = { World, WorldBuilder, CoinTossWorld, Normalizer, FeatureSpec };
