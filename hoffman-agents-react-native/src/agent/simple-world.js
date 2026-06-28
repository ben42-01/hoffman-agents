const { WorldState } = require('./world-state');

const NATIVE_TOKENS = [
  'cross', 'boundary', 'arrive', 'return', 'enter', 'leave',
  'different', 'same', 'familiar', 'again', 'expect', 'notice',
];

class SimpleWorld {
  constructor({ nStates = 10, seed = 42, tokens = null } = {}) {
    this.nStates = nStates;
    this._rng = seedRandom(seed);
    this._tokenPool = tokens || NATIVE_TOKENS;
    this._states = [];
    this._currentIdx = 0;
    this._buildStates();
    this._transitionMatrix = this._buildTransitionMatrix();
  }

  _buildStates() {
    for (let i = 0; i < this.nStates; i++) {
      const length = this._rng() * 3 + 2;
      const tokens = [];
      for (let j = 0; j < length; j++) {
        tokens.push(this._tokenPool[Math.floor(this._rng() * this._tokenPool.length)]);
      }
      this._states.push(tokens);
    }
  }

  _buildTransitionMatrix() {
    const mat = [];
    for (let i = 0; i < this.nStates; i++) {
      const row = [];
      let total = 0;
      for (let j = 0; j < this.nStates; j++) {
        const v = this._rng();
        row.push(v);
        total += v;
      }
      mat.push(row.map(v => v / total));
    }
    return mat;
  }

  step() {
    const row = this._transitionMatrix[this._currentIdx];
    let r = this._rng(), cumulative = 0;
    for (let j = 0; j < row.length; j++) {
      cumulative += row[j];
      if (r <= cumulative) { this._currentIdx = j; break; }
    }
    return WorldState.fromSequence('world', [...this._states[this._currentIdx]]);
  }

  reset() {
    this._currentIdx = Math.floor(this._rng() * this.nStates);
  }

  get currentState() { return [...this._states[this._currentIdx]]; }
}

function seedRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (s >>> 0) / 0xFFFFFFFF;
  };
}

module.exports = { SimpleWorld };
