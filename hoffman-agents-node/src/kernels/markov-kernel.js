const markov = require('../math/markov');
const traceMath = require('../math/trace');

// A stochastic matrix K(r, c) from a finite set of row labels to column labels.
// Rows are probability distributions over columns.
class StochasticMatrix {
  constructor({ rows, cols, matrix, tol = 1e-9 }) {
    this.rows = [...rows];
    this.cols = [...cols];
    this.matrix = matrix.map(r => Float64Array.from(r));
    this._rowIndex = new Map(this.rows.map((s, i) => [s, i]));
    this._colIndex = new Map(this.cols.map((s, i) => [s, i]));
    this.validate(tol);
  }

  validate(tol = 1e-9) {
    if (this.matrix.length !== this.rows.length) {
      throw new Error(`matrix has ${this.matrix.length} rows, expected ${this.rows.length}`);
    }
    this.matrix.forEach((row, i) => {
      if (row.length !== this.cols.length) {
        throw new Error(`row ${this.rows[i]} has ${row.length} entries, expected ${this.cols.length}`);
      }
      for (const v of row) {
        if (!(v >= -tol)) throw new Error(`row ${this.rows[i]} has negative or NaN entry ${v}`);
      }
      const s = markov.rowSum(row);
      if (Math.abs(s - 1) > tol) throw new Error(`row ${this.rows[i]} sums to ${s}, expected 1`);
    });
    return true;
  }

  rowIndex(label) {
    const i = this._rowIndex.get(label);
    if (i === undefined) throw new Error(`unknown row state ${label}`);
    return i;
  }

  prob(from, to) {
    const j = this._colIndex.get(to);
    return j === undefined ? 0 : this.matrix[this.rowIndex(from)][j];
  }

  distribution(from) {
    return Object.fromEntries(this.cols.map((c, j) => [c, this.matrix[this.rowIndex(from)][j]]));
  }

  // Inverse-CDF sampling; consumes exactly one rng() draw.
  sample(from, rng = Math.random) {
    const row = this.matrix[this.rowIndex(from)];
    const r = rng();
    let cumulative = 0;
    for (let j = 0; j < row.length; j++) {
      cumulative += row[j];
      if (r < cumulative) return this.cols[j];
    }
    for (let j = row.length - 1; j >= 0; j--) if (row[j] > 0) return this.cols[j];
    return this.cols[this.cols.length - 1];
  }

  // (K1 K2)(r, c) = sum_m K1(r, m) K2(m, c)
  compose(other) {
    if (this.cols.length !== other.rows.length || this.cols.some((c, i) => c !== other.rows[i])) {
      throw new Error('compose: column states of the first kernel must equal row states of the second');
    }
    return new StochasticMatrix({ rows: this.rows, cols: other.cols, matrix: markov.matMul(this.matrix, other.matrix) });
  }

  // Independent product: (K1 ⊗ K2)((r1,r2), (c1,c2)) = K1(r1,c1) K2(r2,c2)
  tensor(other) {
    return new StochasticMatrix({
      rows: productLabels(this.rows, other.rows),
      cols: productLabels(this.cols, other.cols),
      matrix: markov.kron(this.matrix, other.matrix),
    });
  }

  // Convex combination w K1 + (1 - w) K2 on identical state sets.
  mix(other, w) {
    if (!(w >= 0 && w <= 1)) throw new Error(`mix weight must be in [0, 1], got ${w}`);
    const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
    if (!same(this.rows, other.rows) || !same(this.cols, other.cols)) {
      throw new Error('mix: kernels must share row and column states');
    }
    const matrix = this.matrix.map((row, i) => row.map((v, j) => w * v + (1 - w) * other.matrix[i][j]));
    return new this.constructor({ rows: this.rows, cols: this.cols, states: this.rows, matrix });
  }

  toJSON() {
    return { rows: this.rows, cols: this.cols, matrix: this.matrix.map(r => Array.from(r)) };
  }

  static fromJSON(data) {
    return new StochasticMatrix(data);
  }

  // Estimate from counts; alpha is a symmetric Dirichlet pseudo-count.
  // Rows with no observations become uniform.
  static fromCounts({ rows, cols, counts, alpha = 0 }) {
    const { P, emptyRows } = markov.normalizeRows(counts, alpha);
    for (const i of emptyRows) P[i].fill(1 / cols.length);
    return new StochasticMatrix({ rows, cols, matrix: P });
  }
}

// A Markov kernel on a single state space.
class MarkovKernel extends StochasticMatrix {
  constructor({ states, rows, matrix, tol = 1e-9 }) {
    const s = states || rows;
    super({ rows: s, cols: s, matrix, tol });
  }

  get states() { return this.rows; }

  tensor(other) {
    return new MarkovKernel({ states: productLabels(this.states, other.states), matrix: markov.kron(this.matrix, other.matrix) });
  }

  compose(other) {
    return new MarkovKernel({ states: this.states, matrix: super.compose(other).matrix });
  }

  // The trace chain on a subset of states: what an observer who only sees
  // `window` (state labels) experiences. See math/trace.js.
  trace(window) {
    const idx = window.map(label => this.rowIndex(label));
    return new MarkovKernel({ states: window, matrix: traceMath.traceChain(this.matrix, idx) });
  }

  // True if `other` (a kernel on a subset of these states) is this kernel's trace.
  hasTrace(other, tol = 1e-9) {
    return traceMath.isTraceOf(other.matrix, this.matrix, other.states.map(l => this.rowIndex(l)), tol);
  }

  power(k) {
    let K = MarkovKernel.identity(this.states);
    for (let i = 0; i < k; i++) K = K.compose(this);
    return K;
  }

  stationary() {
    const { pi, converged } = markov.stationary(this.matrix);
    return { distribution: Object.fromEntries(this.states.map((s, i) => [s, pi[i]])), pi, converged };
  }

  diagnostics() {
    const { pi, converged } = markov.stationary(this.matrix);
    const classes = markov.communicatingClasses(this.matrix);
    const irreducible = classes.length === 1;
    const per = markov.period(this.matrix, 0);
    const mixing = irreducible ? markov.mixingTimeEstimate(this.matrix, { pi }) : { lambda2: 1, relaxationTime: Infinity, mixingTime: Infinity };
    return {
      states: this.states,
      stationary: Object.fromEntries(this.states.map((s, i) => [s, pi[i]])),
      converged,
      irreducible,
      period: irreducible ? per : null,
      aperiodic: irreducible && per === 1,
      ergodic: irreducible && per === 1,
      classes: classes.map(c => c.map(i => this.states[i])),
      closedClasses: markov.closedClasses(this.matrix).map(c => c.map(i => this.states[i])),
      entropy: markov.entropy(pi),
      ...mixing,
    };
  }

  toJSON() {
    return { states: this.states, matrix: this.matrix.map(r => Array.from(r)) };
  }

  static fromJSON(data) {
    return new MarkovKernel(data);
  }

  static identity(states) {
    return new MarkovKernel({ states, matrix: states.map((_, i) => states.map((__, j) => (i === j ? 1 : 0))) });
  }

  static fromCounts({ states, counts, alpha = 0 }) {
    const { P, emptyRows } = markov.normalizeRows(counts, alpha);
    for (const i of emptyRows) P[i][i] = 1;
    return new MarkovKernel({ states, matrix: P });
  }
}

function productLabels(a, b) {
  const out = [];
  for (const x of a) for (const y of b) out.push(`${x}|${y}`);
  return out;
}

module.exports = { StochasticMatrix, MarkovKernel, productLabels };
