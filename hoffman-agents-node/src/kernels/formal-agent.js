const { StochasticMatrix, MarkovKernel, productLabels } = require('./markov-kernel');
const markov = require('../math/markov');

// Finite reference implementation of a conscious agent in the sense of
// Hoffman & Prakash (2014), "Objects of consciousness":
//
//   C = (X, G, P, D, A, N)
//   P : W × X -> Δ(X)   perception  (given world w, experience x' ~ P[w](x, ·))
//   D : X     -> Δ(G)   decision    (action g ~ D(x', ·))
//   A : G × W -> Δ(W)   action      (next world w' ~ A[g](w, ·))
//   N                   counter of completed perceive-decide-act cycles
//
// One cycle maps (x, w) to (x', w'). The induced chain on X × W has kernel
//   Q((x, w), (x', w')) = P[w](x, x') · Σ_g D(x', g) · A[g](w, w')
// and questions like "does this agent have a unique ergodic regime?" become
// questions about Q (irreducibility, period, stationary distribution).
class FormalConsciousAgent {
  constructor({ X, G, W, P, D, A, x0 = null, N = 0, rng = Math.random }) {
    this.X = [...X];
    this.G = [...G];
    this.W = [...W];
    this.P = _asKernelMap(P, this.W, this.X, 'P');
    this.D = D instanceof StochasticMatrix ? D : new StochasticMatrix({ rows: this.X, cols: this.G, matrix: D });
    this.A = _asKernelMap(A, this.G, this.W, 'A');
    _checkLabels(this.D.rows, this.X, 'D rows');
    _checkLabels(this.D.cols, this.G, 'D cols');
    this.x = x0 ?? this.X[0];
    this.N = N;
    this._rng = rng;
  }

  // One perceive-decide-act cycle against world state w. Returns { x, g, w }.
  step(w) {
    const x = this.P[w].sample(this.x, this._rng);
    const g = this.D.sample(x, this._rng);
    const next = this.A[g].sample(w, this._rng);
    this.x = x;
    this.N++;
    return { x, g, w: next };
  }

  run(w0, n) {
    const trajectory = [];
    let w = w0;
    for (let i = 0; i < n; i++) {
      const out = this.step(w);
      trajectory.push(out);
      w = out.w;
    }
    return trajectory;
  }

  // Q on X × W, states labelled "x|w".
  jointKernel() {
    const nx = this.X.length, nw = this.W.length;
    const states = productLabels(this.X, this.W);
    const Q = markov.zeros(nx * nw);
    for (let i = 0; i < nx; i++) {
      for (let k = 0; k < nw; k++) {
        const Pw = this.P[this.W[k]].matrix;
        for (let j = 0; j < nx; j++) {
          const pxx = Pw[i][j];
          if (pxx === 0) continue;
          for (let gi = 0; gi < this.G.length; gi++) {
            const dg = this.D.matrix[j][gi];
            if (dg === 0) continue;
            const Ag = this.A[this.G[gi]].matrix;
            for (let l = 0; l < nw; l++) Q[i * nw + k][j * nw + l] += pxx * dg * Ag[k][l];
          }
        }
      }
    }
    return new MarkovKernel({ states, matrix: Q });
  }

  diagnostics() {
    const d = this.jointKernel().diagnostics();
    const pi = this.X.flatMap(x => this.W.map(w => d.stationary[`${x}|${w}`]));
    return {
      ...d,
      experienceMarginal: Object.fromEntries(this.X.map((x, i) => [x, markov.marginalize(pi, this.X.length, this.W.length, 0)[i]])),
      worldMarginal: Object.fromEntries(this.W.map((w, i) => [w, markov.marginalize(pi, this.X.length, this.W.length, 1)[i]])),
    };
  }

  // Combination of two agents acting on a shared world W:
  //   X = X1 × X2,  G = G1 × G2
  //   P[w] = P1[w] ⊗ P2[w]           (independent perception of the same world)
  //   D    = D1 ⊗ D2                 (independent decisions)
  //   A[(g1, g2)] = A1[g1] · A2[g2]  (actions applied in sequence, agent 1 first)
  // The sequential choice for A is a modelling decision: two agents writing
  // to one world need an order (or a joint action kernel) to be well defined.
  static combine(c1, c2, { rng = c1._rng } = {}) {
    _checkLabels(c1.W, c2.W, 'combine: world states');
    const X = productLabels(c1.X, c2.X);
    const G = productLabels(c1.G, c2.G);
    const P = Object.fromEntries(c1.W.map(w => [w, c1.P[w].tensor(c2.P[w])]));
    const D = c1.D.tensor(c2.D);
    const A = {};
    for (const g1 of c1.G) for (const g2 of c2.G) A[`${g1}|${g2}`] = c1.A[g1].compose(c2.A[g2]);
    return new FormalConsciousAgent({ X, G, W: c1.W, P, D, A, x0: `${c1.x}|${c2.x}`, rng });
  }

  toJSON() {
    return {
      X: this.X, G: this.G, W: this.W,
      P: Object.fromEntries(Object.entries(this.P).map(([k, v]) => [k, v.toJSON()])),
      D: this.D.toJSON(),
      A: Object.fromEntries(Object.entries(this.A).map(([k, v]) => [k, v.toJSON()])),
      x: this.x, N: this.N,
    };
  }

  static fromJSON(data, { rng = Math.random } = {}) {
    return new FormalConsciousAgent({
      X: data.X, G: data.G, W: data.W,
      P: Object.fromEntries(Object.entries(data.P).map(([k, v]) => [k, MarkovKernel.fromJSON(v)])),
      D: StochasticMatrix.fromJSON(data.D),
      A: Object.fromEntries(Object.entries(data.A).map(([k, v]) => [k, MarkovKernel.fromJSON(v)])),
      x0: data.x, N: data.N, rng,
    });
  }
}

function _asKernelMap(spec, keys, states, name) {
  const out = {};
  for (const key of keys) {
    const k = spec[key];
    if (k === undefined) throw new Error(`${name}: missing kernel for ${key}`);
    out[key] = k instanceof MarkovKernel ? k : new MarkovKernel({ states, matrix: k });
    _checkLabels(out[key].states, states, `${name}[${key}] states`);
  }
  return out;
}

function _checkLabels(actual, expected, what) {
  if (actual.length !== expected.length || actual.some((s, i) => String(s) !== String(expected[i]))) {
    throw new Error(`${what} mismatch: [${actual}] vs [${expected}]`);
  }
}

module.exports = { FormalConsciousAgent };
