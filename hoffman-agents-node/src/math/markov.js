// Finite-state Markov chain utilities.
//
// Matrices are arrays of rows (Array or Float64Array). A kernel P is
// row-stochastic: P[i][j] >= 0 and sum_j P[i][j] = 1. Distributions are row
// vectors, so one step of the chain is  mu' = mu P.
//
// Ported one-to-one to conscious_agent/math/markov.py.

const { mulberry32 } = require('./rng');

const EPS = 1e-12;

function zeros(n, m = n) {
  return Array.from({ length: n }, () => new Float64Array(m));
}

function rowSum(row) {
  let s = 0;
  for (let j = 0; j < row.length; j++) s += row[j];
  return s;
}

function isStochastic(P, tol = 1e-9) {
  for (const row of P) {
    for (let j = 0; j < row.length; j++) if (!(row[j] >= -tol)) return false;
    if (Math.abs(rowSum(row) - 1) > tol) return false;
  }
  return true;
}

// Turn a non-negative count matrix into a kernel. alpha is a symmetric
// Dirichlet pseudo-count added to every entry of a row. Rows that are still
// all-zero are returned as zero rows; `emptyRows` lists them so the caller
// decides what an unobserved row means rather than silently inventing one.
function normalizeRows(counts, alpha = 0) {
  const n = counts.length;
  const P = zeros(n, n === 0 ? 0 : counts[0].length);
  const emptyRows = [];
  for (let i = 0; i < n; i++) {
    const row = counts[i];
    const total = rowSum(row) + alpha * row.length;
    if (total <= 0) { emptyRows.push(i); continue; }
    for (let j = 0; j < row.length; j++) P[i][j] = (row[j] + alpha) / total;
  }
  return { P, emptyRows };
}

// Iteratively drop states whose outgoing counts (to surviving states) are all
// zero. Their kernel row is unobserved; treating it as a self-loop would turn
// the state into a spurious absorbing attractor. Returns surviving indices.
function pruneUnobservedRows(counts) {
  const n = counts.length;
  const alive = new Array(n).fill(true);
  for (let changed = true; changed;) {
    changed = false;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      let out = 0;
      for (let j = 0; j < n; j++) if (alive[j]) out += counts[i][j];
      if (out === 0) { alive[i] = false; changed = true; }
    }
  }
  return alive.map((a, i) => (a ? i : -1)).filter(i => i >= 0);
}

function subMatrix(M, idx) {
  return idx.map(i => Float64Array.from(idx.map(j => M[i][j])));
}

function matMul(A, B) {
  const n = A.length, m = B[0].length, k = B.length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < k; l++) {
      const a = A[i][l];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += a * B[l][j];
    }
  }
  return C;
}

function leftApply(mu, P) {
  const n = P.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const w = mu[i];
    if (w === 0) continue;
    const row = P[i];
    for (let j = 0; j < n; j++) out[j] += w * row[j];
  }
  return out;
}

// Solve pi (P - I) = 0 with sum(pi) = 1 by Gaussian elimination with partial
// pivoting. Returns null when the system is singular, i.e. when P has more
// than one closed class and the stationary distribution is not unique.
function solveStationary(P) {
  const n = P.length;
  if (n === 0) return new Float64Array(0);
  // Columns of (P - I)^T become rows; replace the last equation by sum = 1.
  const A = zeros(n, n + 1);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i][j] = P[j][i] - (i === j ? 1 : 0);
  }
  for (let j = 0; j < n; j++) A[n - 1][j] = 1;
  A[n - 1][n] = 1;

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
    if (Math.abs(A[pivot][col]) < 1e-12) return null;
    if (pivot !== col) { const t = A[pivot]; A[pivot] = A[col]; A[col] = t; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = A[r][col] / A[col][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) A[r][c] -= f * A[col][c];
    }
  }
  const pi = new Float64Array(n);
  for (let i = 0; i < n; i++) pi[i] = Math.max(0, A[i][n] / A[i][i]);
  const s = rowSum(pi);
  if (!(s > 0)) return null;
  for (let i = 0; i < n; i++) pi[i] /= s;
  return pi;
}

// Stationary distribution of an irreducible kernel.
//
// Power iteration runs on the lazy chain L = (P + I) / 2. L has exactly the
// same stationary distribution as P but is aperiodic, so iteration converges
// even when P is periodic (plain iteration on a 2-cycle oscillates forever).
// If iteration does not reach `tol`, a direct linear solve is used.
function stationary(P, { tol = 1e-12, maxIter = 10000 } = {}) {
  const n = P.length;
  if (n === 0) return { pi: new Float64Array(0), converged: true, iterations: 0, method: 'empty' };
  if (n === 1) return { pi: Float64Array.of(1), converged: true, iterations: 0, method: 'trivial' };

  let pi = new Float64Array(n).fill(1 / n);
  for (let it = 1; it <= maxIter; it++) {
    const stepped = leftApply(pi, P);
    const next = new Float64Array(n);
    let diff = 0, s = 0;
    for (let i = 0; i < n; i++) { next[i] = 0.5 * (pi[i] + stepped[i]); s += next[i]; }
    for (let i = 0; i < n; i++) { next[i] /= s; diff += Math.abs(next[i] - pi[i]); }
    pi = next;
    if (diff < tol) return { pi, converged: true, iterations: it, method: 'power' };
  }

  const solved = solveStationary(P);
  if (solved) return { pi: solved, converged: true, iterations: maxIter, method: 'solve' };
  return { pi, converged: false, iterations: maxIter, method: 'power' };
}

// Strongly connected components (Tarjan, iterative) of the graph i -> j
// whenever P[i][j] > 0. Components are returned as sorted index arrays.
function communicatingClasses(P) {
  const n = P.length;
  const index = new Int32Array(n).fill(-1);
  const low = new Int32Array(n);
  const onStack = new Uint8Array(n);
  const stack = [];
  const classes = [];
  let counter = 0;

  for (let root = 0; root < n; root++) {
    if (index[root] !== -1) continue;
    const work = [[root, 0]];
    index[root] = low[root] = counter++;
    stack.push(root); onStack[root] = 1;
    while (work.length > 0) {
      const frame = work[work.length - 1];
      const v = frame[0];
      let descended = false;
      while (frame[1] < n) {
        const w = frame[1]++;
        if (!(P[v][w] > 0)) continue;
        if (index[w] === -1) {
          index[w] = low[w] = counter++;
          stack.push(w); onStack[w] = 1;
          work.push([w, 0]);
          descended = true;
          break;
        } else if (onStack[w]) {
          low[v] = Math.min(low[v], index[w]);
        }
      }
      if (descended) continue;
      work.pop();
      if (work.length > 0) {
        const parent = work[work.length - 1][0];
        low[parent] = Math.min(low[parent], low[v]);
      }
      if (low[v] === index[v]) {
        const comp = [];
        let w;
        do { w = stack.pop(); onStack[w] = 0; comp.push(w); } while (w !== v);
        comp.sort((a, b) => a - b);
        classes.push(comp);
      }
    }
  }
  classes.sort((a, b) => a[0] - b[0]);
  return classes;
}

// A class is closed (recurrent) when no probability leaves it.
function closedClasses(P) {
  return communicatingClasses(P).filter(cls => {
    const inCls = new Set(cls);
    for (const i of cls) {
      for (let j = 0; j < P.length; j++) if (P[i][j] > 0 && !inCls.has(j)) return false;
    }
    return true;
  });
}

function isIrreducible(P) {
  return P.length > 0 && communicatingClasses(P).length === 1;
}

function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

// Period of the class containing `start` (the gcd of all cycle lengths).
// Uses BFS levels: every edge u -> v inside the class contributes
// level[u] + 1 - level[v] to the gcd.
function period(P, start = 0) {
  const n = P.length;
  if (n === 0) return 0;
  const cls = communicatingClasses(P).find(c => c.includes(start));
  const inCls = new Set(cls);
  const level = new Map([[start, 0]]);
  const queue = [start];
  let g = 0;
  while (queue.length > 0) {
    const u = queue.shift();
    for (let v = 0; v < n; v++) {
      if (!(P[u][v] > 0) || !inCls.has(v)) continue;
      if (!level.has(v)) { level.set(v, level.get(u) + 1); queue.push(v); }
      else g = gcd(g, level.get(u) + 1 - level.get(v));
    }
  }
  return g; // 0 means no cycle through the class (a transient singleton)
}

function isErgodic(P) {
  return isIrreducible(P) && period(P) === 1;
}

// Estimate |lambda_2|, the second-largest eigenvalue modulus, by power
// iteration on the deflated operator  x -> x P - sum(x P) pi.  This removes
// the Perron component (left eigenvector pi, right eigenvector 1). The
// geometric-mean growth rate over a window tolerates complex eigenvalue pairs.
function secondEigenvalueModulus(P, pi, { iterations = 400, window = 50 } = {}) {
  const n = P.length;
  if (n <= 1) return 0;
  if (!pi) pi = stationary(P).pi;
  const rand = mulberry32(0x5EED);
  let x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = rand() - 0.5;
  const deflate = (v) => {
    const s = rowSum(v);
    for (let i = 0; i < n; i++) v[i] -= s * pi[i];
    return v;
  };
  const norm = (v) => { let s = 0; for (let i = 0; i < n; i++) s += v[i] * v[i]; return Math.sqrt(s); };

  deflate(x);
  let nx = norm(x);
  if (nx < EPS) return 0;
  for (let i = 0; i < n; i++) x[i] /= nx;

  const logs = [];
  for (let it = 0; it < iterations; it++) {
    x = deflate(leftApply(x, P));
    nx = norm(x);
    if (nx < 1e-300) return 0;
    logs.push(Math.log(nx));
    for (let i = 0; i < n; i++) x[i] /= nx;
  }
  const tail = logs.slice(-window);
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  return Math.min(1, Math.exp(mean));
}

// Spectral bounds (Levin, Peres & Wilmer, Thm 12.4 / Rem 12.3):
//   t_rel = 1 / (1 - |lambda_2|),  t_mix(eps) <= t_rel * ln(1 / (eps * pi_min)).
// For non-reversible chains these are estimates, not guarantees.
function mixingTimeEstimate(P, { eps = 0.25, pi = null } = {}) {
  const n = P.length;
  if (n <= 1) return { lambda2: 0, relaxationTime: 1, mixingTime: 0 };
  if (!pi) pi = stationary(P).pi;
  const lambda2 = secondEigenvalueModulus(P, pi);
  const gap = 1 - lambda2;
  if (gap < 1e-12) return { lambda2, relaxationTime: Infinity, mixingTime: Infinity };
  let piMin = Infinity;
  for (const p of pi) if (p > 0 && p < piMin) piMin = p;
  const relaxationTime = 1 / gap;
  return { lambda2, relaxationTime, mixingTime: relaxationTime * Math.log(1 / (eps * piMin)) };
}

// Shannon entropy in nats.
function entropy(p) {
  let h = 0;
  for (const v of p) if (v > 0) h -= v * Math.log(v);
  return h;
}

// KL(p || uniform) = ln n - H(p).
function klFromUniform(p) {
  return p.length === 0 ? 0 : Math.max(0, Math.log(p.length) - entropy(p));
}

function totalVariation(p, q) {
  const n = Math.max(p.length, q.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += Math.abs((p[i] || 0) - (q[i] || 0));
  return d / 2;
}

// Kronecker product. For kernels it is the kernel of two independent chains
// run side by side on the product space (row index i*m + k  <->  (i, k)).
function kron(A, B) {
  const n = A.length, m = B.length;
  const ac = n === 0 ? 0 : A[0].length, bc = m === 0 ? 0 : B[0].length;
  const C = zeros(n * m, ac * bc);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < ac; j++) {
      const a = A[i][j];
      if (a === 0) continue;
      for (let k = 0; k < m; k++) {
        for (let l = 0; l < bc; l++) C[i * m + k][j * bc + l] = a * B[k][l];
      }
    }
  }
  return C;
}

// Marginal of a joint distribution over an n x m product space.
function marginalize(joint, n, m, axis = 0) {
  const out = new Float64Array(axis === 0 ? n : m);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) out[axis === 0 ? i : k] += joint[i * m + k];
  }
  return out;
}


// Normalised net probability flux of a stationary chain:
//   sum_{i<j} |pi_i P_ij - pi_j P_ji| / sum_{i<j} (pi_i P_ij + pi_j P_ji)
// 0 = reversible (detailed balance holds: no arrow of time), 1 = every
// transition is one-directional. Self-loops carry no flux and are ignored.
function irreversibility(P, pi = null) {
  if (!pi) pi = stationary(P).pi;
  let net = 0, total = 0;
  for (let i = 0; i < P.length; i++) {
    for (let j = i + 1; j < P.length; j++) {
      const a = pi[i] * P[i][j], b = pi[j] * P[j][i];
      net += Math.abs(a - b);
      total += a + b;
    }
  }
  return total > 0 ? net / total : 0;
}

// Dobrushin contraction coefficient delta(P) = max_{i,j} TV(P_i., P_j.).
// For any two initial distributions, TV(mu P, nu P) <= delta(P) TV(mu, nu):
// delta bounds how much information about the starting state survives a step.
function dobrushin(P) {
  let worst = 0;
  for (let i = 0; i < P.length; i++) {
    for (let j = i + 1; j < P.length; j++) {
      const d = totalVariation(P[i], P[j]);
      if (d > worst) worst = d;
    }
  }
  return worst;
}

function matPow(P, k) {
  let R = P.map((_, i) => { const r = new Float64Array(P.length); r[i] = 1; return r; });
  for (let s = 0; s < k; s++) R = matMul(R, P);
  return R;
}

// Compressed rows: [{ cols: Int32Array, vals: Float64Array }] (zeros skipped).
function toSparseRows(P) {
  return P.map(row => {
    const cols = [], vals = [];
    for (let j = 0; j < row.length; j++) if (row[j] !== 0) { cols.push(j); vals.push(row[j]); }
    return { cols: Int32Array.from(cols), vals: Float64Array.from(vals) };
  });
}

// Mean return probability R(t) = E_i[(L^t)_{ii}] of the lazy chain
// L = (I + P)/2 for t = 1..tMax.
//   startDistribution 'uniform'    - average over states (every state, or a
//                                    deterministic sample of maxStarts)
//   startDistribution 'stationary' - average over i ~ pi (maxStarts draws). Use
//                                    this for chains with drift: returns from
//                                    transient states measure the drift, not
//                                    the geometry. Pass `pi` to skip computing it.
function returnProbabilities(P, { tMax = 32, maxStarts = 64, seed = 1, lazy = true, startDistribution = 'uniform', pi = null } = {}) {
  const n = P.length;
  const rows = toSparseRows(P);
  const r = mulberry32(seed);
  let starts = [...Array(n).keys()];
  if (startDistribution === 'stationary') {
    const p = pi || stationary(P).pi;
    const cdf = new Float64Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) { acc += p[i]; cdf[i] = acc; }
    starts = Array.from({ length: maxStarts }, () => {
      const u = r() * acc;
      let lo = 0, hi = n - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] > u) hi = mid; else lo = mid + 1; }
      return lo;
    }).sort((a, b) => a - b);
  } else if (startDistribution !== 'uniform') {
    throw new Error(`unknown startDistribution "${startDistribution}"`);
  } else if (n > maxStarts) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [starts[i], starts[j]] = [starts[j], starts[i]]; }
    starts = starts.slice(0, maxStarts).sort((a, b) => a - b);
  }
  const R = new Float64Array(tMax + 1);
  R[0] = 1;
  for (const s of starts) {
    let v = new Float64Array(n);
    v[s] = 1;
    for (let t = 1; t <= tMax; t++) {
      const next = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const w = v[i];
        if (w === 0) continue;
        const { cols, vals } = rows[i];
        const move = lazy ? 0.5 * w : w;
        if (lazy) next[i] += 0.5 * w;
        for (let k = 0; k < cols.length; k++) next[cols[k]] += move * vals[k];
      }
      v = next;
      R[t] += v[s] / starts.length;
    }
  }
  return { R, starts: starts.length };
}

// Spectral dimension d_s of a chain: the exponent in R(t) ~ t^(-d_s/2), the
// return-probability scaling of diffusion. d_s = 1, 2, 3 on lattices of that
// dimension; it grows without bound on expanders (no geometry), and it adds
// under independent products: d_s(P ⊗ Q) = d_s(P) + d_s(Q). It is the
// standard probe of emergent dimension in quantum-gravity models.
//
// Returns the least-squares slope of ln R against ln t over [tMin, tMax]
// (dimension = -2 slope), and dyadic local estimates
//   d_s(t) = -2 ln(R(2t)/R(t)) / ln 2.
// Finite chains saturate at R -> pi, so the window must end well before the
// mixing time; `saturation` = R(tMax) * n reports how close it got (1 = mixed).
function spectralDimension(P, { tMin = 2, tMax = 16, maxStarts = 64, seed = 1, lazy = true, startDistribution = 'uniform', pi = null } = {}) {
  const { R, starts } = returnProbabilities(P, { tMax: 2 * tMax, maxStarts, seed, lazy, startDistribution, pi });
  let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
  for (let t = tMin; t <= tMax; t++) {
    if (!(R[t] > 0)) continue;
    const x = Math.log(t), y = Math.log(R[t]);
    sx += x; sy += y; sxx += x * x; sxy += x * y; m++;
  }
  const slope = m >= 2 ? (m * sxy - sx * sy) / (m * sxx - sx * sx) : NaN;
  const local = [];
  for (let t = tMin; 2 * t <= 2 * tMax; t *= 2) {
    if (R[t] > 0 && R[2 * t] > 0) local.push({ t, d: -2 * Math.log(R[2 * t] / R[t]) / Math.LN2 });
  }
  return { dimension: -2 * slope, local, R: Array.from(R), window: [tMin, tMax], starts, saturation: R[tMax] * P.length };
}

// Entropy rate h = sum_i pi_i H(P_i.) in nats: the average surprise per step of
// the stationary chain. Additive under independent combination:
// h(P ⊗ Q) = h(P) + h(Q).
function entropyRate(P, pi = null) {
  if (!pi) pi = stationary(P).pi;
  return P.reduce((s, row, i) => s + pi[i] * entropy(row), 0);
}

// Expected hitting times H[i][j] = E_i[T_j] (steps to first reach j from i;
// H[j][j] = 0), for an irreducible chain. Solves (I - P_{-j,-j}) h = 1 per target.
function hittingTimes(P) {
  const n = P.length;
  const H = zeros(n);
  for (let j = 0; j < n; j++) {
    const others = [...Array(n).keys()].filter(i => i !== j);
    if (others.length === 0) continue;
    // Gaussian elimination on (I - P restricted to others) h = 1
    const m = others.length;
    const A = others.map((i, a) => {
      const row = new Float64Array(m + 1);
      others.forEach((k, b) => { row[b] = (a === b ? 1 : 0) - P[i][k]; });
      row[m] = 1;
      return row;
    });
    for (let c = 0; c < m; c++) {
      let piv = c;
      for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      if (Math.abs(A[piv][c]) < 1e-14) throw new Error('hittingTimes: chain is not irreducible');
      [A[piv], A[c]] = [A[c], A[piv]];
      for (let r = 0; r < m; r++) {
        if (r === c) continue;
        const f = A[r][c] / A[c][c];
        if (f !== 0) for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k];
      }
    }
    others.forEach((i, a) => { H[i][j] = A[a][m] / A[a][a]; });
  }
  return H;
}

// Commute times K[i][j] = E_i[T_j] + E_j[T_i]. A metric on the states of any
// irreducible chain (hitting times satisfy E_i T_k <= E_i T_j + E_j T_k).
function commuteTimes(P) {
  const H = hittingTimes(P);
  return H.map((row, i) => Float64Array.from(row, (v, j) => v + H[j][i]));
}

// Index of the maximum, taking the lowest index among near-ties. Used wherever
// the choice feeds back into dynamics, so Node and Python pick the same state
// despite last-bit floating-point differences.
function argmaxStable(values, tol = 1e-12) {
  let best = -Infinity;
  for (const v of values) if (v > best) best = v;
  for (let i = 0; i < values.length; i++) if (values[i] >= best - tol) return i;
  return -1;
}

module.exports = {
  argmaxStable, entropyRate, hittingTimes, commuteTimes, irreversibility, dobrushin, matPow, toSparseRows, returnProbabilities, spectralDimension,
  zeros, rowSum, isStochastic, normalizeRows, pruneUnobservedRows, subMatrix, matMul, leftApply,
  stationary, solveStationary,
  communicatingClasses, closedClasses, isIrreducible, period, isErgodic,
  secondEigenvalueModulus, mixingTimeEstimate,
  entropy, klFromUniform, totalVariation, kron, marginalize,
};
