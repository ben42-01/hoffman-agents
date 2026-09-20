// Trace chains: what an observer restricted to part of a network experiences.
//
// Let P be a Markov kernel on states V and S ⊂ V an observer's window. The
// observer only registers the chain while it is in S; excursions outside S are
// invisible. The sequence of states in S it registers is again a Markov chain,
// the trace (induced, censored) chain on S:
//
//     P_S = P_SS + P_SC (I - P_CC)^(-1) P_CS          (C = V \ S)
//
// "stay in S" plus "leave S, wander in C for any number of hidden steps, come
// back". Hoffman, Prakash & Chattopadhyay's Trace Chain Theorem (2025) states
// that this effective dynamics exists and is unique; the ordering "A is a
// trace of B" is the basis of their trace logic. Standard facts used below
// (e.g. Levin, Peres & Wilmer; Aldous & Fill):
//   - transitivity: the trace of a trace is the trace on the smaller window;
//   - the stationary distribution of P_S is pi restricted to S, renormalised;
//   - Kac: the mean number of network steps between two observer events is
//     1 / pi(S), so an observer's clock runs at rate pi(S).
//
// Ported one-to-one to conscious_agent/math/trace.py.

const markov = require('./markov');

// Solve A X = B (A: n x n, B: n x k) by Gaussian elimination with partial
// pivoting. Returns null if A is singular.
function solveLinear(A, B) {
  const n = A.length;
  const k = n === 0 ? 0 : B[0].length;
  const M = A.map((row, i) => [...row, ...B[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    if (piv !== col) [M[piv], M[col]] = [M[col], M[piv]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      if (f === 0) continue;
      for (let c = col; c < n + k; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => Float64Array.from({ length: k }, (_, j) => row[n + j] / M[i][i]));
}

function checkWindow(n, S) {
  if (!Array.isArray(S) || S.length === 0) throw new Error('trace window must be a non-empty array of state indices');
  const seen = new Set();
  for (const s of S) {
    if (!Number.isInteger(s) || s < 0 || s >= n) throw new Error(`window index ${s} out of range 0..${n - 1}`);
    if (seen.has(s)) throw new Error(`window index ${s} repeated`);
    seen.add(s);
  }
  return [...Array(n).keys()].filter(i => !seen.has(i));
}

// Trace chain of P on the window S (array of state indices; the result's rows
// and columns follow the order of S). Throws if the chain can wander outside S
// forever from some state (I - P_CC singular), where no trace exists.
function traceChain(P, S) {
  const C = checkWindow(P.length, S);
  const PSS = markov.subMatrix(P, S);
  if (C.length === 0) return PSS;
  const IminusPCC = C.map((i, a) => C.map((j, b) => (a === b ? 1 : 0) - P[i][j]));
  const PCS = C.map(i => S.map(j => P[i][j]));
  const Z = solveLinear(IminusPCC, PCS); // (I - P_CC)^(-1) P_CS
  if (!Z) throw new Error('no trace: from some hidden state the chain never returns to the window');
  return S.map((i, a) => Float64Array.from(S, (_, b) => {
    let v = PSS[a][b];
    C.forEach((c, q) => { v += P[i][c] * Z[q][b]; });
    return v;
  }));
}

// Is the kernel PA (on window S of B's states) the trace of PB on S?
function isTraceOf(PA, PB, S, tol = 1e-9) {
  const T = traceChain(PB, S);
  return T.length === PA.length && T.every((row, i) => row.every((v, j) => Math.abs(v - PA[i][j]) <= tol));
}

// Observer clock rate: fraction of network steps at which the observer
// registers an event (= pi(S), by the ergodic theorem).
function clockRate(pi, S) {
  return S.reduce((s, i) => s + pi[i], 0);
}

// Mean number of network steps between consecutive observer events, computed
// from hitting times (independently of Kac's formula, which says it is 1/pi(S)).
function meanReturnTime(P, S, pi = null) {
  const C = checkWindow(P.length, S);
  if (!pi) pi = markov.stationary(P).pi;
  const piS = clockRate(pi, S);
  let h = new Float64Array(0);
  if (C.length > 0) {
    const IminusPCC = C.map((i, a) => C.map((j, b) => (a === b ? 1 : 0) - P[i][j]));
    const sol = solveLinear(IminusPCC, C.map(() => [1]));
    if (!sol) throw new Error('no return: the chain can avoid the window forever');
    h = Float64Array.from(sol, r => r[0]); // expected hidden steps before re-entering S
  }
  let mean = 0;
  for (const s of S) {
    let hidden = 0;
    C.forEach((c, q) => { hidden += P[s][c] * h[q]; });
    mean += (pi[s] / piS) * (1 + hidden);
  }
  return mean;
}

// H(X_n | X_0) for X_0 ~ pi, in nats: the observer's uncertainty about the
// state n steps after an observation, for n = 0..nMax. Non-decreasing in n for
// a stationary chain (data-processing inequality) even when H(X_n) is constant:
// the entropic arrow that Hoffman, Prakash & Prentner (2023) attribute to
// projection rather than to the dynamics.
function conditionalEntropyProfile(P, pi, nMax) {
  const out = [];
  let M = P.map((_, i) => { const r = new Float64Array(P.length); r[i] = 1; return r; });
  for (let n = 0; n <= nMax; n++) {
    out.push(M.reduce((s, row, i) => s + pi[i] * markov.entropy(row), 0));
    M = markov.matMul(M, P);
  }
  return out;
}

/* ────────────── trace logic (Hoffman & Prakash, "Traces of Consciousness") ────────────── */

// Qualia kernel Q = D A P of an agent in Hoffman & Prakash's original form:
// D: X -> G (|X| x |G|), A: G -> W (|G| x |W|), P: W -> X (|W| x |X|).
// The agent's experience-to-experience dynamics with actions and world
// integrated out; the trace logic is built on these kernels.
function qualiaKernel(D, A, P) {
  return markov.matMul(markov.matMul(D, A), P);
}

// Lebesgue order on probability measures over labelled finite sets
// (measures as Map<label, prob>): nu <=_L mu iff nu's support lies in mu's and
// mu restricted to supp(nu) is proportional to nu. On finite spaces this is the
// order of Bennett, Hoffman & Murthy's Lebesgue logic of probabilistic belief;
// "conditioning mu on supp(nu) gives nu", so Bayesian conditioning is a meet.
function lebesgueLeq(nu, mu, tol = 1e-9) {
  let mass = 0;
  for (const [label, p] of nu) {
    if (p <= tol) continue;
    if (!mu.has(label)) return false;
    mass += mu.get(label);
  }
  if (!(mass > 0)) return false;
  for (const [label, p] of nu) {
    const q = (mu.get(label) || 0) / mass;
    if (Math.abs(q - p) > tol) return false;
  }
  return true;
}

// The stationary measure of a labelled kernel {states, matrix} as Map<label, prob>.
function stationaryMeasure(states, P) {
  const { pi } = markov.stationary(P);
  return new Map(states.map((s, i) => [s, pi[i]]));
}

module.exports = {
  solveLinear, traceChain, isTraceOf, clockRate, meanReturnTime, conditionalEntropyProfile,
  qualiaKernel, lebesgueLeq, stationaryMeasure,
};
