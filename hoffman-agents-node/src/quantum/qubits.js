// Quantum agents: small, dependency-free complex linear algebra for qubit
// systems (dimension <= 16), used by examples/17_quantum_agents.
//
// A quantum conscious agent replaces Hoffman's Markov kernels by their quantum
// generalisation: experiences are density matrices, perception and action are
// quantum channels (completely positive trace-preserving maps, given by Kraus
// operators), and decisions are measurements. A Markov kernel is the special
// case of a channel that destroys all coherence.
//
// Ported one-to-one to conscious_agent/quantum/qubits.py.

// Complex matrix: { re: Float64Array[], im: Float64Array[] } (row-major).
function cmat(re, im = null) {
  const n = re.length, m = re[0].length;
  return {
    re: re.map(r => Float64Array.from(r)),
    im: im ? im.map(r => Float64Array.from(r)) : Array.from({ length: n }, () => new Float64Array(m)),
  };
}

function zerosC(n, m = n) {
  return { re: Array.from({ length: n }, () => new Float64Array(m)), im: Array.from({ length: n }, () => new Float64Array(m)) };
}

function identity(n) {
  const I = zerosC(n);
  for (let i = 0; i < n; i++) I.re[i][i] = 1;
  return I;
}

function mul(A, B) {
  const n = A.re.length, k = B.re.length, m = B.re[0].length;
  const C = zerosC(n, m);
  for (let i = 0; i < n; i++) for (let l = 0; l < k; l++) {
    const ar = A.re[i][l], ai = A.im[i][l];
    if (ar === 0 && ai === 0) continue;
    for (let j = 0; j < m; j++) {
      C.re[i][j] += ar * B.re[l][j] - ai * B.im[l][j];
      C.im[i][j] += ar * B.im[l][j] + ai * B.re[l][j];
    }
  }
  return C;
}

function add(A, B, sa = 1, sb = 1) {
  return {
    re: A.re.map((r, i) => r.map((v, j) => sa * v + sb * B.re[i][j])),
    im: A.im.map((r, i) => r.map((v, j) => sa * v + sb * B.im[i][j])),
  };
}

// Multiply by the complex scalar (cr + i ci).
function scale(A, cr, ci = 0) {
  return {
    re: A.re.map((r, i) => r.map((v, j) => cr * v - ci * A.im[i][j])),
    im: A.im.map((r, i) => r.map((v, j) => cr * v + ci * A.re[i][j])),
  };
}

function dagger(A) {
  const n = A.re.length, m = A.re[0].length;
  const D = zerosC(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { D.re[j][i] = A.re[i][j]; D.im[j][i] = -A.im[i][j]; }
  return D;
}

function kron(A, B) {
  const n = A.re.length, m = A.re[0].length, p = B.re.length, q = B.re[0].length;
  const C = zerosC(n * p, m * q);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const ar = A.re[i][j], ai = A.im[i][j];
    if (ar === 0 && ai === 0) continue;
    for (let k = 0; k < p; k++) for (let l = 0; l < q; l++) {
      C.re[i * p + k][j * q + l] = ar * B.re[k][l] - ai * B.im[k][l];
      C.im[i * p + k][j * q + l] = ar * B.im[k][l] + ai * B.re[k][l];
    }
  }
  return C;
}

// Real part of the trace (the trace of a Hermitian product is real).
function traceRe(A) {
  let t = 0;
  for (let i = 0; i < A.re.length; i++) t += A.re[i][i];
  return t;
}

const PAULI = {
  I: cmat([[1, 0], [0, 1]]),
  X: cmat([[0, 1], [1, 0]]),
  Y: cmat([[0, 0], [0, 0]], [[0, -1], [1, 0]]),
  Z: cmat([[1, 0], [0, -1]]),
};

// Pure state |psi><psi| from amplitude vectors (re, im).
function densityFromState(re, im = null) {
  const n = re.length;
  const ai = im || new Float64Array(n);
  const rho = zerosC(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    rho.re[i][j] = re[i] * re[j] + ai[i] * ai[j];
    rho.im[i][j] = ai[i] * re[j] - re[i] * ai[j];
  }
  return rho;
}

// exp(-i t H) for H with H^2 = I (a product of Pauli matrices): cos t I - i sin t H.
function pauliRotation(H, t) {
  return add(scale(identity(H.re.length), Math.cos(t)), scale(H, 0, -Math.sin(t)));
}

function conjugate(rho, U) {
  return mul(mul(U, rho), dagger(U));
}

// Channel rho -> sum_k K rho K^dagger.
function applyChannel(rho, kraus) {
  return kraus.reduce((acc, K) => add(acc, conjugate(rho, K)), zerosC(rho.re.length));
}

// Kraus operators, lifted to act on qubit `which` of `nQubits`.
function onQubit(ops, which, nQubits) {
  return ops.map(K => {
    let M = null;
    for (let q = 0; q < nQubits; q++) {
      const f = q === which ? K : PAULI.I;
      M = M ? kron(M, f) : f;
    }
    return M;
  });
}

// Depolarising channel: with probability p the qubit is replaced by noise.
function depolarizing(p) {
  return [scale(PAULI.I, Math.sqrt(1 - 3 * p / 4)), scale(PAULI.X, Math.sqrt(p / 4)), scale(PAULI.Y, Math.sqrt(p / 4)), scale(PAULI.Z, Math.sqrt(p / 4))];
}

// Dephasing channel: with probability p the phase is flipped (coherence decays).
function dephasing(p) {
  return [scale(PAULI.I, Math.sqrt(1 - p)), scale(PAULI.Z, Math.sqrt(p))];
}

// Kraus operators of a classical Markov kernel P (n x n) as a quantum channel:
// K_ij = sqrt(P_ij) |j><i|. On diagonal (classical) states it acts exactly as P
// (diag(p) -> diag(pP)); it destroys all coherence. Every Markov agent is thus a
// quantum agent that has fully decohered.
function markovChannel(P) {
  const n = P.length, out = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (!(P[i][j] > 0)) continue;
    const K = zerosC(n);
    K.re[j][i] = Math.sqrt(P[i][j]);
    out.push(K);
  }
  return out;
}

// Largest deviation of sum_k K^dagger K from the identity (0 for a channel).
function krausDeviation(kraus) {
  const n = kraus[0].re.length;
  const S = kraus.reduce((acc, K) => add(acc, mul(dagger(K), K)), zerosC(n));
  let dev = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    dev = Math.max(dev, Math.abs(S.re[i][j] - (i === j ? 1 : 0)), Math.abs(S.im[i][j]));
  }
  return dev;
}

// Spin observable along the unit vector d = [x, y, z]: d · (X, Y, Z).
function spin(d) {
  return add(add(scale(PAULI.X, d[0]), scale(PAULI.Y, d[1])), scale(PAULI.Z, d[2]));
}

// Correlation matrix T_ij = Tr(rho sigma_i ⊗ sigma_j) of a two-qubit state.
// Measuring spins along a (Alice) and b (Bob) gives correlation a^T T b.
function correlationMatrix(rho) {
  const S = [PAULI.X, PAULI.Y, PAULI.Z];
  return S.map(a => S.map(b => traceRe(mul(rho, kron(a, b)))));
}

// Eigen-decomposition of a real symmetric matrix (cyclic Jacobi), eigenvalues
// descending; vectors[k] is the unit eigenvector of values[k].
function symmetricEigen(M) {
  const n = M.length;
  const A = M.map(r => Float64Array.from(r));
  const V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [...Array(n).keys()].sort((a, b) => A[b][b] - A[a][a]);
  return { values: order.map(k => A[k][k]), vectors: order.map(k => V.map(row => row[k])) };
}

const norm3 = (v) => Math.hypot(v[0], v[1], v[2]);
const matVec3 = (T, v) => T.map(row => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);

// Horodecki criterion (1995): the maximum CHSH value of a two-qubit state over
// all spin measurements is 2 sqrt(m1 + m2), where m1 >= m2 are the two largest
// eigenvalues of T^T T. It never exceeds Tsirelson's bound 2 sqrt(2).
function maxChsh(rho) {
  const T = correlationMatrix(rho);
  const TtT = [0, 1, 2].map(i => [0, 1, 2].map(j => T.reduce((s, row) => s + row[i] * row[j], 0)));
  const { values } = symmetricEigen(TtT);
  return 2 * Math.sqrt(Math.max(0, values[0] + values[1]));
}

// Measurement directions attaining maxChsh (Horodecki's construction):
// Bob b, b' = cos(eta) c1 ± sin(eta) c2 with tan(eta) = sqrt(m2/m1);
// Alice a ∝ T c1, a' ∝ T c2. Returns { alice: [a, a'], bob: [b, b'] }.
function optimalSettings(rho) {
  const T = correlationMatrix(rho);
  const TtT = [0, 1, 2].map(i => [0, 1, 2].map(j => T.reduce((s, row) => s + row[i] * row[j], 0)));
  const { values, vectors } = symmetricEigen(TtT);
  const [c1, c2] = vectors;
  const m1 = Math.max(0, values[0]), m2 = Math.max(0, values[1]);
  const eta = m1 + m2 > 0 ? Math.atan2(Math.sqrt(m2), Math.sqrt(m1)) : 0;
  const unit = (v, fallback) => { const l = norm3(v); return l > 1e-12 ? v.map(x => x / l) : fallback; };
  const b = c1.map((x, i) => Math.cos(eta) * x + Math.sin(eta) * c2[i]);
  const b2 = c1.map((x, i) => Math.cos(eta) * x - Math.sin(eta) * c2[i]);
  return { alice: [unit(matVec3(T, c1), c1), unit(matVec3(T, c2), c2)], bob: [b, b2] };
}

// Behaviour p[a][b][x][y]: Alice measures spin along dirsA[a], Bob along
// dirsB[b]; x, y = 0 for outcome +1 and 1 for outcome -1 (Born rule).
function behaviour(rho, dirsA, dirsB) {
  const proj = (d, x) => scale(add(PAULI.I, spin(d), 1, x === 0 ? 1 : -1), 0.5);
  return [0, 1].map(a => [0, 1].map(b => [0, 1].map(x => [0, 1].map(y =>
    traceRe(mul(rho, kron(proj(dirsA[a], x), proj(dirsB[b], y))))))));
}

// Random two-qubit unitary: a product of rotations exp(-i theta P) about all 15
// non-identity Pauli products P, with angles theta uniform in [0, pi) from r.
function randomUnitary2(r) {
  const S = [PAULI.I, PAULI.X, PAULI.Y, PAULI.Z];
  let U = identity(4);
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
    if (a === 0 && b === 0) continue;
    U = mul(pauliRotation(kron(S[a], S[b]), Math.PI * r()), U);
  }
  return U;
}

module.exports = {
  cmat, zerosC, identity, mul, add, scale, dagger, kron, traceRe, PAULI,
  densityFromState, pauliRotation, conjugate, applyChannel, onQubit, depolarizing, dephasing,
  markovChannel, krausDeviation, spin, correlationMatrix, symmetricEigen, maxChsh, optimalSettings, behaviour,
  randomUnitary2,
};
