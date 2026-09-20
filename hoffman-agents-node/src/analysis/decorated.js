// Decorated permutations of Markov chains (Hoffman, Prakash & Prentner,
// "Fusions of Consciousness", Entropy 2023, Definitions 1-2).
//
// A decorated permutation on {1..n} is a map sigma: {1..n} -> {1..2n} with
// a <= sigma(a) <= a + n and sigma(a) mod n a permutation. The paper maps a
// Markov kernel on states 1..n (in a given order) to one:
//   transient a            -> a
//   recurrent singleton a  -> a + n          (absorbing state)
//   other recurrent a      -> the first b > a such that the cyclic interval
//                             (a, a+1, ..., b), read mod n, contains a's class
// Decorated permutations index cells of the positive Grassmannian, the
// geometry behind the amplituhedron; the paper conjectures this links agent
// dynamics to particle physics. The map uses only which states communicate,
// not the probabilities, and it depends on the order (labelling) of the states.

const markov = require('../math/markov');

// P: kernel whose row i is state i+1. Returns sigma as an array (1-based values).
function decoratedPermutation(P) {
  const n = P.length;
  const classes = markov.communicatingClasses(P);
  const recurrent = new Set(markov.closedClasses(P).flat());
  const classOf = new Map();
  classes.forEach(c => c.forEach(i => classOf.set(i, c)));
  return Array.from({ length: n }, (_, i) => {
    const a = i + 1;
    if (!recurrent.has(i)) return a;
    const cls = classOf.get(i).map(j => j + 1);
    if (cls.length === 1) return a + n;
    for (let b = a + 1; b <= a + n; b++) {
      const covered = new Set();
      for (let c = a; c <= b; c++) covered.add(((c - 1) % n) + 1);
      if (cls.every(x => covered.has(x))) return b;
    }
    throw new Error('unreachable: a class is always covered within n steps');
  });
}

// The kernel with states relabelled: new state perm[i] is old state i
// (perm is a permutation of 0..n-1).
function relabel(P, perm) {
  const n = P.length;
  const Q = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Q[perm[i]][perm[j]] = P[i][j];
  return Q;
}

// What sigma "should" become under a relabelling if it were a property of the
// chain rather than of the numbering: conjugate the underlying permutation and
// carry the decoration (fixed-point type) along. Returns null for decorations
// that do not survive (not needed for the tests below).
function conjugate(sigma, perm) {
  const n = sigma.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = i + 1, s = sigma[i];
    const target = ((s - 1) % n);               // underlying image (0-based)
    const na = perm[i] + 1, nt = perm[target] + 1;
    if (s === a) out[perm[i]] = na;              // transient fixed point
    else if (s === a + n) out[perm[i]] = na + n; // absorbing fixed point
    else out[perm[i]] = nt > na ? nt : nt + n;   // a -> first position of target after na
  }
  return out;
}

const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// Is sigma covariant under this relabelling (does it just rename, or change shape)?
function isCovariant(P, perm) {
  return same(decoratedPermutation(relabel(P, perm)), conjugate(decoratedPermutation(P), perm));
}

module.exports = { decoratedPermutation, relabel, conjugate, isCovariant };
