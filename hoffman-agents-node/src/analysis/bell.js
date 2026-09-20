// Bell-CHSH analysis of a two-party, two-setting, two-outcome experiment.
//
// A behaviour is p[a][b][x][y] = P(x, y | a, b) with settings a, b in {0, 1}
// and outcomes x, y in {0, 1}. Correlators E(a, b) = sum (-1)^(x xor y) p.
//
// Regions (for unbiased marginals):
//   local          all CHSH variants <= 2           (Bell / Fine)
//   quantum        TLM condition holds, CHSH <= 2*sqrt(2) (Tsirelson)
//   post-quantum   no-signalling but outside the quantum set, up to 4 (PR box)
//   signalling     one party's marginal depends on the other's setting

const TSIRELSON = 2 * Math.SQRT2;

function correlators(p) {
  return [0, 1].map(a => [0, 1].map(b => {
    let e = 0;
    for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) e += (x === y ? 1 : -1) * p[a][b][x][y];
    return e;
  }));
}

// The four CHSH expressions (the minus sign on each of the four terms).
// Their maximum absolute value is the CHSH value of the behaviour.
function chshVariants(E) {
  const terms = [E[0][0], E[0][1], E[1][0], E[1][1]];
  return terms.map((_, k) => terms.reduce((s, e, i) => s + (i === k ? -e : e), 0));
}

function chsh(E) {
  return Math.max(...chshVariants(E).map(Math.abs));
}

// Tsirelson-Landau-Masanes: correlators with unbiased marginals are quantum
// iff for every placement of the minus sign |sum ± asin E(a,b)| <= pi.
function isQuantum(E, tol = 1e-9) {
  const as = [E[0][0], E[0][1], E[1][0], E[1][1]].map(e => Math.asin(Math.max(-1, Math.min(1, e))));
  return as.every((_, k) => Math.abs(as.reduce((s, v, i) => s + (i === k ? -v : v), 0)) <= Math.PI + tol);
}

// Largest change in one party's outcome distribution caused by the other
// party's setting (0 = no-signalling).
function signalling(p) {
  let worst = 0;
  for (let b = 0; b < 2; b++) for (let y = 0; y < 2; y++) {
    const pb = (a) => p[a][b][0][y] + p[a][b][1][y];
    worst = Math.max(worst, Math.abs(pb(0) - pb(1)));
  }
  for (let a = 0; a < 2; a++) for (let x = 0; x < 2; x++) {
    const pa = (b) => p[a][b][x][0] + p[a][b][x][1];
    worst = Math.max(worst, Math.abs(pa(0) - pa(1)));
  }
  return worst;
}

function classify(p, tol = 1e-9) {
  if (signalling(p) > tol) return 'signalling';
  const E = correlators(p);
  const s = chsh(E);
  if (s <= 2 + tol) return 'local';
  return isQuantum(E, tol) ? 'quantum' : 'post-quantum';
}

// Behaviour with uniform marginals and the given correlators.
function fromCorrelators(E) {
  return [0, 1].map(a => [0, 1].map(b => [0, 1].map(x => [0, 1].map(y => (1 + (x === y ? 1 : -1) * E[a][b]) / 4))));
}

module.exports = { TSIRELSON, correlators, chshVariants, chsh, isQuantum, signalling, classify, fromCorrelators };
