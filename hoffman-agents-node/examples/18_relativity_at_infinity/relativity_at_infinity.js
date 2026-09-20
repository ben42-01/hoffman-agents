/**
 * Relativity at Infinity
 *
 * Hoffman conjectures that Minkowski spacetime emerges from the traces of
 * n-cycle Markov chains as n -> infinity. No finite network has relativity; the
 * question is what appears in the limit.
 *
 * The agent: its experience is a position on a ring of n sites per unit length
 * plus one bit of memory, the direction it is heading. Each step it moves one
 * site; with probability a/n it reverses (for a quantum agent: amplitude
 * i sin(m/n), a discrete-time quantum walk). Two observers:
 *   - the lab, which sees the agent's position after t·n steps;
 *   - the agent's clock, which registers only its changes of experience (the
 *     reversals), the observer-restricted time of experiment 15.
 *
 *   A. Light cone: does a maximal speed survive n -> infinity?
 *   B. Time dilation: does the clock depend only on proper time √(t² − x²)?
 *   C. Preferred frame: is the whole dynamics Lorentz invariant, classically
 *      and with complex amplitudes?
 *
 * Everything is computed exactly (dynamic programming), not sampled.
 */
const A_RATE = 5; // classical reversal rate per unit time
const MASS = 5;   // quantum reversal amplitude rate (the walk's mass)
const f = (x, d = 3) => x.toFixed(d);
const NS = [20, 80, 320, 1280];

function besselI(nu, z) {
  let t = Math.pow(z / 2, nu);
  for (let k = 1; k <= nu; k++) t /= k;
  let s = 0;
  for (let k = 0; k < 80; k++) { s += t; t *= (z / 2) ** 2 / ((k + 1) * (k + 1 + nu)); }
  return s;
}

function besselJ0(z) {
  let s = 0, t = 1;
  for (let k = 0; k < 120; k++) { s += t; t *= -((z / 2) ** 2) / ((k + 1) ** 2); }
  return s;
}

// erfc (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7), identical in both languages.
function erfc(x) {
  const t = 1 / (1 + 0.3275911 * x);
  const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return y * Math.exp(-x * x);
}

// Continuum predictions for a walk that starts heading right and ends heading left.
const clockPrediction = (a, tau) => 1 + a * tau * besselI(1, a * tau) / besselI(0, a * tau);
const densityPrediction = (a, t, tau) => a * Math.exp(-a * t) * besselI(0, a * tau);
const amplitudePrediction = (m, tau) => m * besselJ0(m * tau);

function properTimeFromClock(a, reading) {
  let lo = 0, hi = 10;
  for (let k = 0; k < 100; k++) { const mid = (lo + hi) / 2; if (clockPrediction(a, mid) < reading) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}

// Exact evolution with n steps per unit time. marks: [{t, x}] with t·n, x·n
// integers of equal parity. For each mark, for walks starting right and ending
// left: the probability density P (per unit length), the expected number of
// reversals E (the clock reading), and the quantum amplitude density Q.
function evolve(n, marks, a = A_RATE, m = MASS) {
  const T = Math.max(...marks.map(k => Math.round(k.t * n)));
  const W = 2 * T + 1;
  let MR = new Float64Array(W), ML = new Float64Array(W), FR = new Float64Array(W), FL = new Float64Array(W);
  let RR = new Float64Array(W), RI = new Float64Array(W), LR = new Float64Array(W), LI = new Float64Array(W);
  MR[T] = 1; RR[T] = 1;
  const p = a / n, c = Math.cos(m / n), s = Math.sin(m / n);
  const out = new Array(marks.length);
  let norm = 1;
  for (let step = 1; step <= T; step++) {
    const nMR = new Float64Array(W), nML = new Float64Array(W), nFR = new Float64Array(W), nFL = new Float64Array(W);
    const nRR = new Float64Array(W), nRI = new Float64Array(W), nLR = new Float64Array(W), nLI = new Float64Array(W);
    for (let i = 0; i < W; i++) {
      // move: right-movers from i-1, left-movers from i+1
      const mr = i > 0 ? MR[i - 1] : 0, fr = i > 0 ? FR[i - 1] : 0, rr = i > 0 ? RR[i - 1] : 0, ri = i > 0 ? RI[i - 1] : 0;
      const ml = i < W - 1 ? ML[i + 1] : 0, fl = i < W - 1 ? FL[i + 1] : 0, lr = i < W - 1 ? LR[i + 1] : 0, li = i < W - 1 ? LI[i + 1] : 0;
      // then reverse (classical: probability p, counted by the clock; quantum: amplitude i s)
      nMR[i] = (1 - p) * mr + p * ml; nML[i] = (1 - p) * ml + p * mr;
      nFR[i] = (1 - p) * fr + p * (fl + ml); nFL[i] = (1 - p) * fl + p * (fr + mr);
      nRR[i] = c * rr - s * li; nRI[i] = c * ri + s * lr;
      nLR[i] = c * lr - s * ri; nLI[i] = c * li + s * rr;
    }
    MR = nMR; ML = nML; FR = nFR; FL = nFL; RR = nRR; RI = nRI; LR = nLR; LI = nLI;
    marks.forEach((k, idx) => {
      if (Math.round(k.t * n) !== step) return;
      const i = T + Math.round(k.x * n);
      out[idx] = { ...k, tau: Math.sqrt(k.t * k.t - k.x * k.x), P: ML[i] * n, E: FL[i] / ML[i], Q: LI[i] * n, Qre: LR[i] * n };
    });
  }
  norm = 0;
  for (let i = 0; i < W; i++) norm += RR[i] ** 2 + RI[i] ** 2 + LR[i] ** 2 + LI[i] ** 2;
  return { marks: out, norm };
}

// Memoryless control with the same long-run diffusion constant D = 1/(2a):
// steps ±h with h = √(2D/n), n steps per unit time. Mass beyond |x| > 1 at t = 1.
function memorylessOutsideCone(n, a = A_RATE) {
  const h = Math.sqrt(1 / (a * n));
  const logFact = [0];
  for (let k = 1; k <= n; k++) logFact.push(logFact[k - 1] + Math.log(k));
  let outside = 0;
  for (let k = 0; k <= n; k++) {
    if (Math.abs(h * (2 * k - n)) > 1 + 1e-12) outside += Math.exp(logFact[n] - logFact[k] - logFact[n - k] - n * Math.LN2);
  }
  return { outside, maxSpeed: h * n };
}

/* ────────────── A. light cone ────────────── */

function partA() {
  console.log('\n  A. Light cone. Where can the agent be after one unit of lab time?');
  const limit = erfc(Math.sqrt(A_RATE / 2));
  console.log('      n       memoryless agent: top speed, mass beyond x = ±1      agent with a direction bit: top speed, mass beyond');
  const rows = NS.map(n => {
    const m = memorylessOutsideCone(n);
    console.log(`      ${String(n).padEnd(6)}  ${f(m.maxSpeed, 1).padStart(8)}   ${f(m.outside, 4)}                               ${f(1, 1).padStart(8)}   0 (exactly)`);
    return m;
  });
  console.log(`      limit      ∞          ${f(limit, 4)} (diffusion)                         1.0        0`);
  console.log('    With one bit of memory the agent has a maximal speed c = 1 at every n, and it survives the limit.');
  console.log('    Without memory, the same spread needs ever faster steps: top speed √(n/a) → ∞ and no light cone.');
  return { rows, limit };
}

/* ────────────── B. time dilation ────────────── */

function partB() {
  console.log('\n  B. Time dilation. The clock counts the agent\'s changes of experience (reversals) during one unit of lab time;');
  console.log('     readings are averaged over walks that start heading right and end heading left at displacement x;');
  console.log(`     the continuum prediction depends only on proper time τ = √(t² − x²): 1 + aτ·I1(aτ)/I0(aτ), a = ${A_RATE}.`);
  const moving = [{ t: 1, x: 0 }, { t: 1, x: 0.6 }, { t: 1, x: 0.8 }];
  const equalTau = [{ t: 0.6, x: 0 }, { t: 0.75, x: 0.45 }, { t: 1, x: 0.8 }];
  const byN = NS.map(n => ({ n, moving: evolve(n, moving).marks, equal: evolve(n, equalTau).marks }));
  const last = byN[byN.length - 1];
  console.log(`      speed v   clock reading (n = ${last.n})   prediction   proper time read off the clock   √(1 − v²)`);
  const dil = last.moving.map(k => {
    const v = k.x / k.t, tau = properTimeFromClock(A_RATE, k.E), pred = clockPrediction(A_RATE, k.tau);
    console.log(`      ${f(v, 1).padEnd(8)}  ${f(k.E, 4).padEnd(26)}  ${f(pred, 4).padEnd(11)}  ${f(tau, 4).padEnd(31)}  ${f(Math.sqrt(1 - v * v), 4)}`);
    return { v, E: k.E, pred, tau };
  });
  const dilErr = Math.max(...dil.map(d => Math.abs(d.tau - Math.sqrt(1 - d.v * d.v))));
  console.log(`    Moving clocks run slow by the Lorentz factor (within ${f(dilErr, 4)} at n = ${last.n}). Three agents with the same proper time`);
  console.log('    τ = 0.6 but speeds 0, 0.6, 0.8 (lab times 0.6, 0.75, 1):');
  console.log('      n       clock readings                      spread     largest error vs prediction');
  const conv = byN.map(({ n, equal }) => {
    const Es = equal.map(k => k.E);
    const spread = Math.max(...Es) - Math.min(...Es);
    const err = Math.max(...equal.map(k => Math.abs(k.E - clockPrediction(A_RATE, k.tau))));
    console.log(`      ${String(n).padEnd(6)}  ${Es.map(e => f(e, 4)).join('  ').padEnd(34)}  ${f(spread, 4).padEnd(9)}  ${f(err, 4)}`);
    return { n, spread, err };
  });
  const rate = conv[conv.length - 2].spread / conv[conv.length - 1].spread;
  console.log(`    The spread shrinks by ${f(rate, 1)}× per 4× in n (∝ 1/n): at every finite n the lattice has a preferred frame;`);
  console.log('    in the limit the clock depends on proper time alone.');
  return { dil, conv, rate, dilErr };
}

/* ────────────── C. preferred frame ────────────── */

function partC() {
  console.log('\n  C. Is the whole dynamics Lorentz invariant? Same three agents with τ = 0.6:');
  const equalTau = [{ t: 0.6, x: 0 }, { t: 0.75, x: 0.45 }, { t: 1, x: 0.8 }];
  const byN = NS.map(n => ({ n, ...evolve(n, equalTau) }));
  const last = byN[byN.length - 1].marks;
  console.log(`      classical agent, probability density   ${last.map(k => f(k.P, 4)).join('  ')}   continuum ${equalTau.map(k => f(densityPrediction(A_RATE, k.t, 0.6), 4)).join('  ')}`);
  console.log(`      quantum agent, amplitude density       ${last.map(k => f(k.Q, 4) + 'i').join('  ')}   continuum ${f(amplitudePrediction(MASS, 0.6), 4)}i at all three`);
  const qSpread = byN.map(({ n, marks }) => ({ n, spread: Math.max(...marks.map(k => k.Q)) - Math.min(...marks.map(k => k.Q)) }));
  const maxRe = Math.max(...byN.flatMap(b => b.marks.map(k => Math.abs(k.Qre))));
  const normErr = Math.max(...byN.map(b => Math.abs(b.norm - 1)));
  console.log(`      quantum spread across the three by n: ${qSpread.map(q => `${q.n}: ${f(q.spread, 4)}`).join(', ')}`);
  console.log(`      (quantum evolution unitary: total probability 1 within ${normErr < 1e-9 ? 'machine precision' : normErr.toExponential(1)})`);
  console.log('    The classical density carries e^(−at): it depends on lab time, so the network\'s rest frame is special.');
  console.log('    The quantum amplitude tends to i·m·J0(mτ), a function of τ alone. It is one component of the 1+1-dimensional');
  console.log('    Dirac propagator (Feynman\'s checkerboard), which is Lorentz covariant: no preferred frame, and the reversal');
  console.log('    rate m plays the role of mass.');
  return { last, qSpread, normErr, maxRe };
}

function main() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Relativity at infinity');
  console.log('='.repeat(78));
  const A = partA(), B = partB(), C = partC();

  console.log('\n' + '─'.repeat(78));
  console.log('Findings');
  console.log('─'.repeat(78));
  console.log('  1. A light cone needs memory. An agent that remembers its direction (one bit of experience) keeps a');
  console.log('     maximal speed as n → ∞. A memoryless agent\'s top speed diverges: in the limit it spreads by diffusion,');
  console.log(`     ${f(A.limit, 4)} of it lands beyond distance 1 after unit time, and no finite speed bounds it.`);
  console.log('  2. Time dilation appears in the limit. The clock that counts the agent\'s own changes of experience');
  console.log(`     converges to a function of proper time alone: at speed 0.6 it reads ${f(B.dil[1].tau, 3)} and at 0.8 it reads ${f(B.dil[2].tau, 3)} units`);
  console.log('     per unit of lab time, Einstein\'s √(1 − v²). In this model the time-dilation part of Hoffman\'s conjecture');
  console.log('     holds, and only in the limit: every finite network has a preferred frame, with errors shrinking like 1/n.');
  console.log('  3. Full Lorentz invariance needs complex amplitudes. The classical agent keeps its network\'s rest frame;');
  console.log('     the quantum agent\'s limit is the Dirac propagator, Lorentz covariant, with mass as reversal rate.');
  console.log('     The mathematics is classical (Goldstein 1951, Kac 1974, Feynman & Hibbs 1965, Gaveau et al. 1984);');
  console.log('     the model is 1+1-dimensional and ours, not Hoffman\'s exact construction.');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
  return { A, B, C };
}

if (require.main === module) main();

module.exports = {
  besselI, besselJ0, erfc, clockPrediction, densityPrediction, amplitudePrediction, properTimeFromClock,
  evolve, memorylessOutsideCone, A_RATE, MASS,
};
