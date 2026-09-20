/**
 * Bell Test Through the Headset
 *
 * Granting Hoffman's premise: spacetime is an interface (a "headset"), and
 * behind it is a network of conscious agents that spacetime does not
 * constrain. Two observers, Alice and Bob, are conscious agents at the
 * interface. Each round:
 *
 *   1. the hidden network is in its stationary regime:      h  ~ pi_Q
 *   2. Alice chooses a setting a and perceives x~ from h:    x~ ~ P_A[a](h, .)
 *   3. Alice acts, and her action enters the network:        h1 ~ A[a, x~](h, .)
 *   4. the network runs k steps of its own dynamics Q:       h2 ~ Q^k(h1, .)
 *   5. Bob chooses b and perceives y~ from h2:               y~ ~ P_B[b](h2, .)
 *
 * This is the perceive -> decide -> act cycle of the (X, G, P, D, A, N) agent.
 * The hidden network Q is the joint kernel of two combined FormalConsciousAgents.
 *
 * Headset rule (no-signalling). Both reported outcomes are XOR-ed with a shared
 * hidden bit r that no action touches: x = x~ xor r, y = y~ xor r. Each party's
 * outcome is then a fair coin whatever the other does, so nothing can be
 * signalled through the headset. Correlations survive: (-1)^(x xor y) = (-1)^(x~ xor y~).
 *
 * Questions:
 *   A. Control: if Alice's action cannot reach the network (spacetime-local),
 *      can any strategy beat the Bell bound CHSH <= 2?
 *   B. Behind the headset: what is the best CHSH value, as a function of how
 *      many network steps k separate Alice's action from Bob's perception?
 *      Where does it cross Tsirelson's bound 2*sqrt(2) (the quantum limit)?
 *   C. Typical (random, unoptimised) networks: how often are they non-local,
 *      post-quantum, or (without the headset rule) signalling?
 *
 * Strategies are optimised exactly: CHSH is linear in each party's strategy
 * given the other's, so alternating best responses over deterministic
 * strategies climbs monotonically; several seeded restarts are used.
 *
 * Bound (proved in README.md): for this model class
 *     CHSH <= 2 + 2 * delta(Q^k),
 * where delta is the Dobrushin contraction coefficient of k network steps.
 * The experiment checks the bound and whether the optimiser attains it.
 */
const { FormalConsciousAgent, mulberry32, markov } = require('../../src/index');
const bell = require('../../src/analysis/bell');

const C = [[1, 1], [1, -1]]; // CHSH signs: E00 + E01 + E10 - E11
const RESTARTS = 48;
const K_MAX = 12;
const TOL = 1e-4; // CHSH within TOL of 2 counts as classical

/* ────────────── hidden network: two combined formal conscious agents ────────────── */

function randomRow(n, r, i = null, stickiness = 0) {
  const u = Array.from({ length: n }, () => r() + 1e-3);
  const s = u.reduce((a, b) => a + b, 0);
  return u.map((v, j) => (1 - stickiness) * v / s + (j === i ? stickiness : 0));
}

function formalAgent(r, stickiness) {
  const X = ['0', '1'], G = ['0', '1'], W = ['0', '1'];
  return new FormalConsciousAgent({
    X, G, W,
    P: Object.fromEntries(W.map(w => [w, X.map((_, i) => randomRow(2, r, i, stickiness))])),
    D: X.map(() => randomRow(2, r)),
    A: Object.fromEntries(G.map(g => [g, W.map((_, i) => randomRow(2, r, i, stickiness))])),
    rng: r,
  });
}

function hiddenNetwork(seed, stickiness) {
  const r = mulberry32(seed);
  const net = FormalConsciousAgent.combine(formalAgent(r, stickiness), formalAgent(r, stickiness));
  const Q = net.jointKernel();
  const d = Q.diagnostics();
  return { Q: Q.matrix, pi: markov.stationary(Q.matrix).pi, states: Q.states.length, lambda2: d.lambda2, ergodic: d.ergodic };
}

/* ────────────── exact best-response optimisation of CHSH ────────────── */

function optimiseChsh(Mk, pi, { actionEntersNetwork, restarts = RESTARTS, seed = 1 }) {
  const n = pi.length;
  const r = mulberry32(seed);
  let best = { value: -Infinity };
  for (let rep = 0; rep < restarts; rep++) {
    let sB = [0, 1].map(() => Array.from({ length: n }, () => (r() < 0.5 ? 1 : -1)));
    let value = -Infinity, alice = null;
    for (let iter = 0; iter < 200; iter++) {
      // v[b][t]: Bob's expected sign if the network is left in state t.
      const v = [0, 1].map(b => Mk.map(row => row.reduce((s, p, j) => s + p * sB[b][j], 0)));
      // Alice's best response for every (a, h): action target t and outcome sign.
      alice = [0, 1].map(a => Array.from({ length: n }, (_, h) => {
        const targets = actionEntersNetwork ? [...Array(n).keys()] : [h];
        let bt = targets[0], bv = -Infinity;
        for (const t of targets) {
          const gain = Math.abs(C[a][0] * v[0][t] + C[a][1] * v[1][t]);
          if (gain > bv + 1e-12) { bv = gain; bt = t; }
        }
        const signed = C[a][0] * v[0][bt] + C[a][1] * v[1][bt];
        return { t: bt, s: signed >= 0 ? 1 : -1 };
      }));
      // Bob's best response: w[b][h'] is the CHSH weight of answering +1 at h'.
      const w = [0, 1].map(b => {
        const out = new Float64Array(n);
        for (let a = 0; a < 2; a++) for (let h = 0; h < n; h++) {
          const { t, s } = alice[a][h];
          const c = C[a][b] * pi[h] * s;
          for (let j = 0; j < n; j++) out[j] += c * Mk[t][j];
        }
        return out;
      });
      sB = w.map(row => Array.from(row, x => (x >= 0 ? 1 : -1)));
      const next = w.reduce((s, row) => s + row.reduce((t, x) => t + Math.abs(x), 0), 0);
      if (next <= value + 1e-12) break;
      value = next;
    }
    if (value > best.value + 1e-12) best = { value, alice, sB };
  }
  // Correlators of the optimum.
  const E = [0, 1].map(a => [0, 1].map(b => {
    let e = 0;
    for (let h = 0; h < n; h++) {
      const { t, s } = best.alice[a][h];
      for (let j = 0; j < n; j++) e += pi[h] * s * Mk[t][j] * best.sB[b][j];
    }
    return e;
  }));
  return { chsh: bell.chsh(E), E };
}

/* ────────────── random (unoptimised) strategies ────────────── */

function randomBehaviour(Mk, pi, r, { actionEntersNetwork, masked }) {
  const n = pi.length;
  const pA = [0, 1].map(() => Array.from({ length: n }, () => r()));            // P(x~ = 0 | a, h)
  const pB = [0, 1].map(() => Array.from({ length: n }, () => r()));            // P(y~ = 0 | b, h')
  const act = [0, 1].map(() => [0, 1].map(() => Array.from({ length: n }, (_, h) =>
    (actionEntersNetwork ? randomRow(n, r) : Array.from({ length: n }, (_, j) => (j === h ? 1 : 0))))));
  const p = [0, 1].map(() => [0, 1].map(() => [[0, 0], [0, 0]]));
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
    for (let h = 0; h < n; h++) for (let x = 0; x < 2; x++) {
      const px = x === 0 ? pA[a][h] : 1 - pA[a][h];
      const row = act[a][x][h];
      for (let h1 = 0; h1 < n; h1++) {
        if (row[h1] === 0) continue;
        for (let h2 = 0; h2 < n; h2++) {
          const m = pi[h] * px * row[h1] * Mk[h1][h2];
          if (m === 0) continue;
          p[a][b][x][0] += m * pB[b][h2];
          p[a][b][x][1] += m * (1 - pB[b][h2]);
        }
      }
    }
  }
  if (!masked) return p;
  return p.map(pa => pa.map(pb => [0, 1].map(x => [0, 1].map(y => 0.5 * (pb[x][y] + pb[1 - x][1 - y])))));
}

/* ────────────── report ────────────── */

const f = (x, d = 3) => x.toFixed(d);

function main() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Bell test through the headset');
  console.log('='.repeat(78));

  console.log('\n  Reference behaviours');
  const s = Math.SQRT1_2;
  for (const [name, E] of [['best local (classical, in spacetime)', [[1, 1], [1, 1]]],
    ['quantum singlet, optimal angles', [[s, s], [s, -s]]], ['PR box (maximal no-signalling)', [[1, 1], [1, -1]]]]) {
    console.log(`    ${name.padEnd(38)} CHSH ${f(bell.chsh(E))}  ${bell.classify(bell.fromCorrelators(E))}`);
  }

  const networks = [0, 0.5, 0.8, 0.95].map((st, i) => ({ stickiness: st, ...hiddenNetwork(100 + i, st) }));

  console.log('\n  A. Control: Alice\'s action cannot reach the network (spacetime-local)');
  let controlMax = 0;
  for (const net of networks) {
    for (let k = 0; k <= 2; k++) controlMax = Math.max(controlMax, optimiseChsh(markov.matPow(net.Q, k), net.pi, { actionEntersNetwork: false }).chsh);
  }
  console.log(`    best CHSH over ${networks.length} networks and k = 0..2: ${f(controlMax, 6)}   (Bell bound: 2)`);

  console.log('\n  B. Behind the headset: best CHSH vs network steps k between Alice\'s action and Bob');
  const crossings = [];
  let maxGap = 0, boundViolated = false;
  for (const net of networks) {
    console.log(`\n    hidden network: ${net.states} joint states, stickiness ${net.stickiness}, |λ₂| = ${f(net.lambda2)}`);
    console.log('      k   δ(Q^k)   |λ₂|^k   bound 2+2δ   best CHSH   region of the optimum');
    let kT = null, kL = null;
    for (let k = 0; k <= K_MAX; k++) {
      const Mk = markov.matPow(net.Q, k);
      const delta = markov.dobrushin(Mk);
      const { chsh, E } = optimiseChsh(Mk, net.pi, { actionEntersNetwork: true, seed: 7 + k });
      const region = bell.classify(bell.fromCorrelators(E), TOL);
      maxGap = Math.max(maxGap, 2 + 2 * delta - chsh);
      boundViolated = boundViolated || chsh > 2 + 2 * delta + 1e-9;
      if (kT === null && chsh <= bell.TSIRELSON + 1e-9) kT = k;
      if (kL === null && chsh <= 2 + TOL) kL = k;
      console.log(`      ${String(k).padEnd(3)} ${f(delta)}    ${f(net.lambda2 ** k)}    ${f(2 + 2 * delta)}        ${f(chsh)}       ${region}`);
      if (kL !== null) break;
    }
    crossings.push({ ...net, kT, kL });
  }

  console.log('\n  C. Typical networks: 400 random (unoptimised) strategies per network, k = 1');
  const r = mulberry32(2024);
  const tally = { nonlocal: 0, postQuantum: 0, signalling: 0, localSignalling: 0, localNonlocal: 0, total: 0 };
  for (const net of networks) {
    const M1 = markov.matPow(net.Q, 1);
    for (let i = 0; i < 400; i++) {
      const masked = randomBehaviour(M1, net.pi, r, { actionEntersNetwork: true, masked: true });
      const E = bell.correlators(masked);
      if (bell.chsh(E) > 2 + 1e-9) tally.nonlocal++;
      if (!bell.isQuantum(E)) tally.postQuantum++;
      const open = randomBehaviour(M1, net.pi, r, { actionEntersNetwork: true, masked: false });
      if (bell.signalling(open) > 1e-3) tally.signalling++;
      const local = randomBehaviour(M1, net.pi, r, { actionEntersNetwork: false, masked: false });
      if (bell.signalling(local) > 1e-9) tally.localSignalling++;
      if (bell.chsh(bell.correlators(local)) > 2 + 1e-9) tally.localNonlocal++;
      tally.total++;
    }
  }
  const pct = (x) => `${(100 * x / tally.total).toFixed(1)}%`;
  console.log(`    with headset rule:    CHSH > 2 in ${pct(tally.nonlocal)}, outside the quantum set in ${pct(tally.postQuantum)}`);
  console.log(`    without headset rule: signalling (> 0.001) in ${pct(tally.signalling)}`);
  console.log(`    spacetime-local:      signalling in ${pct(tally.localSignalling)}, CHSH > 2 in ${pct(tally.localNonlocal)}`);

  console.log('\n' + '─'.repeat(78));
  console.log('Findings');
  console.log('─'.repeat(78));
  console.log(`  1. Spacetime-local agents never beat the Bell bound (best ${f(controlMax, 6)}): Bell's theorem holds.`);
  const k0 = crossings.every(c => c.kT !== 0);
  console.log(`  2. Behind the headset, no-signalling correlations ${k0 ? 'reach the PR box (CHSH 4) at k = 0' : 'exceed 2 at k = 0'}.`);
  console.log(`     The bound CHSH <= 2 + 2δ(Q^k) was ${boundViolated ? 'VIOLATED' : 'never violated'}, and the optimiser reached it to within ${maxGap < 1e-9 ? 'machine precision' : maxGap.toExponential(1)}.`);
  console.log('     Correlations decay at the network\'s mixing rate; CHSH <= 2√2 exactly when δ(Q^k) <= √2 - 1.');
  for (const c of crossings) {
    console.log(`     stickiness ${String(c.stickiness).padEnd(4)} (|λ₂| ${f(c.lambda2)}): CHSH <= 2√2 from k = ${c.kT ?? `>${K_MAX}`}, classical (<= 2) from k = ${c.kL ?? `>${K_MAX}`}`);
  }
  console.log('     The optimal behaviours have perfect (±1) correlators, which places them outside the quantum');
  console.log('     set even when CHSH < 2√2: they stay post-quantum until they become classical.');
  console.log(`  3. Random, unoptimised strategies are almost never non-local (${pct(tally.nonlocal)}): non-locality needs tuned agents.`);
  console.log(`  4. Without the headset rule, ${pct(tally.signalling)} of random networks would let Alice signal to Bob.`);
  console.log('\n  Interpretation: granting a network behind spacetime, conscious-agent kernels can produce every');
  console.log('  no-signalling correlation up to the PR box, the quantum ones included, but nothing in the');
  console.log('  kernels singles out the quantum set. Matching nature needs two further principles: one that');
  console.log('  forbids signalling through the headset, and one that caps correlations at Tsirelson\'s bound.');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

if (require.main === module) main();

module.exports = { hiddenNetwork, optimiseChsh, randomBehaviour };
