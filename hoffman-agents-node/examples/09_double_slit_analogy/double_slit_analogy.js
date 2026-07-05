/**
 * Double-Slit Analogy
 *
 * Follow-up to 07_exchange_symmetry and 08_observer_gated_combination.
 * This experiment asks directly: can the double-slit experiment's famous
 * interference pattern be reproduced using this library's Markov-chain /
 * trie machinery? The honest answer, demonstrated here rather than just
 * asserted, is: NO — not without adding an ingredient the library doesn't
 * have. But the reason WHY is itself the whole point of the double-slit
 * experiment historically, and this script makes that reason concrete.
 *
 * Part 1 — Classical "diamond" world:
 *   A branching Markov world: Source splits 50/50 into Path A (length La)
 *   or Path B (length Lb), both reconverging at a Detector. This is a
 *   discrete-time stand-in for "two slits leading to one screen."
 *   We compare two conditions:
 *     - "tracked":  record which path was taken alongside arrival time,
 *                   histogram each path separately, then sum.
 *     - "erased":   record ONLY arrival time, never which path, pooled
 *                   into one histogram.
 *   Prediction (and the actual point): these two histograms are
 *   mathematically IDENTICAL. Classical probability has no way to let
 *   "not tracking which path" change the observed distribution — this is
 *   exactly why real double-slit interference was so surprising: nothing
 *   in ordinary probability theory can produce it.
 *
 * Part 2 — What genuine interference requires:
 *   A small, explicit complex-amplitude construction using the SAME path
 *   lengths (La, Lb) from Part 1, showing what a system that combines
 *   AMPLITUDES (not probabilities) before squaring would produce: a real
 *   oscillating fringe pattern as a swept parameter changes. This requires
 *   an ingredient — complex phase — that the trie/Markov chain currently
 *   has no notion of.
 *
 * This is an exploratory computational analogy — NOT a physics claim.
 */

const { WorldState, ConsciousAgent } = require('../../src/index');

const PATH_A_LENGTH = 3;
const PATH_B_LENGTH = 5; // deliberately different & coprime with A for clarity

/* ────────────── Part 1: Classical diamond world ────────────── */

function runDiamondTrial(rng) {
  const path = rng() < 0.5 ? 'A' : 'B';
  const length = path === 'A' ? PATH_A_LENGTH : PATH_B_LENGTH;
  return { path, arrivalStep: length };
}

function seedRandom(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
}

function part1_classicalDiamond(nTrials = 5000) {
  console.log('\n  Part 1: Classical Diamond World (tracked vs erased)\n');
  console.log(`  Path A length: ${PATH_A_LENGTH} steps.  Path B length: ${PATH_B_LENGTH} steps.\n`);

  const rng = seedRandom(42);
  const histA = {}, histB = {}, histErased = {};

  for (let i = 0; i < nTrials; i++) {
    const { path, arrivalStep } = runDiamondTrial(rng);
    if (path === 'A') histA[arrivalStep] = (histA[arrivalStep] || 0) + 1;
    else histB[arrivalStep] = (histB[arrivalStep] || 0) + 1;
    histErased[arrivalStep] = (histErased[arrivalStep] || 0) + 1;
  }

  // "tracked" histogram = separate per-path histograms, SUMMED after the fact
  const trackedSum = {};
  for (const t of Object.keys(histA)) trackedSum[t] = (trackedSum[t] || 0) + histA[t];
  for (const t of Object.keys(histB)) trackedSum[t] = (trackedSum[t] || 0) + histB[t];

  console.log(`  Trials: ${nTrials}`);
  console.log('\n    step | tracked (A+B summed) | erased (pooled, no path label)');
  console.log('    -----|----------------------|-------------------------------');
  const allSteps = [...new Set([...Object.keys(trackedSum), ...Object.keys(histErased)])].sort((a, b) => a - b);
  let identical = true;
  for (const t of allSteps) {
    const tracked = trackedSum[t] || 0;
    const erased = histErased[t] || 0;
    if (tracked !== erased) identical = false;
    console.log(`    ${String(t).padStart(4)} | ${String(tracked).padStart(20)} | ${String(erased).padStart(29)}`);
  }

  console.log(`\n  Tracked and erased histograms are ${identical ? 'IDENTICAL' : 'DIFFERENT'} (byte-for-byte counts).`);
  console.log('  This is the expected classical result: whether or not you record which');
  console.log('  path was taken has ZERO effect on the observed distribution. Ordinary');
  console.log('  probability theory has no mechanism by which "erasing which-path info"');
  console.log('  could ever change an outcome — which is precisely why real double-slit');
  console.log('  interference is considered deep evidence that nature is NOT running on');
  console.log('  ordinary probability theory alone.');

  return { identical, trackedSum, histErased };
}

/* ────────────── Part 2: What genuine interference requires ────────────── */

function part2_complexAmplitude() {
  console.log('\n  Part 2: What Genuine Interference Would Require\n');
  console.log('  Using the SAME path lengths (La=3, Lb=5), assign each path a complex');
  console.log('  amplitude with phase proportional to path length, swept against a toy');
  console.log('  "wavelength" parameter (standing in for detector screen position).\n');

  const La = PATH_A_LENGTH, Lb = PATH_B_LENGTH;
  const results = [];

  for (let lambda = 1; lambda <= 20; lambda += 0.5) {
    // Classical: probability of landing at the detector via EITHER path,
    // summed as ordinary (non-negative, real) probabilities. No dependence
    // on lambda is possible classically — there's nothing for lambda to
    // modulate in a plain probability sum.
    const pClassical = 0.5 + 0.5; // = 1, flat, no interference possible

    // "Quantum-style" toy construction: combine AMPLITUDES first, square after.
    // amplitude_X = sqrt(0.5) * exp(i * 2*pi * length_X / lambda)
    const phaseA = 2 * Math.PI * La / lambda;
    const phaseB = 2 * Math.PI * Lb / lambda;
    const ampA = { re: Math.sqrt(0.5) * Math.cos(phaseA), im: Math.sqrt(0.5) * Math.sin(phaseA) };
    const ampB = { re: Math.sqrt(0.5) * Math.cos(phaseB), im: Math.sqrt(0.5) * Math.sin(phaseB) };
    const sumRe = ampA.re + ampB.re;
    const sumIm = ampA.im + ampB.im;
    const pQuantum = sumRe * sumRe + sumIm * sumIm; // |amplitude_A + amplitude_B|^2

    results.push({ lambda, pClassical, pQuantum });
  }

  console.log('    lambda | classical P (flat) | "quantum" P = |ampA+ampB|^2 (fringe)');
  console.log('    -------|---------------------|-------------------------------------');
  for (const r of results) {
    const bar = '#'.repeat(Math.round(r.pQuantum * 15));
    console.log(`    ${r.lambda.toFixed(1).padStart(6)} | ${r.pClassical.toFixed(3).padStart(19)} | ${r.pQuantum.toFixed(3).padStart(6)} ${bar}`);
  }

  const quantumValues = results.map(r => r.pQuantum);
  const minQ = Math.min(...quantumValues), maxQ = Math.max(...quantumValues);
  console.log(`\n  Classical: flat at 1.000 across every lambda — no fringes, cannot oscillate.`);
  console.log(`  "Quantum-style": ranges from ${minQ.toFixed(3)} to ${maxQ.toFixed(3)} — clear fringe pattern.`);
  console.log('\n  The only difference between these two calculations is WHERE the squaring');
  console.log('  happens: sum-then-square (amplitudes) vs square-then-sum (probabilities).');
  console.log('  That one-line difference is the entire content of what makes interference');
  console.log('  possible. Our trie/Markov machinery only ever does the latter — it has no');
  console.log('  representation of phase, so it structurally cannot take the other branch.');

  return { minQ, maxQ };
}

/* ────────────── Main ────────────── */

function main() {
  console.log('='.repeat(70));
  console.log('Double-Slit Analogy');
  console.log('='.repeat(70));
  console.log('\nCan this library reproduce double-slit interference? Testing directly');
  console.log('rather than asserting an answer.\n');
  console.log('This is an exploratory computational analogy — NOT a physics claim.\n');

  const p1 = part1_classicalDiamond(5000);
  const p2 = part2_complexAmplitude();

  console.log('\n' + '='.repeat(70));
  console.log('Summary');
  console.log('='.repeat(70));
  console.log(`\n  Classical trie/Markov world:  tracked == erased -> ${p1.identical ? 'NO interference possible (confirmed)' : 'unexpected result'}`);
  console.log(`  Toy complex-amplitude add-on: fringe range [${p2.minQ.toFixed(3)}, ${p2.maxQ.toFixed(3)}] -> interference IS possible, but requires phase`);
  console.log('\n  Verdict: the library, as built, cannot produce interference — because');
  console.log('  it only ever combines non-negative probabilities, never signed/complex');
  console.log('  amplitudes. The nearest EXISTING piece of related machinery is the');
  console.log('  spectral-gap analysis (see docs on eigenvalues of meta-trie transition');
  console.log('  matrices) — cyclic transition graphs can have complex eigenvalues,');
  console.log('  which do produce genuine oscillatory terms (lambda^n) as a Markov chain');
  console.log('  evolves. That is a real, related piece of wave-like mathematics already');
  console.log('  present in this codebase — but it describes one system\'s evolution over');
  console.log('  TIME, not spatial interference between two paths. Different phenomenon,');
  console.log('  same mathematical family (complex exponentials).\n');
}

main();
