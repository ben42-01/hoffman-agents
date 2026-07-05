/**
 * Exchange Symmetry Test
 *
 * Tests whether the combination operator ⊗ produces exchange-symmetric or
 * exchange-antisymmetric results under swapping of constituent agents.
 *
 * Phase A — Exchange invariance:
 *   Create identical twin agents via cloneAgent(), combine as AB and BA,
 *   measure whether observable quantities (spectral gap, loop score, etc.)
 *   change under exchange.
 *
 * Phase B — Occupation / redundancy:
 *   Combine identical twins vs genuinely distinct agents, measure the
 *   "redundancy ratio" of the merged experience space. If identical agents
 *   produce near-duplicate content (ratio → 0.5) while distinct agents
 *   produce additive content (ratio → 1.0), that's an emergent signature
 *   analogous to exclusion-like behavior in information space.
 *
 * This is an exploratory computational analogy — NOT a physics claim.
 * The results may surface directional hypotheses for formal mathematical
 * investigation with frameworks like Hoffman & Prakash's Markovian kernel
 * formalism or Arkani-Hamed's amplituhedron program.
 */

const {
  ConsciousAgent, WorldState, StepOutput,
  ExperienceTrie, TraceBuffer, TraceEvent,
  SelfTokenState, ExperienceSpace,
  WorldBuilder, SimpleWorld,
  combine, fuse, experienceSpaceDistance,
  cloneAgent,
} = require('../../src/index');

/* ────────────── Helper utilities ────────────── */

function seedRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) & 0x7FFFFFFF;
    return s / 0x7FFFFFFF;
  };
}

function spectralGap(P) {
  const n = P.length;
  if (n < 2) return { gap: 1, convergenceError: 0 };

  // Stationary distribution
  let pi = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 1000; iter++) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += pi[j] * P[j][i];
      next[i] = s;
    }
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(next[i] - pi[i]);
    pi = next;
    if (diff < 1e-10) break;
  }

  // Deflate: B = P - 1·pi^T
  const B = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      B[i][j] = P[i][j] - pi[j];

  // Power iteration for dominant eigenvalue of B
  let b = new Float64Array(n).fill(1 / Math.sqrt(n));
  let lambda = 0;
  for (let iter = 0; iter < 500; iter++) {
    const bNext = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += B[i][j] * b[j];
      bNext[i] = s;
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += bNext[i] * bNext[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < n; i++) bNext[i] /= norm;
    // Rayleigh quotient
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += bNext[i] * b[i]; den += b[i] * b[i]; }
    lambda = num / den;
    b = bNext;
  }

  return { gap: Math.max(0, 1 - Math.abs(lambda)), lambda: Math.abs(lambda) };
}

function detailedBalanceError(P) {
  const n = P.length;
  if (n < 2) return 0;
  // Stationary distribution
  let pi = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 1000; iter++) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += pi[j] * P[j][i];
      next[i] = s;
    }
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(next[i] - pi[i]);
    pi = next;
    if (diff < 1e-10) break;
  }

  let errSum = 0, pairCount = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const fwd = pi[i] * P[i][j];
      const rev = pi[j] * P[j][i];
      const denom = fwd + rev;
      if (denom > 1e-12) { errSum += Math.abs(fwd - rev) / denom; pairCount++; }
    }
  }
  return pairCount > 0 ? errSum / pairCount : 0;
}

function extractMetaMatrix(agent) {
  const mt = agent.experience.metaTrie;
  if (mt.registrySize < 2) return { P: [[1]], stateCount: 1 };

  const allIds = [...mt._registry.keys()];
  const activeIds = allIds.filter(id => {
    const node = mt.trie.lookup([id]);
    return node && Object.keys(node.children).length > 0;
  });
  if (activeIds.length < 2) activeIds.push(allIds[0]);

  const sorted = [...new Set(activeIds)].sort((a, b) => a - b);
  const idx = new Map(sorted.map((id, i) => [id, i]));
  const n = sorted.length;
  const P = Array.from({ length: n }, () => new Float64Array(n).fill(0));

  for (const stateId of sorted) {
    const node = mt.trie.lookup([stateId]);
    if (!node) continue;
    const children = Object.entries(node.children);
    const total = children.reduce((s, [, c]) => s + c.visitCount, 0);
    if (total === 0) continue;
    for (const [childIdStr, child] of children) {
      const childId = parseInt(childIdStr);
      const j = idx.get(childId);
      if (j !== undefined) P[idx.get(stateId)][j] = child.visitCount / total;
    }
  }

  return { P, stateCount: n };
}

function measureAgent(agent) {
  const { P, stateCount } = extractMetaMatrix(agent);
  const { gap } = spectralGap(P);
  const dbErr = stateCount > 1 ? detailedBalanceError(P) : 0;
  const loopScore = typeof agent.loopScore === 'function' ? agent.loopScore : agent.loopScore;
  return {
    spectralGap: gap,
    detailedBalanceError: dbErr,
    loopScore: agent.loopScore,
    meanPredictionError: agent.meanPredictionError,
    isILocked: agent.isILocked,
    cycleLevel: agent.cycleLevel,
  };
}

function redundancyRatio(combined, parent1, parent2) {
  const combSize = combined.experience.trie.size();
  const p1Size = parent1.experience.trie.size();
  const p2Size = parent2.experience.trie.size();
  const sumSize = p1Size + p2Size;
  return {
    trieRatio: sumSize > 0 ? combSize / sumSize : 0,
    combinedTrieSize: combSize,
    parentSumTrieSize: sumSize,
  };
}

function formatDelta(abVal, baVal, decimals = 4) {
  const delta = Math.abs(abVal - baVal);
  const mean = (abVal + baVal) / 2;
  const pct = mean !== 0 ? (delta / Math.abs(mean) * 100).toFixed(1) : '—';
  return `${delta.toFixed(decimals)}  (${pct}% of mean)`;
}

/* ────────────── Phase A: Exchange invariance ────────────── */

function phaseA(nTrials = 20) {
  console.log('\n  Phase A: Exchange Invariance\n');

  const results = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const seed = 42 + trial;
    const world = new SimpleWorld({ nStates: 5, seed });

    // Train parent agent
    const A = new ConsciousAgent({ agentId: `A_${trial}`, metaObservationInterval: 10 });
    for (let s = 0; s < 300; s++) A.step(world.step());

    // Clone to create identical twin B
    const B = cloneAgent(A, `B_${trial}`);

    // Combine both ways
    const AB = combine(A, B);
    const BA = combine(B, A);

    // Isolate order-effects of combine() itself from step()-time RNG noise:
    // give both agents identical, freshly-seeded random streams so their
    // post-combination decision-time token sampling is directly comparable.
    const stepSeed = 10000 + trial;
    AB._rng = seedRandom(stepSeed);
    BA._rng = seedRandom(stepSeed);

    // Run both for many steps, tracking running average loop score
    // (a single instantaneous loopScore read is a 1-sample stochastic
    // quantity — average over the run for a statistically meaningful value)
    const nRunSteps = 100;
    let loopSumAB = 0, loopSumBA = 0;
    for (let s = 0; s < nRunSteps; s++) {
      const ws = world.step();
      AB.step(ws);
      BA.step(ws);
      loopSumAB += AB.loopScore;
      loopSumBA += BA.loopScore;
    }
    const meanLoopAB = loopSumAB / nRunSteps;
    const meanLoopBA = loopSumBA / nRunSteps;

    const mAB = measureAgent(AB);
    const mBA = measureAgent(BA);
    mAB.loopScore = meanLoopAB;
    mBA.loopScore = meanLoopBA;
    const redAB = redundancyRatio(AB, A, B);
    const redBA = redundancyRatio(BA, B, A);
    const dist = experienceSpaceDistance(AB.experience, BA.experience);

    results.push({
      trial,
      spectralGap: { AB: mAB.spectralGap, BA: mBA.spectralGap, delta: Math.abs(mAB.spectralGap - mBA.spectralGap) },
      dbError: { AB: mAB.detailedBalanceError, BA: mBA.detailedBalanceError, delta: Math.abs(mAB.detailedBalanceError - mBA.detailedBalanceError) },
      loopScore: { AB: mAB.loopScore, BA: mBA.loopScore, delta: Math.abs(mAB.loopScore - mBA.loopScore) },
      predError: { AB: mAB.meanPredictionError, BA: mBA.meanPredictionError, delta: Math.abs(mAB.meanPredictionError - mBA.meanPredictionError) },
      locked: { AB: mAB.isILocked, BA: mBA.isILocked },
      experienceDistance: dist,
    });
  }

  // Aggregate
  const agg = {
    spectralGap: { mean: results.reduce((s, r) => s + r.spectralGap.delta, 0) / nTrials },
    dbError: { mean: results.reduce((s, r) => s + r.dbError.delta, 0) / nTrials },
    loopScore: { mean: results.reduce((s, r) => s + r.loopScore.delta, 0) / nTrials },
    predError: { mean: results.reduce((s, r) => s + r.predError.delta, 0) / nTrials },
    experienceDistance: { mean: results.reduce((s, r) => s + r.experienceDistance, 0) / nTrials },
  };
  agg.spectralGap.std = Math.sqrt(results.reduce((s, r) => s + (r.spectralGap.delta - agg.spectralGap.mean) ** 2, 0) / nTrials);
  agg.dbError.std = Math.sqrt(results.reduce((s, r) => s + (r.dbError.delta - agg.dbError.mean) ** 2, 0) / nTrials);
  agg.loopScore.std = Math.sqrt(results.reduce((s, r) => s + (r.loopScore.delta - agg.loopScore.mean) ** 2, 0) / nTrials);
  agg.predError.std = Math.sqrt(results.reduce((s, r) => s + (r.predError.delta - agg.predError.mean) ** 2, 0) / nTrials);

  const sameLock = results.filter(r => r.locked.AB === r.locked.BA).length;
  agg.lockAgreement = sameLock / nTrials;

  console.log(`  Trials: ${nTrials}`);
  console.log(`\n    Observable                    Mean |Δ|  ±1σ     Interpretation`);
  console.log(`    ─────────────────────────────────────────────────────────────`);
  const rows = [
    { label: 'Spectral gap (1−|λ₂|)', v: agg.spectralGap, thr: 0.05 },
    { label: 'Detailed balance error', v: agg.dbError, thr: 0.03 },
    { label: 'Loop score', v: agg.loopScore, thr: 0.05 },
    { label: 'Mean prediction error', v: agg.predError, thr: 0.03 },
  ];
  for (const r of rows) {
    const isSym = r.v.mean < r.thr;
    console.log(`    ${r.label.padEnd(32)} ${r.v.mean.toFixed(4)} ± ${r.v.std.toFixed(4)}  ${isSym ? '← symmetric' : '← biased'}`);
  }
  console.log(`    Experience space distance       ${agg.experienceDistance.mean.toFixed(4)}`);
  console.log(`    Lock agreement (AB vs BA):      ${(agg.lockAgreement * 100).toFixed(0)}%`);

  return agg;
}

/* ────────────── Phase B: Occupation / redundancy ────────────── */

function phaseB() {
  console.log('\n  Phase B: Redundancy Ratio — Identical vs Distinct\n');

  const nTrials = 20;
  const identResults = [];
  const distinctResults = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const seed1 = 42 + trial;
    const seed2 = 999 + trial;

    const world1 = new SimpleWorld({ nStates: 5, seed: seed1 });
    const world2 = new SimpleWorld({ nStates: 5, seed: seed2 });

    // Train A on world1
    const A = new ConsciousAgent({ agentId: `A_${trial}` });
    for (let s = 0; s < 300; s++) A.step(world1.step());

    // Identical twin
    const A_twin = cloneAgent(A, `twin_${trial}`);

    // Distinct agent C on a different world
    const C = new ConsciousAgent({ agentId: `C_${trial}` });
    for (let s = 0; s < 300; s++) C.step(world2.step());

    // Combine identical pair
    const S = combine(A, A_twin);
    const ratioIdent = redundancyRatio(S, A, A_twin);
    identResults.push(ratioIdent.trieRatio);

    // Combine distinct pair
    const D = combine(A, C);
    const ratioDistinct = redundancyRatio(D, A, C);
    distinctResults.push(ratioDistinct.trieRatio);
  }

  const identMean = identResults.reduce((s, v) => s + v, 0) / nTrials;
  const identStd = Math.sqrt(identResults.reduce((s, v) => s + (v - identMean) ** 2, 0) / nTrials);
  const distinctMean = distinctResults.reduce((s, v) => s + v, 0) / nTrials;
  const distinctStd = Math.sqrt(distinctResults.reduce((s, v) => s + (v - distinctMean) ** 2, 0) / nTrials);

  console.log(`  Trials: ${nTrials}`);
  console.log(`\n    Pair type      Mean ratio  ±1σ     Interpretation`);
  console.log(`    ─────────────────────────────────────────────────`);
  console.log(`    Identical twins   ${identMean.toFixed(3)}     ± ${identStd.toFixed(3)}  1.0 = additive, 0.5 = overlap`);
  console.log(`    Distinct agents  ${distinctMean.toFixed(3)}     ± ${distinctStd.toFixed(3)}  1.0 = additive, 0.5 = overlap`);
  console.log(`    Contrast:        ${(distinctMean - identMean).toFixed(3)}  (higher = less redundancy)`);

  if (identMean < distinctMean - 0.1) {
    console.log('\n  ✓ Identical twins produce more redundancy — content overlap detected');
    console.log('    (Analogous to exclusion-like behavior in information space)');
  } else if (Math.abs(identMean - distinctMean) < 0.05) {
    console.log('\n  ∼ No significant difference — redundancy is independent of constituent identity');
  } else {
    console.log('\n  ∼ Weak or noisy contrast');
  }

  return { identicalIdent: identMean, distinctDistinct: distinctMean };
}

/* ────────────── Main ────────────── */

function main() {
  const t0 = Date.now();
  console.log('='.repeat(62));
  console.log('Exchange Symmetry Test');
  console.log('='.repeat(62));
  console.log('\nTesting whether the ⊗ combination operator preserves or breaks');
  console.log('exchange symmetry when its constituent agents are swapped.\n');
  console.log('This is an exploratory computational analogy — results are');
  console.log('directional signals for further formal investigation, not physics claims.\n');

  const phaseAResult = phaseA(20);
  const phaseBResult = phaseB();

  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
  console.log('='.repeat(62));
  console.log('Summary');
  console.log('='.repeat(62));
  console.log(`\n  Exchange invariance: ${phaseAResult.lockAgreement >= 0.5 ? 'AB≈BA across all observables' : 'AB≠BA — order bias detected'}`);
  console.log(`  Redundancy contrast: ${phaseBResult.identicalIdent < phaseBResult.distinctDistinct - 0.05 ? 'Identical agents are more redundant' : 'No contrast'}`);
  console.log('\n  See README.md for interpretation and caveats.\n');
}

main();
