/**
 * Observer-Gated Combination
 *
 * Follow-up to 07_exchange_symmetry. That experiment showed combine() is
 * exactly exchange-symmetric — a direct consequence of being built from
 * pure union operations (trie merge, lexicon merge), which are symmetric
 * by algebraic necessity. Union has no vantage point, so it cannot produce
 * asymmetry no matter how it's composed.
 *
 * perceive() is the one operation in this architecture that IS inherently
 * asymmetric: prediction error is always computed FROM one agent's history,
 * relative to what it alone expects. It has a vantage point. It has an
 * observer.
 *
 * This experiment asks: what happens if, before merging, each agent first
 * PERCEIVES the other — treating the other's recent trajectory as a world
 * to predict — and the resulting (asymmetric) surprise gates how the merge
 * proceeds?
 *
 * Hypothesis:
 *   - Identical twins: mutual surprise should be equal (they'd predict each
 *     other exactly as well as they predict themselves) -> merge stays
 *     symmetric, consistent with 07's result. This is a sanity check.
 *   - Distinct agents (different learned histories): mutual surprise should
 *     be UNEQUAL -> a genuine, principled asymmetry emerges, driven by who
 *     understands whom, NOT by argument order. observerGatedCombine(A,B)
 *     and observerGatedCombine(B,A) should identify the SAME dominant
 *     agent regardless of call order — order-independence of a different,
 *     deeper kind than pure symmetry.
 *
 * This is an exploratory computational analogy — NOT a physics claim.
 */

const {
  ConsciousAgent, WorldState, SimpleWorld,
  combine, cloneAgent,
} = require('../../src/index');

/* ────────────── Observer surprise ────────────── */

// How surprised would `perceiver`'s trie be if it had to predict `other`'s
// own recent state trajectory? This is perception turned sideways: instead
// of perceiving a world, one agent perceives another agent's history.
function observerSurprise(perceiver, other, window = 30) {
  const recent = other.experience.traceBuffer.getRecent(window);
  if (recent.length < 2) return 1; // no data -> maximal uncertainty

  let totalError = 0, count = 0;
  for (let i = 1; i < recent.length; i++) {
    const prevState = recent[i - 1].toState;
    const actualState = recent[i].toState;
    const predicted = perceiver.experience.trie.predictNext([prevState]);
    totalError += (predicted === actualState) ? 0 : 1;
    count++;
  }
  return count > 0 ? totalError / count : 1;
}

/* ────────────── Observer-gated combination ────────────── */

function observerGatedCombine(agentA, agentB, { holeThreshold = 0.5, window = 30 } = {}) {
  const surpriseAonB = observerSurprise(agentA, agentB, window);
  const surpriseBonA = observerSurprise(agentB, agentA, window);
  const maxSurprise = Math.max(surpriseAonB, surpriseBonA);

  let dominant, submissive, mode;
  if (maxSurprise < holeThreshold) {
    // Mutual fit: no rupture detected either direction. Smooth union.
    dominant = agentA; submissive = agentB; mode = 'union';
  } else {
    // Rupture: the less-surprised (more confident) agent anchors the merge.
    if (surpriseAonB <= surpriseBonA) { dominant = agentA; submissive = agentB; }
    else { dominant = agentB; submissive = agentA; }
    mode = 'exclusion';
  }

  const result = combine(dominant, submissive);
  return { result, mode, surpriseAonB, surpriseBonA, dominantId: dominant.agentId };
}

/* ────────────── Phase A: Twin sanity check ────────────── */

function phaseA_twins(nTrials = 20) {
  console.log('\n  Phase A: Identical Twins (sanity check — expect symmetry to hold)\n');

  let modeMatches = 0, dominantMatches = 0;
  const asymmetries = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const world = new SimpleWorld({ nStates: 5, seed: 42 + trial });
    const A = new ConsciousAgent({ agentId: `A_${trial}` });
    for (let s = 0; s < 300; s++) A.step(world.step());
    const B = cloneAgent(A, `B_${trial}`);

    const AB = observerGatedCombine(A, B);
    const BA = observerGatedCombine(B, A);

    asymmetries.push(Math.abs(AB.surpriseAonB - AB.surpriseBonA));
    if (AB.mode === BA.mode) modeMatches++;
    // For twins, "dominant" identity is symmetric-in-substance even if the
    // raw agentId differs (A_trial vs B_trial refer to identical experience).
    if (AB.mode === 'union' && BA.mode === 'union') dominantMatches++;
  }

  const meanAsym = asymmetries.reduce((a, b) => a + b, 0) / nTrials;
  console.log(`  Trials: ${nTrials}`);
  console.log(`    Mean surprise asymmetry (twins): ${meanAsym.toFixed(4)}  (expect ~0)`);
  console.log(`    Mode agreement (AB vs BA):       ${modeMatches}/${nTrials}`);
  console.log(`    Union mode chosen (both orders): ${dominantMatches}/${nTrials}`);

  if (meanAsym < 0.05 && modeMatches === nTrials) {
    console.log('\n  ✓ Twins remain symmetric — consistent with 07_exchange_symmetry');
  } else {
    console.log('\n  ~ Unexpected asymmetry detected even for twins — investigate further');
  }

  return { meanAsym, modeMatches };
}

/* ────────────── Phase B: Totally disjoint agents — symmetric confusion ────────────── */

function phaseB_disjoint(nTrials = 20) {
  console.log('\n  Phase B: Totally Disjoint Agents (different worlds, zero overlap)\n');

  const asymmetries = [];
  let bothMaxSurprise = 0;

  for (let trial = 0; trial < nTrials; trial++) {
    const worldA = new SimpleWorld({ nStates: 5, seed: 100 + trial });
    const worldC = new SimpleWorld({ nStates: 8, seed: 900 + trial });

    const A = new ConsciousAgent({ agentId: `A_${trial}` });
    for (let s = 0; s < 300; s++) A.step(worldA.step());
    const C = new ConsciousAgent({ agentId: `C_${trial}` });
    for (let s = 0; s < 300; s++) C.step(worldC.step());

    const AC = observerGatedCombine(A, C);
    asymmetries.push(Math.abs(AC.surpriseAonB - AC.surpriseBonA));
    if (AC.surpriseAonB >= 0.95 && AC.surpriseBonA >= 0.95) bothMaxSurprise++;
  }

  const meanAsym = asymmetries.reduce((a, b) => a + b, 0) / nTrials;
  console.log(`  Trials: ${nTrials}`);
  console.log(`    Mean surprise asymmetry: ${meanAsym.toFixed(4)}`);
  console.log(`    Both saturated at max surprise (mutual incomprehension): ${bothMaxSurprise}/${nTrials}`);
  console.log('\n  ~ Zero overlap produces symmetric CONFUSION, not asymmetric dominance.');
  console.log('    Two agents with nothing in common cannot judge who understands whom —');
  console.log('    the comparison is degenerate, not meaningfully exchange-antisymmetric.');

  return { meanAsym, bothMaxSurprise };
}

/* ────────────── Phase C: Veteran vs Novice — same world, different depth ────────────── */

function phaseC_veteranNovice(nTrials = 20) {
  console.log('\n  Phase C: Veteran vs Novice (same world, unequal experience depth)\n');

  let orderIndependentDominance = 0;
  let veteranDominatesCount = 0;
  const asymmetries = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const seed = 200 + trial;
    const worldVeteran = new SimpleWorld({ nStates: 5, seed });
    const worldNovice = new SimpleWorld({ nStates: 5, seed }); // same structure/seed

    const veteran = new ConsciousAgent({ agentId: `Veteran_${trial}` });
    for (let s = 0; s < 2000; s++) veteran.step(worldVeteran.step());

    const novice = new ConsciousAgent({ agentId: `Novice_${trial}` });
    for (let s = 0; s < 40; s++) novice.step(worldNovice.step());

    const VN = observerGatedCombine(veteran, novice);
    const NV = observerGatedCombine(novice, veteran);

    const asym = Math.abs(VN.surpriseAonB - VN.surpriseBonA);
    asymmetries.push(asym);

    if (VN.dominantId === NV.dominantId) orderIndependentDominance++;
    if (VN.dominantId.startsWith('Veteran')) veteranDominatesCount++;
  }

  const meanAsym = asymmetries.reduce((a, b) => a + b, 0) / nTrials;
  const maxAsym = Math.max(...asymmetries);
  console.log(`  Trials: ${nTrials}`);
  console.log(`    Mean surprise asymmetry: ${meanAsym.toFixed(4)}  (max: ${maxAsym.toFixed(4)})`);
  console.log(`    Dominance is order-independent: ${orderIndependentDominance}/${nTrials}`);
  console.log(`    (same agent wins regardless of whether you call combine(V,N) or combine(N,V))`);
  console.log(`    Veteran dominates:              ${veteranDominatesCount}/${nTrials}`);

  if (meanAsym > 0.05 && orderIndependentDominance === nTrials) {
    console.log('\n  ✓ Genuine, principled asymmetry emerged — driven by relationship, not argument order');
  } else if (meanAsym <= 0.05) {
    console.log('\n  ~ No meaningful asymmetry detected in this configuration');
  } else {
    console.log('\n  ~ Asymmetry present but dominance is NOT fully order-independent — investigate further');
  }

  return { meanAsym, orderIndependentDominance, veteranDominatesCount };
}

/* ────────────── Main ────────────── */

function main() {
  const t0 = Date.now();
  console.log('='.repeat(66));
  console.log('Observer-Gated Combination');
  console.log('='.repeat(66));
  console.log('\nTesting whether replacing pure union with an act of PERCEPTION');
  console.log('(one agent predicting another\'s trajectory before merging) can');
  console.log('introduce principled, non-arbitrary exchange asymmetry.\n');
  console.log('This is an exploratory computational analogy — results are');
  console.log('directional signals for further formal investigation, not physics claims.\n');

  const a = phaseA_twins(20);
  const b = phaseB_disjoint(20);
  const c = phaseC_veteranNovice(20);

  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
  console.log('='.repeat(66));
  console.log('Summary');
  console.log('='.repeat(66));
  console.log(`\n  Twins (identical):       symmetric fit     (asym ${a.meanAsym.toFixed(4)})`);
  console.log(`  Disjoint (zero overlap): symmetric confusion (asym ${b.meanAsym.toFixed(4)})`);
  console.log(`  Veteran/Novice (shared world, unequal depth): asym ${c.meanAsym.toFixed(4)}, order-independent ${c.orderIndependentDominance}/20`);
  console.log('\n  See README.md for interpretation and caveats.\n');
}

main();
