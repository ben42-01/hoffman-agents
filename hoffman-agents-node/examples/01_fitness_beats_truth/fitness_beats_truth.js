/**
 * Fitness Beats Truth — Hoffman's Interface Theory, Experiment 1
 *
 * Two agents experience the same hidden Markov world through different
 * interfaces: one sees compressed groups (5 groups), the other sees
 * all 20 raw states. The interface agent should predict BETTER.
 */
const { ConsciousAgent } = require('../../src/index');
const { WorldBuilder } = require('../../src/index');

function buildDirectionMap(stateSequence, dirLabels) {
  const map = {};
  for (let i = 0; i < stateSequence.length; i++) {
    const sid = stateSequence[i];
    const d = dirLabels[i];
    if (d === null || d === undefined) continue;
    if (!map[sid]) map[sid] = { up: 0, down: 0, same: 0 };
    map[sid][d]++;
  }
  return map;
}

function predictDir(sid, map) {
  const counts = map[sid];
  if (!counts) return null;
  let best = 'up', bestCount = 0;
  for (const [d, c] of Object.entries(counts)) {
    if (c > bestCount) { best = d; bestCount = c; }
  }
  return best;
}

function run() {
  console.log('='.repeat(60));
  console.log('Experiment 1: Fitness Beats Truth');
  console.log('='.repeat(60));

  // Build a hidden Markov world: 5 groups x 4 states
  const nGroups = 5, statesPerGroup = 4;
  const nStates = nGroups * statesPerGroup;

  // Generate sequence
  const seq = [];
  let current = 0;
  for (let i = 0; i < 2000; i++) {
    const r = Math.random();
    if (r < 0.5) {
      current = current;
    } else if (r < 0.8) {
      const groupStart = Math.floor(current / statesPerGroup) * statesPerGroup;
      const groupEnd = groupStart + statesPerGroup;
      current = groupStart + Math.floor(Math.random() * statesPerGroup);
    } else {
      current = Math.floor(Math.random() * nStates);
    }
    seq.push(current);
  }

  // Interface world: compress to group IDs
  const ifaceSeq = seq.map(s => Math.floor(s / statesPerGroup));

  // Train and evaluate both agents
  const split = Math.floor(seq.length * 0.8);

  function evaluate(name, stateSeq) {
    // Direction labels for evaluation: next state direction
    const dirLabels = [];
    for (let i = 0; i < stateSeq.length - 1; i++) {
      const diff = stateSeq[i + 1] - stateSeq[i];
      if (Math.abs(diff) <= 0) dirLabels.push('same');
      else if (diff > 0) dirLabels.push('up');
      else dirLabels.push('down');
    }
    dirLabels.push(null);

    const trainSeq = stateSeq.slice(0, split);
    const testSeq = stateSeq.slice(split);

    // Build direction map from training data
    const dirMap = buildDirectionMap(trainSeq, dirLabels.slice(0, split));

    // Evaluate on test
    let correct = 0, total = 0;
    for (let i = 0; i < testSeq.length; i++) {
      const actual = dirLabels[split + i];
      if (actual === null) continue;
      const pred = predictDir(testSeq[i], dirMap);
      if (pred) { total++; if (pred === actual) correct++; }
    }
    const acc = total > 0 ? correct / total : 0;
    const improvement = (acc - 1/3) / (1/3) * 100;
    console.log(`  ${name.padEnd(25)} acc: ${(acc*100).toFixed(1)}%  improvement: ${improvement.toFixed(1)}%`);
    return { acc, improvement };
  }

  const iface = evaluate('Interface (5 groups)', ifaceSeq);
  const truth = evaluate('Truth (20 raw states)', seq);

  if (iface.improvement > truth.improvement) {
    console.log(`\n  ✓ Interface dominates — less information → better prediction`);
  } else {
    console.log(`\n  ✗ Expected Interface > Truth`);
  }
}

run();
