/**
 * Experiment 6: Fitness Beats Truth — Tree of Life
 *
 * Agents inhabit a hidden Markov world with a compressed interface (groups)
 * vs veridical perception (raw states). They interact and combine.
 * Tests whether combined agents prefer simplified interfaces,
 * supporting Hoffman's claim that perception is tuned for fitness, not truth.
 */
const { ConsciousAgent, WorldState, combine } = require('../../src/index');

function buildFbtWorld(nGroups = 5, statesPerGroup = 4, seqLen = 2000) {
  const nStates = nGroups * statesPerGroup;
  const seq = [];
  let current = 0;
  for (let i = 0; i < seqLen; i++) {
    const r = Math.random();
    if (r < 0.5) {
      current = current;
    } else if (r < 0.8) {
      const gs = Math.floor(current / statesPerGroup) * statesPerGroup;
      const ge = gs + statesPerGroup;
      current = gs + Math.floor(Math.random() * statesPerGroup);
    } else {
      current = Math.floor(Math.random() * nStates);
    }
    seq.push(current);
  }
  const iface = seq.map(s => Math.floor(s / statesPerGroup));
  return { raw: seq, iface, nGroups, nStates };
}

function run() {
  console.log('='.repeat(66));
  console.log('Experiment 6: Fitness Beats Truth — Tree of Life');
  console.log('='.repeat(66));

  const world = buildFbtWorld(5, 4, 300);
  const nInterface = 4, nVeridical = 4;
  const allAgents = {};

  // Create interface agents (see compressed groups)
  for (let i = 0; i < nInterface; i++) {
    const aid = `IFACE_${String(i).padStart(3, '0')}`;
    allAgents[aid] = new ConsciousAgent({ agentId: aid });
    for (const sid of world.iface) {
      allAgents[aid].step(new WorldState({ world: [`g${sid}`] }));
    }
  }

  // Create veridical agents (see all raw states)
  for (let i = 0; i < nVeridical; i++) {
    const aid = `TRUTH_${String(i).padStart(3, '0')}`;
    allAgents[aid] = new ConsciousAgent({ agentId: aid });
    for (const sid of world.raw) {
      allAgents[aid].step(new WorldState({ world: [`s${sid}`] }));
    }
  }

  // Interaction + combination rounds
  const nRounds = 80;
  for (let rnd = 0; rnd < nRounds; rnd++) {
    const outputs = {};
    for (const [aid, ag] of Object.entries(allAgents)) outputs[aid] = ag.getOutput();
    for (const [aid, ag] of Object.entries(allAgents)) {
      for (const [oa, o] of Object.entries(outputs)) {
        if (oa !== aid) ag.step(new WorldState({ [oa]: o }));
      }
    }

    if (rnd > 0 && rnd % 20 === 0) {
      const ripe = Object.entries(allAgents)
        .filter(([, a]) => a.experience.selfToken.locked && !a._combined)
        .map(([id]) => id);
      if (ripe.length >= 2) {
        ripe.sort((a, b) =>
          allAgents[a].experience.traceBuffer.predictionErrorMean(5) -
          allAgents[b].experience.traceBuffer.predictionErrorMean(5)
        );
        for (let i = 0; i < ripe.length - 1; i += 2) {
          const c = combine(allAgents[ripe[i]], allAgents[ripe[i + 1]]);
          c.agentId = `L${c.cycleLevel}_${ripe[i]}_${ripe[i + 1]}`;
          allAgents[c.agentId] = c;
          allAgents[ripe[i]]._combined = true;
          allAgents[ripe[i + 1]]._combined = true;
        }
      }
    }
  }

  // Report
  console.log(`\n  ${'─'.repeat(60)}`);
  console.log(`  Final Agent State`);
  console.log(`  ${'─'.repeat(60)}`);
  console.log(`  ${'Agent'.padEnd(22)} ${'Type'.padEnd(10)} ${'Lvl'.padEnd(4)} ${'States'.padEnd(7)} ${'Err'.padEnd(8)} Locked`);

  const byType = { iface: [], truth: [], combined: [] };
  for (const [aid, ag] of Object.entries(allAgents).sort()) {
    const type = aid.startsWith('IFACE') ? 'iface' : aid.startsWith('TRUTH') ? 'truth' : 'combined';
    const err = ag.experience.traceBuffer.predictionErrorMean(20);
    console.log(`  ${aid.padEnd(22)} ${type.padEnd(10)} ${String(ag.cycleLevel).padEnd(4)} ${String(ag.experience.metaTrie.registrySize).padEnd(7)} ${err.toFixed(4).padEnd(8)} ${ag.experience.selfToken.locked}`);
    byType[type].push(err);
  }

  const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  console.log(`\n  Summary:`);
  console.log(`    Interface agents mean error:  ${avg(byType.iface).toFixed(4)} (see ${world.nGroups} groups)`);
  console.log(`    Veridical agents mean error:  ${avg(byType.truth).toFixed(4)} (see ${world.nStates} raw states)`);
  console.log(`    Combined agents mean error:   ${avg(byType.combined).toFixed(4)}`);
  if (avg(byType.iface) < avg(byType.truth)) {
    console.log(`  ✓ Interface dominates — less information → better prediction (Hoffman confirmed)`);
  } else {
    console.log(`  ✗ Veridical equal or better — no fitness advantage for compression`);
  }
}

run();
