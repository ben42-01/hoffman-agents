/**
 * Self-Reference ON/OFF Contrast — The Cleanest Causal Result
 *
 * One parameter changes everything:
 *   ON (threshold=0.25): 100% agents lock "I"
 *   OFF (threshold=1.5):   0% agents ever lock
 *
 * Without the "I" attractor mechanism, agents don't develop identity.
 */
const { ConsciousAgent, WorldState, SelfTokenState, ExperienceSpace } = require('../../src/index');

function strangeLoopScore(agent) {
  const out = agent._lastOutput;
  const nI = out.filter(t => t === 'I').length;
  return nI === 0 ? 0 : Math.min(0.5 * nI, 2);
}

function runCondition(name, lockThreshold, nAgents = 8, nSteps = 500) {
  console.log(`\n  ── ${name} ──`);
  const agents = [];
  for (let i = 0; i < nAgents; i++) {
    const st = new SelfTokenState({ lockThreshold });
    const exp = new ExperienceSpace({ selfToken: st });
    agents.push(new ConsciousAgent({ agentId: `Agent_${String(i).padStart(2, '0')}`, experience: exp }));
  }

  const lockGenerations = new Set();
  const outputsLog = {};

  for (let step = 0; step < nSteps; step++) {
    for (let i = 0; i < agents.length; i++) {
      const ws = WorldState.fromSequence('world', [`state_${step % 10}`]);
      const out = agents[i].step(ws);
      if (!outputsLog[agents[i].agentId]) outputsLog[agents[i].agentId] = [];
      outputsLog[agents[i].agentId].push(out.sequenceStr);
      if (out.iLocked) lockGenerations.add(agents[i].agentId);
    }
  }

  const lockRate = lockGenerations.size / nAgents;
  const loopScores = [];
  for (const aid of Object.keys(outputsLog).sort()) {
    const recent = outputsLog[aid].slice(-100);
    const score = recent.reduce((s, seq) => s + strangeLoopScore({ _lastOutput: seq.split(' ') }), 0) / Math.max(recent.length, 1);
    loopScores.push(score);
  }
  const meanLoop = loopScores.reduce((a, b) => a + b, 0) / loopScores.length;

  let nonTrivial = 0, total = 0;
  for (const seqs of Object.values(outputsLog)) {
    for (const s of seqs.slice(-100)) { total++; if (s !== 'wait') nonTrivial++; }
  }
  const outputSync = nonTrivial / Math.max(total, 1);

  console.log(`    Agents:          ${nAgents}`);
  console.log(`    Lock threshold:  ${lockThreshold}`);
  console.log(`    Lock rate:       ${(lockRate * 100).toFixed(0)}%`);
  console.log(`    Mean loop depth: ${meanLoop.toFixed(3)}`);
  console.log(`    Non-trivial out: ${(outputSync * 100).toFixed(1)}%`);

  return { lockRate, meanLoopDepth: meanLoop, outputSyncRatio: outputSync, locked: lockGenerations.size, unlocked: nAgents - lockGenerations.size };
}

function main() {
  const t0 = Date.now();
  console.log('='.repeat(62));
  console.log('Self-Reference ON/OFF Contrast');
  console.log('='.repeat(62));
  console.log('\nThe same world, same architecture — one parameter changes everything.\n');

  const on = runCondition('Self-Reference ON  (threshold=0.25)', 0.25);
  const off = runCondition('Self-Reference OFF (threshold=1.5)', 1.5);

  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
  console.log('='.repeat(62));
  console.log('Result Summary');
  console.log('='.repeat(62));
  console.log(`  Lock rate:        ON ${(on.lockRate*100).toFixed(0)}%  OFF ${(off.lockRate*100).toFixed(0)}%`);
  console.log(`  Mean loop depth:  ON ${on.meanLoopDepth.toFixed(3)}  OFF ${off.meanLoopDepth.toFixed(3)}`);
  console.log(`  Non-trivial out:  ON ${(on.outputSyncRatio*100).toFixed(0)}%  OFF ${(off.outputSyncRatio*100).toFixed(0)}%`);

  if (on.lockRate === 1 && off.lockRate === 0) {
    console.log('\n  ✓ Absolute contrast — self-reference is causally necessary');
  } else {
    console.log(`\n  ~ Partial contrast (${(on.lockRate*100).toFixed(0)}% vs ${(off.lockRate*100).toFixed(0)}%)`);
  }
}

main();
