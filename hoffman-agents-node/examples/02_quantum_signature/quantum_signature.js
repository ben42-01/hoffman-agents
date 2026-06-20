const { ConsciousAgent, WorldState, combine } = require('../../src/index');

function extractMetaMatrix(agent) {
  const mt = agent.experience.metaTrie;
  if (mt.registrySize < 2) return null;
  const allIds = [...mt._registry.keys()].sort((a, b) => a - b);
  const active = new Set();
  for (const sid of allIds) {
    const node = mt.trie.lookup([sid]);
    if (node && Object.keys(node.children).length > 0) active.add(sid);
  }
  if (mt.lastMetaState !== null) active.add(mt.lastMetaState);
  if (active.size < 2) return null;
  const stateIds = [...active].sort((a, b) => a - b);
  const idx = new Map(stateIds.map((id, i) => [id, i]));
  const n = stateIds.length;
  const P = Array.from({ length: n }, () => new Float64Array(n));
  for (const stateId of stateIds) {
    const node = mt.trie.lookup([stateId]);
    if (node && Object.keys(node.children).length > 0) {
      let total = 0;
      for (const child of Object.values(node.children)) total += child.visitCount;
      if (total > 0) {
        for (const [cs, cn] of Object.entries(node.children)) {
          const ci = idx.get(parseInt(cs));
          if (ci !== undefined) P[idx.get(stateId)][ci] = cn.visitCount / total;
        }
      }
    }
  }
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += P[i][j]; if (s === 0) P[i][i] = 1; }
  return P;
}

function spectralGap(P) {
  const n = P.length;
  if (n < 2) return 1;
  let sumMax = 0;
  for (let i = 0; i < n; i++) { let m = 0; for (let j = 0; j < n; j++) if (P[i][j] > m) m = P[i][j]; sumMax += m; }
  const avgMax = sumMax / n, uniform = 1 / n;
  return avgMax === uniform ? 0 : (avgMax - uniform) / (1 - uniform);
}

function detailedBalanceError(P) {
  const n = P.length;
  let pi = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 200; iter++) {
    const pn = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) pn[j] += pi[i] * P[i][j];
    pi = pn;
  }
  const errs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (pi[i] > 0 && pi[j] > 0) {
      const l = pi[i] * P[i][j], r = pi[j] * P[j][i];
      if (Math.abs(l + r) > 1e-12) errs.push(Math.abs(l - r) / (l + r));
    }
  }
  return errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
}

function analyze(agents, label) {
  const byLevel = {};
  console.log(`\n  ${'─'.repeat(50)}`);
  console.log(`  ${label}`);
  console.log(`  ${'─'.repeat(50)}`);
  console.log(`  ${'Agent'.padEnd(22)} ${'Lvl'.padEnd(4)} ${'States'.padEnd(7)} ${'Gap'.padEnd(10)} ${'DB Err'.padEnd(10)}`);
  for (const [aid, agent] of Object.entries(agents).sort()) {
    const P = extractMetaMatrix(agent);
    let gap = null, dbe = null, gs = 'N/A', ds = 'N/A';
    if (P) { gap = spectralGap(P); dbe = detailedBalanceError(P); gs = gap.toFixed(4); ds = dbe.toFixed(4); }
    const lvl = agent.cycleLevel;
    console.log(`  ${aid.padEnd(22)} ${String(lvl).padEnd(4)} ${String(agent.experience.metaTrie.registrySize).padEnd(7)} ${gs.padEnd(10)} ${ds.padEnd(10)}`);
    if (gap !== null) { if (!byLevel[lvl]) byLevel[lvl] = []; byLevel[lvl].push({ gap, dbe }); }
  }
  return byLevel;
}

function run() {
  const nBase = 6, nRounds = 200;
  const t0 = Date.now();
  console.log('='.repeat(66));
  console.log('Quantum Signature — Tree-of-Life Spectral Analysis');
  console.log('='.repeat(66));
  console.log(`\n${nBase} base agents, ${nRounds} rounds...`);

  // Phase 1: isolated agents
  const agents = {};
  for (let i = 0; i < nBase; i++) {
    const aid = `CA_${String(i).padStart(3, '0')}`;
    agents[aid] = new ConsciousAgent({ agentId: aid });
    for (let t = 0; t < 30; t++) agents[aid].step(new WorldState({ world: [`s${i}_${t}`] }));
  }
  analyze(agents, 'Phase 1: Isolated agents');

  // Phase 2 + 3: interaction then combination
  let snapTaken = false;
  for (let rnd = 0; rnd < nRounds; rnd++) {
    const outputs = {};
    for (const [aid, ag] of Object.entries(agents)) outputs[aid] = ag.getOutput();
    for (const [aid, ag] of Object.entries(agents)) {
      for (const [oa, o] of Object.entries(outputs)) if (oa !== aid) ag.step(new WorldState({ [oa]: o }));
    }
    if (rnd === 19 && !snapTaken) { analyze(agents, 'Phase 2: Interacting, pre-combination'); snapTaken = true; }
    if (rnd > 0 && rnd % 20 === 0) {
      const ripe = Object.entries(agents).filter(([, a]) => a.experience.selfToken.locked && !a._combined).map(([id]) => id);
      if (ripe.length >= 2) {
        const scored = ripe.sort((a, b) => agents[a].experience.traceBuffer.predictionErrorMean(5) - agents[b].experience.traceBuffer.predictionErrorMean(5));
        for (let i = 0; i < scored.length - 1; i += 2) {
          const c = combine(agents[scored[i]], agents[scored[i + 1]]);
          c.agentId = `L${c.cycleLevel}_${scored[i].slice(-3)}_${scored[i + 1].slice(-3)}`;
          agents[c.agentId] = c;
          agents[scored[i]]._combined = true;
          agents[scored[i + 1]]._combined = true;
        }
      }
    }
  }

  const post = analyze(agents, 'Phase 3: Post-combination');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

  // Cross-level summary
  console.log(`${'─'.repeat(66)}`);
  console.log('Cross-Level Quantum Signature Summary');
  console.log(`${'─'.repeat(66)}`);
  let pg = null;
  for (const lvl of Object.keys(post).sort((a, b) => a - b)) {
    const items = post[lvl];
    const gs = items.map(x => x.gap), ds = items.map(x => x.dbe);
    const mg = gs.reduce((a, b) => a + b, 0) / gs.length, md = ds.reduce((a, b) => a + b, 0) / ds.length;
    let tag = mg < 0.3 && md > 0.1 ? '  ← QUANTUM-LIKE' : mg > 0.85 && md < 0.15 ? '  ← CLASSICAL' : '';
    let ch = pg !== null ? (mg > pg + 0.05 ? ' ↑ recovery' : mg < pg - 0.05 ? ' ↓ collapse' : '') : '';
    console.log(`  ${(lvl === '0' ? 'Base' : `Level ${lvl}`).padEnd(8)} (${items.length} agents)  gap=${mg.toFixed(4)}  db_err=${md.toFixed(4)}${tag}${ch}`);
    pg = mg;
  }
  console.log(`\n  gap~1.0, db_err~0.0 = classical | gap~0.0, db_err>0.1 = quantum-like`);
}

run();
