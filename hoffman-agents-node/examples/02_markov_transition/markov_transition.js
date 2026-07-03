const { ConsciousAgent, WorldState, combine, fuse } = require('../../src/index');

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

  let pi = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 1000; iter++) {
    const piNew = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) piNew[j] += pi[i] * P[i][j];
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(piNew[i] - pi[i]);
    pi = piNew;
    if (diff < 1e-12) break;
  }

  const B = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i][j] = P[i][j] - pi[j];

  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = 1 / Math.sqrt(n);
  for (let iter = 0; iter < 1000; iter++) {
    const vNew = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) vNew[i] += B[i][j] * v[j];
    let norm = 0;
    for (let i = 0; i < n; i++) norm += vNew[i] * vNew[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-15) return 1;
    for (let i = 0; i < n; i++) vNew[i] /= norm;
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(vNew[i] - v[i]);
    v = vNew;
    if (diff < 1e-10) break;
  }

  let Bv = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Bv[i] += B[i][j] * v[j];
  let lambda2 = 0;
  for (let i = 0; i < n; i++) lambda2 += v[i] * Bv[i];

  return 1 - Math.abs(lambda2);
}

function mixingTime(gap) {
  if (gap <= 0 || gap >= 1) return gap <= 0 ? Infinity : 0;
  const t = -1 / Math.log(1 - gap);
  return t > 1e6 ? Infinity : t;
}

function entropyProductionRate(P) {
  const n = P.length;
  let pi = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 500; iter++) {
    const pn = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) pn[j] += pi[i] * P[i][j];
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(pn[i] - pi[i]);
    pi = pn;
    if (diff < 1e-12) break;
  }

  let epr = 0;
  for (let i = 0; i < n; i++) {
    if (pi[i] <= 0) continue;
    for (let j = 0; j < n; j++) {
      if (P[i][j] <= 0 || P[j][i] <= 0) continue;
      epr += pi[i] * P[i][j] * Math.log(P[i][j] / P[j][i]);
    }
  }
  return epr;
}

function analyze(agents, label) {
  const byLevel = {};
  console.log(`\n  ${'─'.repeat(60)}`);
  console.log(`  ${label}`);
  console.log(`  ${'─'.repeat(60)}`);
  console.log(`  ${'Agent'.padEnd(22)} ${'Lvl'.padEnd(4)} ${'States'.padEnd(7)} ${'Gap'.padEnd(10)} ${'MixTime'.padEnd(10)} ${'EPR'.padEnd(10)}`);
  for (const [aid, agent] of Object.entries(agents).sort()) {
    const P = extractMetaMatrix(agent);
    let gap = null, epr = null, mt = null, gs = 'N/A', es = 'N/A', ms = 'N/A';
    if (P) {
      gap = spectralGap(P);
      epr = entropyProductionRate(P);
      mt = mixingTime(gap);
      gs = gap.toFixed(4);
      es = epr.toFixed(4);
      ms = Number.isFinite(mt) ? mt.toFixed(1) : '∞';
    }
    const lvl = agent.cycleLevel;
    console.log(`  ${aid.padEnd(22)} ${String(lvl).padEnd(4)} ${String(agent.experience.metaTrie.registrySize).padEnd(7)} ${gs.padEnd(10)} ${ms.padEnd(10)} ${es.padEnd(10)}`);
    if (gap !== null) { if (!byLevel[lvl]) byLevel[lvl] = []; byLevel[lvl].push({ gap, epr, mt: Number.isFinite(mt) ? mt : NaN }); }
  }
  return byLevel;
}

function buildInteractionGraph(agentIds, connectivity) {
  const n = agentIds.length;
  if (connectivity >= n) return null;
  const graph = {};
  for (const id of agentIds) {
    const others = agentIds.filter(x => x !== id);
    const shuffled = [...others].sort(() => Math.random() - 0.5);
    graph[id] = shuffled.slice(0, connectivity);
  }
  return graph;
}

function run(connectivityOverride) {
  const nBase = 8, nRounds = 400;
  const connectivity = connectivityOverride !== undefined ? connectivityOverride : nBase;
  const t0 = Date.now();
  console.log('='.repeat(66));
  console.log('Markov Structural Transition — Tree-of-Life Analysis');
  console.log('='.repeat(66));
  console.log(`\n${nBase} base agents, ${nRounds} rounds, connectivity=${connectivity === nBase ? 'all' : connectivity}...`);

  const agents = {};
  for (let i = 0; i < nBase; i++) {
    const aid = `CA_${String(i).padStart(3, '0')}`;
    agents[aid] = new ConsciousAgent({ agentId: aid });
    for (let t = 0; t < 400; t++) agents[aid].step(new WorldState({ world: [`s${i}_${t}`] }));
  }
  analyze(agents, 'Phase 1: Isolated agents');

  let snapTaken = false;
  for (let rnd = 0; rnd < nRounds; rnd++) {
    const outputs = {};
    for (const [aid, ag] of Object.entries(agents)) outputs[aid] = ag.getOutput();
    const graph = connectivity < nBase ? buildInteractionGraph(Object.keys(agents), connectivity) : null;
    for (const [aid, ag] of Object.entries(agents)) {
      const targets = graph ? graph[aid] : Object.keys(agents);
      for (const oa of targets) {
        if (oa !== aid) ag.step(new WorldState({ [oa]: outputs[oa] }));
      }
    }
    if (rnd === 39 && !snapTaken) { analyze(agents, 'Phase 2: Interacting (40 rounds)'); snapTaken = true; }
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

  const topAgents = Object.entries(agents).filter(([aid]) => aid.startsWith('L')).sort();
  if (topAgents.length > 0) {
    const highest = topAgents[topAgents.length - 1][1];
    const fused = fuse(highest);
    const fusedMap = {};
    for (const f of fused) { fusedMap[f.agentId] = f; }
    analyze(fusedMap, `Phase 4: Fusion of ${highest.agentId}`);
  }

  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

  console.log(`${'─'.repeat(66)}`);
  console.log('Cross-Level Summary');
  console.log(`${'─'.repeat(66)}`);
  let pg = null;
  for (const lvl of Object.keys(post).sort((a, b) => a - b)) {
    const items = post[lvl];
    const gs = items.map(x => x.gap), eps = items.map(x => x.epr);
    const mg = gs.reduce((a, b) => a + b, 0) / gs.length, me = eps.reduce((a, b) => a + b, 0) / eps.length;
    const tag = mg < 0.05 ? '  ← SLOW MIXING' : mg > 0.8 ? '  ← FAST MIXING' : '';
    let ch = pg !== null ? (mg > pg + 0.05 ? ' ↑ faster' : mg < pg - 0.05 ? ' ↓ slower' : '') : '';
    console.log(`  ${(lvl === '0' ? 'Base' : `Level ${lvl}`).padEnd(8)} (${items.length} agents)  gap=${mg.toFixed(4)}  epr=${me.toFixed(4)}${tag}${ch}`);
    pg = mg;
  }
  console.log(`\n  gap ~ 1.0 = fast mixing (near-uniform transitions)`);
  console.log(`  gap ~ 0.0 = slow mixing (near-reducible / cyclic structure)`);
  console.log(`  EPR > 0   = irreversible dynamics (directed flow)`);
}

run();
