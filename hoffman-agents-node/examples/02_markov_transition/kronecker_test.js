/**
 * Kronecker Product Distance Test
 *
 * Measures whether the combination operator ⊗ produces dynamics closer to
 * a tensor product (P₁ ⊗ P₂) or a direct sum (P₁ ⊕ P₂).
 *
 * Run separately from the main experiment — computationally expensive.
 *
 * Usage: node examples/02_markov_transition/kronecker_test.js
 */
const { ConsciousAgent, WorldState, combine, fuse } = require('../../src/index');
const { MetaTrie } = require('../../src/core/meta-trie');

const MAX_KRONECKER = 10000;

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
  return { P, stateIds, idx, n };
}

function frobeniusNorm(A, B) {
  const n = A.length, m = A[0].length;
  let sum = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) sum += (A[i][j] - B[i][j]) ** 2;
  return Math.sqrt(sum);
}

function buildDirectSum(P1, P2) {
  const n = P1.length, m = P2.length;
  const D = Array.from({ length: n + m }, () => new Float64Array(n + m));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i][j] = P1[i][j];
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) D[n + i][n + j] = P2[i][j];
  return D;
}

function buildKroneckerProduct(P1, P2) {
  const n = P1.length, m = P2.length;
  const K = Array.from({ length: n * m }, () => new Float64Array(n * m));
  for (let i1 = 0; i1 < n; i1++) {
    for (let j1 = 0; j1 < n; j1++) {
      for (let i2 = 0; i2 < m; i2++) {
        for (let j2 = 0; j2 < m; j2++) {
          K[i1 * m + i2][j1 * m + j2] = P1[i1][j1] * P2[i2][j2];
        }
      }
    }
  }
  return K;
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

function run() {
  console.log('='.repeat(66));
  console.log('Kronecker Product Distance Test');
  console.log('='.repeat(66));

  const nBase = 6, connectivity = Math.min(nBase, 10);
  const agents = {};

  // Phase 1: train isolated agents
  for (let i = 0; i < nBase; i++) {
    const aid = `CA_${String(i).padStart(3, '0')}`;
    agents[aid] = new ConsciousAgent({ agentId: aid });
    for (let t = 0; t < 400; t++) agents[aid].step(new WorldState({ world: [`s${i}_${t}`] }));
  }

  // Phase 2: interact
  const interact = (rounds) => {
    for (let rnd = 0; rnd < rounds; rnd++) {
      const outputs = {};
      for (const [aid, ag] of Object.entries(agents)) outputs[aid] = ag.getOutput();
      const graph = connectivity < Object.keys(agents).length ? buildInteractionGraph(Object.keys(agents), connectivity) : null;
      for (const [aid, ag] of Object.entries(agents)) {
        const targets = graph ? graph[aid] : Object.keys(agents);
        for (const oa of targets) {
          if (oa !== aid) ag.step(new WorldState({ [oa]: outputs[oa] }));
        }
      }
    }
  };

  interact(60);

  // Phase 3: combine ripe agents, measure Kronecker distance
  const results = [];
  for (let rnd = 0; rnd < 10; rnd++) {
    const outputs = {};
    for (const [aid, ag] of Object.entries(agents)) outputs[aid] = ag.getOutput();
    const graph = connectivity < Object.keys(agents).length ? buildInteractionGraph(Object.keys(agents), connectivity) : null;
    for (const [aid, ag] of Object.entries(agents)) {
      const targets = graph ? graph[aid] : Object.keys(agents);
      for (const oa of targets) {
        if (oa !== aid) ag.step(new WorldState({ [oa]: outputs[oa] }));
      }
    }

    const locked = Object.entries(agents).filter(([, a]) => a.experience.selfToken.locked).length;
    const uncombined = Object.entries(agents).filter(([, a]) => a.experience.selfToken.locked && !a._combined).length;
    if (rnd === 0) console.log(`  Round ${rnd}: ${locked} locked, ${uncombined} uncombined, ${Object.keys(agents).length} total agents`);

    const ripe = Object.entries(agents).filter(([, a]) => a.experience.selfToken.locked && !a._combined).map(([id]) => id);
    if (ripe.length < 2) { console.log('  Round '+rnd+': skipping - only '+ripe.length+' ripe'); continue; }

    const scored = ripe.sort((a, b) => agents[a].experience.traceBuffer.predictionErrorMean(5) - agents[b].experience.traceBuffer.predictionErrorMean(5));

    for (let i = 0; i < scored.length - 1; i += 2) {
      const aId = scored[i], bId = scored[i + 1];
      const a = agents[aId], b = agents[bId];
      console.log('  Combining', aId, bId);

      // Extract parent matrices BEFORE combining
      const r1 = extractMetaMatrix(a);
      const r2 = extractMetaMatrix(b);
      if (!r1 || !r2) continue;
      if (r1.P.length < 2 || r2.P.length < 2) continue;

      // Skip Kronecker computation if product space is too large
      if (r1.n * r2.n > MAX_KRONECKER) {
        console.log(`  Skipping Kronecker for ${aId}x${bId}: ${r1.n}x${r2.n} = ${r1.n*r2.n} > ${MAX_KRONECKER}`);
      }

      // Combine
      const c = combine(a, b);
      c.agentId = `L${c.cycleLevel}_${aId.slice(-3)}_${bId.slice(-3)}`;
      agents[c.agentId] = c;
      agents[aId]._combined = true;
      agents[bId]._combined = true;

      // Extract child matrix
      const rc = extractMetaMatrix(c);
      if (!rc || rc.P.length < 2) continue;

      const P1 = r1.P, P2 = r2.P, Pc = rc.P;

      if (r1.n * r2.n > MAX_KRONECKER) { console.log(`  Computing ${aId}x${bId}: ${r1.n}x${r2.n} = ${r1.n*r2.n}`); continue; }

      console.log(`  Computing Kronecker for ${aId}x${bId}: ${r1.n}x${r2.n} = ${r1.n*r2.n}, Pc=${rc.n}`);

      // Build Kronecker product P1 ⊗ P2
      const K = buildKroneckerProduct(P1, P2);

      // Build direct sum P1 ⊕ P2
      const D = buildDirectSum(P1, P2);

      // Compute distances — embed Pc into the larger space by zero-padding
      const maxN = Math.max(Pc.length, K.length, D.length);
      const padPc = Array.from({ length: maxN }, () => new Float64Array(maxN));
      const padK = Array.from({ length: maxN }, () => new Float64Array(maxN));
      const padD = Array.from({ length: maxN }, () => new Float64Array(maxN));
      for (let ii = 0; ii < Pc.length; ii++) for (let jj = 0; jj < Pc.length; jj++) padPc[ii][jj] = Pc[ii][jj];
      for (let ii = 0; ii < K.length; ii++) for (let jj = 0; jj < K.length; jj++) padK[ii][jj] = K[ii][jj];
      for (let ii = 0; ii < D.length; ii++) for (let jj = 0; jj < D.length; jj++) padD[ii][jj] = D[ii][jj];

      const dK = frobeniusNorm(padPc, padK);
      const dD = frobeniusNorm(padPc, padD);
      const ratio = dD > 0 ? dK / dD : Infinity;

      results.push({
        child: c.agentId,
        n1: P1.length, n2: P2.length, nc: Pc.length,
        nK: K.length, nD: D.length,
        distKronecker: dK, distDirectSum: dD,
        ratio,
        closer: ratio < 1 ? 'PRODUCT' : 'SUM',
      });
    }
  }

  // Print results
  console.log(`\n  ${'─'.repeat(66)}`);
  console.log(`  Kronecker Distance Results`);
  console.log(`  ${'─'.repeat(66)}`);
  console.log(`  ${'Child'.padEnd(16)} ${'n₁'.padEnd(5)} ${'n₂'.padEnd(5)} ${'n_c'.padEnd(5)} ${'n_K'.padEnd(5)} ${'|Pc-P₁⊗P₂|'.padEnd(15)} ${'|Pc-P₁⊕P₂|'.padEnd(15)} ${'Ratio'.padEnd(8)} Closer`);
  let sumRatio = 0, count = 0;
  for (const r of results) {
    console.log(`  ${r.child.padEnd(16)} ${String(r.n1).padEnd(5)} ${String(r.n2).padEnd(5)} ${String(r.nc).padEnd(5)} ${String(r.nK).padEnd(5)} ${r.distKronecker.toFixed(4).padEnd(15)} ${r.distDirectSum.toFixed(4).padEnd(15)} ${r.ratio.toFixed(4).padEnd(8)} ${r.closer}`);
    sumRatio += r.ratio;
    count++;
  }
  if (count > 0) {
    const avg = sumRatio / count;
    console.log(`\n  Average ratio: ${avg.toFixed(4)} (${avg < 1 ? 'closer to PRODUCT' : 'closer to SUM'})`);
    console.log(`  Ratio < 1 = Pc more product-like; Ratio > 1 = Pc more sum-like`);
  }
}

run();
