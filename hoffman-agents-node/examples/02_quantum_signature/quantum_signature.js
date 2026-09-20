/**
 * Quantum Signature? — Spectral Analysis of Combination, with Classical Controls
 *
 * 2.x version of this experiment reported a "quantum-like" collapse of the
 * spectral gap at combination levels 1-2. That signal was an artifact:
 *   - combined agents carry their constituents' chains as disconnected pieces;
 *     a reducible chain has |λ₂| = 1, so its gap is 0 by definition
 *   - unobserved rows were turned into absorbing states
 *   - agents only combined because the 2.x "I" lock fired in any world
 * and the criterion itself (small gap + detailed-balance violation) is met by
 * ordinary classical chains, e.g. a clock.
 *
 * This version asks answerable questions:
 *   1. Classical controls: what do gap and irreversibility look like for
 *      chains that are classical by construction?
 *   2. Agents: gap (1 − |λ₂|), period and irreversibility of each agent's own
 *      recurrent meta-state chain (MetaTrie.ergodicDiagnostics).
 *   3. Tensor-product prediction: for independent agents A, B the product
 *      kernel M_A ⊗ M_B has |λ₂| = max(|λ₂(A)|, |λ₂(B)|). Does the combined
 *      agent's learned chain match that prediction?
 *   4. Fusion: does fuse() restore the constituents' chains exactly?
 *
 * Every quantity here describes a classical Markov chain. Gap measures mixing
 * speed and irreversibility measures net probability circulation; neither is
 * evidence of quantum behaviour. See ../09_double_slit_analogy for what that
 * would require.
 */
const {
  ConsciousAgent, WorldState, combine, fuse, productKernel, metaKernel, MarkovKernel, markov,
} = require('../../src/index');

const N_BASE = 8;
const ISOLATED_STEPS = 400;
const ROUNDS = 200;
const COMBINE_EVERY = 20;

/* ────────────── measurements ────────────── */

function analyseKernel(P) {
  const { pi } = markov.stationary(P);
  const closed = markov.closedClasses(P).length;
  const lambda2 = closed > 1 ? 1 : markov.secondEigenvalueModulus(P, pi);
  return { n: P.length, closed, period: closed > 1 ? null : markov.period(P, 0), gap: 1 - lambda2, irrev: markov.irreversibility(P, pi) };
}

// The agent's own recurrent meta-chain. `evidence` uses the same rule as the
// "I" lock: at least 20 transitions and 2 per state, otherwise the estimate is noise.
function analyseAgent(agent) {
  const K = metaKernel(agent);
  if (!K) return null;
  const d = agent.experience.metaTrie.ergodicDiagnostics();
  const r = analyseKernel(K.matrix);
  return { ...r, transitions: d.nTransitions, evidence: d.nTransitions >= Math.max(20, 2 * r.n) };
}

const countsOf = (agent) => JSON.stringify(agent.experience.metaTrie.transitionCounts().counts.map(r => Array.from(r)));

// What 2.x effectively measured: all transitions, inherited pieces included.
function closedClassesIncludingInherited(agent) {
  const { counts } = agent.experience.metaTrie.transitionCounts({ includeInherited: true });
  const kept = markov.pruneUnobservedRows(counts);
  return kept.length ? markov.closedClasses(markov.subMatrix(counts, kept)).length : 0;
}

const f = (x, d = 3) => (x === null || x === undefined ? '  -  ' : x.toFixed(d));

/* ────────────── 1. classical controls ────────────── */

function controls() {
  const cycle = (n, forward, stay) => Array.from({ length: n }, (_, i) => {
    const r = new Array(n).fill(0); r[i] += stay; r[(i + 1) % n] += forward; r[(i + n - 1) % n] += 1 - forward - stay; return r;
  });
  const uniform = (n) => Array.from({ length: n }, () => new Array(n).fill(1 / n));
  const coin = [[0.5, 0.5], [0.5, 0.5]];
  const twoPieces = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]];
  const K = new MarkovKernel({ states: ['a', 'b'], matrix: [[0.9, 0.1], [0.3, 0.7]] });
  return [
    ['clock: 10-cycle, forward 0.95', cycle(10, 0.95, 0.05)],
    ['lazy symmetric walk on 10-cycle', cycle(10, 0.25, 0.5)],
    ['i.i.d. uniform over 10 states', uniform(10)],
    ['two disconnected coin chains', twoPieces],
    ['independent product K ⊗ K', K.tensor(K).matrix],
    ['single coin chain', coin],
  ].map(([name, P]) => [name, analyseKernel(P)]);
}

/* ────────────── 2-4. agents ────────────── */

function printAgents(title, agents) {
  console.log(`\n  ${title}`);
  console.log(`  ${'agent'.padEnd(16)} lvl  locked  class  trans  period  gap     irrev   closed incl. inherited`);
  for (const a of agents) {
    const r = analyseAgent(a);
    const note = !r ? '' : !r.evidence ? '  (too little data)' : r.period > 1 ? '  (periodic: gap 0 by definition)' : '';
    console.log(`  ${a.agentId.padEnd(16)} ${String(a.cycleLevel).padEnd(4)} ${String(a.isILocked).padEnd(7)} ${String(r ? r.n : 0).padEnd(6)} ${String(r ? r.transitions : 0).padEnd(6)} ${String(r && r.period !== null ? r.period : '-').padEnd(7)} ${f(r && r.gap)}   ${f(r && r.irrev)}   ${closedClassesIncludingInherited(a)}${note}`);
  }
}

function run() {
  const t0 = Date.now();
  console.log('='.repeat(78));
  console.log('Quantum Signature? — spectral analysis of combination, with classical controls');
  console.log('='.repeat(78));

  console.log('\n  1. Classical controls (classical by construction)');
  console.log(`  ${'chain'.padEnd(34)} states  closed  gap     irrev`);
  for (const [name, r] of controls()) {
    console.log(`  ${name.padEnd(34)} ${String(r.n).padEnd(7)} ${String(r.closed).padEnd(7)} ${f(r.gap)}   ${f(r.irrev)}`);
  }
  console.log('  → small gap + irreversibility (the 2.x "quantum" criterion) is a plain clock;');
  console.log('    gap 0 is what any chain made of disconnected pieces gives.');

  const agents = new Map();
  for (let i = 0; i < N_BASE; i++) {
    const id = `CA_${String(i).padStart(3, '0')}`;
    const agent = new ConsciousAgent({ agentId: id, seed: i + 1 });
    for (let t = 0; t < ISOLATED_STEPS; t++) agent.step(new WorldState({ world: [`s${i}_${t}`] }));
    agents.set(id, agent);
  }
  printAgents(`2a. Isolated agents (${ISOLATED_STEPS} steps each)`, [...agents.values()]);

  const combinations = [];
  const available = new Set(agents.keys());
  for (let rnd = 1; rnd <= ROUNDS; rnd++) {
    const outputs = new Map([...agents].map(([id, a]) => [id, a.getOutput()]));
    for (const [id, agent] of agents) {
      for (const [other, seq] of outputs) if (other !== id) agent.step(new WorldState({ [other]: seq }));
    }
    // Combination on a fixed schedule. (Gating on the "I" lock, as 2.x did, never
    // fires here in v3: this world has no dominant experiential attractor.)
    if (rnd % COMBINE_EVERY === 0 && available.size >= 2) {
      const ready = [...available]
        .filter(id => metaKernel(agents.get(id)))
        .sort((a, b) => agents.get(a).experience.traceBuffer.predictionErrorMean(5) - agents.get(b).experience.traceBuffer.predictionErrorMean(5) || (a < b ? -1 : 1));
      for (let i = 0; i + 1 < ready.length; i += 2) {
        const [x, y] = [agents.get(ready[i]), agents.get(ready[i + 1])];
        const prior = productKernel(x, y);
        const c = combine(x, y);
        c.agentId = `L${c.cycleLevel}_${ready[i].slice(-3)}_${ready[i + 1].slice(-3)}`;
        agents.set(c.agentId, c);
        available.delete(ready[i]); available.delete(ready[i + 1]); available.add(c.agentId);
        combinations.push({ agent: c, snapshots: new Map([[x.agentId, countsOf(x)], [y.agentId, countsOf(y)]]), round: rnd, prior: prior ? analyseKernel(prior.matrix) : null });
      }
    }
  }
  printAgents(`2b. After ${ROUNDS} interaction rounds (combined every ${COMBINE_EVERY} rounds)`, [...agents.values()]);
  console.log('  "closed incl. inherited" > 1 means a naive analysis of that trie is reducible → gap 0 (the 2.x artifact).');

  console.log('\n  3. Tensor-product prediction vs learned joint dynamics');
  console.log(`  ${'combined'.padEnd(16)} round  prior gap  learned gap  |Δ|     prior irrev  learned irrev`);
  const deltas = [];
  for (const { agent, round, prior } of combinations) {
    const learned = analyseAgent(agent);
    const delta = prior && learned ? Math.abs(prior.gap - learned.gap) : null;
    if (delta !== null) deltas.push(delta);
    console.log(`  ${agent.agentId.padEnd(16)} ${String(round).padEnd(6)} ${f(prior && prior.gap).padEnd(10)} ${f(learned && learned.gap).padEnd(12)} ${f(delta).padEnd(7)} ${f(prior && prior.irrev).padEnd(12)} ${f(learned && learned.irrev)}`);
  }
  if (deltas.length) {
    console.log(`  mean |Δgap| = ${f(deltas.reduce((a, b) => a + b, 0) / deltas.length)}`);
  }
  console.log('  The prior is exactly the product of the constituents\' kernels (gap 0 if either is periodic).');
  console.log('  The learned chain is what the combined agent experiences afterwards in a shared world;');
  console.log('  nothing forces the two to agree, and here they do not.');

  console.log('\n  4. Fusion restores each constituent\'s chain as it was at combination time');
  for (const { agent, snapshots } of combinations) {
    const parts = fuse(agent);
    const ok = parts.every(p => snapshots.get(p.agentId) === countsOf(p));
    console.log(`  fuse(${agent.agentId.padEnd(16)}) → ${parts.map(p => p.agentId).join(' + ').padEnd(28)} exact: ${ok ? 'yes' : 'NO'}`);
  }

  console.log('\n' + '─'.repeat(78));
  console.log('Summary by combination level (agents\' own recurrent chains)');
  console.log('─'.repeat(78));
  const byLevel = new Map();
  const excluded = [];
  for (const a of agents.values()) {
    const r = analyseAgent(a);
    if (!r) continue;
    if (!r.evidence || r.period > 1) { excluded.push(`${a.agentId} (${!r.evidence ? 'too little data' : `period ${r.period}`})`); continue; }
    if (!byLevel.has(a.cycleLevel)) byLevel.set(a.cycleLevel, []);
    byLevel.get(a.cycleLevel).push(r);
  }
  for (const [lvl, rs] of [...byLevel].sort((a, b) => a[0] - b[0])) {
    const mean = (k) => rs.reduce((s, r) => s + r[k], 0) / rs.length;
    console.log(`  level ${lvl}  (${rs.length} agent${rs.length === 1 ? '' : 's'})  gap=${f(mean('gap'))}  irreversibility=${f(mean('irrev'))}`);
  }
  if (excluded.length) console.log(`  excluded (gap not meaningful): ${excluded.join(', ')}`);
  console.log('\n  All of these are classical Markov chains: gap = mixing speed, irreversibility = net circulation.');
  console.log(`\n  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}

run();
