const crypto = require('crypto');
const { ExperienceSpace } = require('./experience-space');
const { perceive } = require('./perceptual-map');
const { decide, buildDecisionKernel, CORE_TOKENS, DEFAULT_LEXICON_ROW } = require('./decision-map');
const { computeSelfReferenceScore } = require('../core/strange-loop');
const { mulberry32 } = require('../math/rng');
const markov = require('../math/markov');
const { MarkovKernel, StochasticMatrix } = require('../kernels/markov-kernel');

class StepOutput {
  constructor({ step, generation, state, stateLabel, predictionError, sequence, sequenceStr, loopDepth, iLocked, iStability, interrupt, actionDistribution }) {
    this.step = step;
    this.generation = generation;
    this.state = state;
    this.stateLabel = stateLabel;
    this.predictionError = predictionError;
    this.sequence = sequence;
    this.sequenceStr = sequenceStr;
    this.loopDepth = loopDepth;
    this.iLocked = iLocked;
    this.iStability = iStability;
    this.interrupt = interrupt || null;
    this.actionDistribution = actionDistribution || {};
  }
}

class Prediction {
  constructor({ stateId, stateLabel, confidence, topK } = {}) {
    this.stateId = stateId;
    this.stateLabel = stateLabel;
    this.confidence = confidence;
    this._topK = topK;
  }

  topK(n) {
    if (!this._topK) return [];
    return this._topK.slice(0, Math.max(0, n));
  }
}

const MODES = { learning: 'learning', frozen: 'frozen', debug: 'debug' };

class ConsciousAgent {
  constructor({
    agentId = null,
    experience = new ExperienceSpace(),
    world = null,
    generation = 0,
    stepCount = 0,
    metaObservationInterval = 20,
    constituentIds = new Set(),
    leafConstituentIds = new Set(),
    cycleLevel = 0,
    expressionTemp = 1,
    pStable = 0.8,
    pLexicon = 0.1,
    pExplore = 0.05,
    mode = 'learning',
    rng = null,
    seed = null,
    allowableTokens = null,
    mathVersion = null,
    lexiconRow = DEFAULT_LEXICON_ROW,
  } = {}) {
    this.agentId = agentId || `CA_${crypto.randomBytes(4).toString('hex')}`;
    this.experience = experience;
    this.world = world;
    this.generation = generation;
    this.stepCount = stepCount;
    this.metaObservationInterval = metaObservationInterval;
    this.constituentIds = constituentIds;
    this.leafConstituentIds = leafConstituentIds;
    this.cycleLevel = cycleLevel;
    this.expressionTemp = expressionTemp;
    this.pStable = pStable;
    this.pLexicon = pLexicon;
    this.pExplore = pExplore;
    this._ergodicState = 'idle';
    this._lastOutput = ['wait'];
    this._combined = false;
    this._mode = MODES[mode] || 'learning';
    this.seed = seed;
    this._rng = rng || (seed !== null && seed !== undefined ? mulberry32(seed) : Math.random.bind(Math));
    this._allowableTokens = allowableTokens ? new Set(allowableTokens) : null;
    this.lexiconRow = [...lexiconRow];
    // Kept for fuse(): parameters and provenance of the agents this one was combined from.
    this.combinationPrior = null;
    if (mathVersion) this.experience.setMathVersion(mathVersion);
    if (this.mathVersion !== 'legacy') this.decisionKernel; // validate p-parameters early
  }

  get mathVersion() { return this.experience.mathVersion; }
  setMathVersion(v) { this.experience.setMathVersion(v); }

  // The decision kernel D over ['core', 'lexicon', 'explore', 'idle'].
  get decisionKernel() {
    return buildDecisionKernel({ pStable: this.pStable, pLexicon: this.pLexicon, pExplore: this.pExplore, lexiconRow: this.lexiconRow });
  }

  static fromConfig(agentId, config = {}) {
    const agentCfg = config.agent || {};
    const stCfg = agentCfg.selfToken || {};
    const { SelfTokenState } = require('../core/self-token');
    const st = new SelfTokenState(SelfTokenState.lockOptions(stCfg));
    const exp = new ExperienceSpace({ selfToken: st });
    return new ConsciousAgent({
      agentId,
      experience: exp,
      metaObservationInterval: agentCfg.metaObservationInterval ?? 20,
      expressionTemp: agentCfg.expressionTemp ?? 1.0,
      pStable: agentCfg.pStable ?? 0.8,
      pLexicon: agentCfg.pLexicon ?? 0.1,
      pExplore: agentCfg.pExplore ?? 0.05,
      lexiconRow: agentCfg.lexiconRow ?? DEFAULT_LEXICON_ROW,
      seed: agentCfg.seed ?? null,
      mathVersion: agentCfg.mathVersion ?? 'v3',
    });
  }

  setAllowableTokens(tokens) {
    this._allowableTokens = tokens ? new Set(tokens) : null;
  }

  get allowableTokens() { return this._allowableTokens ? [...this._allowableTokens] : null; }

  setMode(mode) {
    if (!MODES[mode]) throw new Error(`Invalid mode "${mode}". Use: ${Object.keys(MODES).join(', ')}`);
    this._mode = MODES[mode];
  }

  thaw() { this._mode = MODES.learning; }
  refreeze() { this._mode = MODES.frozen; }

  get mode() { return this._mode; }

  step(world) {
    if (!world && this.world) {
      if (typeof this.world.step === 'function') {
        world = this.world.step();
      }
    }

    const isFrozen = this._mode === MODES.frozen || this._mode === MODES.debug;

    if (world) {
      this.experience = perceive(
        world,
        this.experience,
        this.stepCount,
        this.metaObservationInterval,
        isFrozen,
        this._ergodicState,
        this._rng,
        this.generation
      );
    }

    const pStable = isFrozen ? 1.0 : this.pStable;
    const pLexicon = isFrozen ? 0.0 : this.pLexicon;
    const pExplore = isFrozen ? 0.0 : this.pExplore;

    let [output, nextState] = decide(this.experience, {
      pStable,
      pLexicon,
      pExplore,
      ergodicState: this._ergodicState,
      rng: this._rng,
      lexiconRow: this.lexiconRow,
    });
    this._ergodicState = nextState;

    if (!isFrozen) {
      for (const token of output) {
        if (this.experience.metaTrie.lastMetaState !== null) {
          this.experience.metaTrie.recordToken(this.experience.metaTrie.lastMetaState, token);
        }
      }
    }

    if (this._allowableTokens) {
      output = output.filter(t => this._allowableTokens.has(t));
      if (output.length === 0) {
        const fallback = CORE_TOKENS.find(t => this._allowableTokens.has(t)) || 'wait';
        output = [fallback];
      }
    }

    this._lastOutput = output;
    this.stepCount++;
    if (this.stepCount > 0 && this.stepCount % this.metaObservationInterval === 0) {
      this.generation++;
    }

    const actionDistribution = this._computeActionDistribution(output);

    const stepOutput = new StepOutput({
      step: this.stepCount,
      generation: this.generation,
      state: this.experience.lastWorldStateId ?? -1,
      stateLabel: String(this.experience.lastWorldStateId ?? '?'),
      predictionError: this.experience.traceBuffer.predictionErrorMean(5),
      sequence: [...output],
      sequenceStr: output.join(' '),
      loopDepth: computeSelfReferenceScore(output),
      iLocked: this.experience.selfToken.locked,
      iStability: this.experience.selfToken.stationaryVariance(),
      actionDistribution,
      interrupt: this.experience.selfToken.consumeEvent ? this.experience.selfToken.consumeEvent() : null,
    });

    if (this._mode === MODES.debug) {
      stepOutput._mode = MODES.debug;
      stepOutput._frozenParams = { pStable, pLexicon, pExplore };
    }

    return stepOutput;
  }

  run(nSteps) {
    const outputs = [];
    for (let i = 0; i < nSteps; i++) outputs.push(this.step());
    return outputs;
  }

  observe(outputSequence, sourceId) {
    return this.step(new (require('./world-state')).WorldState({ [sourceId]: outputSequence }));
  }

  injectObservation(worldState) {
    return this.step(worldState);
  }

  predictNext() {
    const lastStateId = this.experience.lastWorldStateId;
    if (lastStateId == null) return null;

    const node = this.experience.trie.lookup([lastStateId]);
    if (!node || !node.children || Object.keys(node.children).length === 0) return null;

    const children = Object.entries(node.children).map(([id, child]) => ({
      stateId: parseInt(id),
      visitCount: child.visitCount,
    }));
    const total = children.reduce((s, c) => s + c.visitCount, 0);
    if (total === 0) return null;

    children.sort((a, b) => b.visitCount - a.visitCount);
    const best = children[0];
    const confidence = best.visitCount / total;
    const stateLabel = String(best.stateId);

    const topK = children.map(c => ({
      stateId: c.stateId,
      stateLabel: String(c.stateId),
      confidence: c.visitCount / total,
    }));

    return new Prediction({ stateId: best.stateId, stateLabel, confidence, topK });
  }

  // Ergodic analysis of the agent's two chains: the decision kernel D (exact)
  // and the empirical meta-state chain M that the "I" attractor is defined on.
  ergodicStats() {
    const meta = this.experience.metaTrie.ergodicDiagnostics();
    const st = this.experience.selfToken;
    return {
      mathVersion: this.mathVersion,
      decision: this.decisionKernel.diagnostics(),
      meta,
      lock: {
        locked: st.locked,
        referent: st.referentMetaStateId,
        criteria: st.lockCriteria && meta.classSize > 0 ? st.lockCriteria(meta).checks : null,
        history: st.lockHistory ? [...st.lockHistory] : [],
      },
      combinationPrior: this.combinationPrior,
    };
  }

  // Export the agent's learned dynamics as explicit kernels, in the vocabulary
  // of Hoffman & Prakash's (X, G, P, D, A, N):
  //   X - experience states: meta-states of the recurrent class
  //   P - perception: empirical world-state transition kernel (experience trie)
  //   M - the empirical meta-state kernel on X
  //   D - decision kernel over output modes
  //   A - action: meta-state -> emitted token distribution
  //   N - step counter
  toFormal() {
    const meta = this.experience.metaTrie.ergodicDiagnostics();
    const { ids: metaIds, counts: metaCounts } = this.experience.metaTrie.transitionCounts();
    const pos = new Map(metaIds.map((id, i) => [id, i]));
    const classIdx = meta.states.map(s => pos.get(s));
    const M = meta.classSize > 0
      ? new MarkovKernel({ states: meta.states, matrix: markov.normalizeRows(markov.subMatrix(metaCounts, classIdx)).P })
      : null;

    const P = worldKernel(this.experience.trie);

    const registry = this.experience.metaTrie._tokenRegistry;
    const rows = [...registry.keys()].filter(id => registry.get(id).size > 0).sort((a, b) => a - b);
    const cols = [...new Set(rows.flatMap(id => [...registry.get(id).keys()]))].sort();
    const A = rows.length > 0
      ? StochasticMatrix.fromCounts({ rows, cols, counts: rows.map(id => cols.map(t => registry.get(id).get(t) || 0)) })
      : null;

    return { X: meta.states, G: cols, P, M, D: this.decisionKernel, A, N: this.stepCount, diagnostics: meta };
  }

  getOutput() { return [...this._lastOutput]; }
  setWorld(w) { this.world = w; }

  get metrics() {
    return {
      predictionError: this.experience.traceBuffer.predictionErrorMean(5),
      iLocked: this.experience.selfToken.locked,
      iStability: this.experience.selfToken.stationaryVariance(),
      loopDepth: computeSelfReferenceScore(this._lastOutput),
      outputTokens: [...this._lastOutput],
    };
  }

  get loopScore() { return computeSelfReferenceScore(this._lastOutput); }

  get meanPredictionError() {
    return this.experience.traceBuffer.predictionErrorMean(100);
  }

  get isILocked() { return this.experience.selfToken.locked; }
  get isIdentityStable() { return this.experience.isIdentityStable; }
  get isRipe() { return this.isIdentityStable; }

  _computeActionDistribution(outputTokens) {
    const dist = {};
    for (const token of outputTokens) {
      dist[token] = (dist[token] || 0) + 1;
    }
    if (Object.keys(dist).length === 0) return {};
    const sum = Object.values(dist).reduce((a, b) => a + b, 0);
    for (const token of Object.keys(dist)) dist[token] /= sum;
    return dist;
  }

  clearMemory() {
    this.experience.traceBuffer.clear();
    this.stepCount = 0;
    this.generation = 0;
    this._lastOutput = ['wait'];
    this._ergodicState = 'idle';
  }

  clear() {
    this.clearMemory();
    this.experience.trie.clear();
    this.experience.metaTrie.clear();
    this.experience.lexicon.clear();
  }
}

// Empirical Markov kernel of world-state transitions (depth-2 trie counts),
// restricted to states whose outgoing transitions have been observed.
function worldKernel(trie) {
  const ids = new Set();
  const edges = [];
  for (const [fromStr, node] of Object.entries(trie.root.children)) {
    for (const [toStr, child] of Object.entries(node.children)) {
      if (child.visitCount <= 0) continue;
      const from = parseInt(fromStr), to = parseInt(toStr);
      edges.push([from, to, child.visitCount]);
      ids.add(from); ids.add(to);
    }
  }
  if (edges.length === 0) return null;
  const sorted = [...ids].sort((a, b) => a - b);
  const idx = new Map(sorted.map((id, i) => [id, i]));
  const counts = markov.zeros(sorted.length);
  for (const [f, t, c] of edges) counts[idx.get(f)][idx.get(t)] += c;
  const kept = markov.pruneUnobservedRows(counts);
  if (kept.length === 0) return null;
  return new MarkovKernel({ states: kept.map(i => sorted[i]), matrix: markov.normalizeRows(markov.subMatrix(counts, kept)).P });
}

module.exports = { ConsciousAgent, StepOutput, Prediction, worldKernel };
