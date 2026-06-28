const crypto = require('crypto');
const { ExperienceSpace } = require('./experience-space');
const { perceive } = require('./perceptual-map');
const { decide, CORE_TOKENS } = require('./decision-map');
const { computeSelfReferenceScore } = require('../core/strange-loop');

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
    allowableTokens = null,
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
    this._rng = rng || Math.random.bind(Math);
    this._allowableTokens = allowableTokens ? new Set(allowableTokens) : null;
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
        isFrozen
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
      state: this.experience.lastWorldStateId || -1,
      stateLabel: String(this.experience.lastWorldStateId || '?'),
      predictionError: this.experience.traceBuffer.predictionErrorMean(5),
      sequence: [...output],
      sequenceStr: output.join(' '),
      loopDepth: computeSelfReferenceScore(output),
      iLocked: this.experience.selfToken.locked,
      iStability: this.experience.selfToken.stationaryVariance(),
      actionDistribution,
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

module.exports = { ConsciousAgent, StepOutput };
