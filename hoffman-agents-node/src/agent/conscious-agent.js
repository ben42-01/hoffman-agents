const crypto = require('crypto');
const { ExperienceSpace } = require('./experience-space');
const { perceive } = require('./perceptual-map');
const { decide } = require('./decision-map');
const { computeSelfReferenceScore } = require('../core/strange-loop');

class StepOutput {
  constructor({ step, generation, state, stateLabel, predictionError, sequence, sequenceStr, loopDepth, iLocked, iStability, interrupt }) {
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
  }
}

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
  }

  step(world) {
    if (!world && this.world) {
      if (typeof this.world.step === 'function') {
        world = this.world.step();
      }
    }

    if (world) {
      this.experience = perceive(
        world,
        this.experience,
        this.stepCount,
        this.metaObservationInterval
      );
    }

    const [output, nextState] = decide(this.experience, {
      pStable: this.pStable,
      pLexicon: this.pLexicon,
      pExplore: this.pExplore,
      ergodicState: this._ergodicState,
    });
    this._ergodicState = nextState;

    for (const token of output) {
      if (this.experience.metaTrie.lastMetaState !== null) {
        this.experience.metaTrie.recordToken(this.experience.metaTrie.lastMetaState, token);
      }
    }

    this._lastOutput = output;
    this.stepCount++;
    if (this.stepCount > 0 && this.stepCount % this.metaObservationInterval === 0) {
      this.generation++;
    }

    return new StepOutput({
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
    });
  }

  run(nSteps) {
    const outputs = [];
    for (let i = 0; i < nSteps; i++) outputs.push(this.step());
    return outputs;
  }

  observe(outputSequence, sourceId) {
    return this.step(new (require('./world-state')).WorldState({ [sourceId]: outputSequence }));
  }

  getOutput() { return [...this._lastOutput]; }
  setWorld(w) { this.world = w; }

  get loopScore() { return computeSelfReferenceScore(this._lastOutput); }

  get meanPredictionError() {
    return this.experience.traceBuffer.predictionErrorMean(100);
  }

  get isILocked() { return this.experience.selfToken.locked; }
  get isIdentityStable() { return this.experience.isIdentityStable; }
  get isRipe() { return this.isIdentityStable; }

  clear() {
    this.experience.traceBuffer.clear();
    this.experience.trie.clear();
    this.experience.metaTrie.clear();
    this.experience.lexicon.clear();
    this.stepCount = 0;
    this.generation = 0;
    this._lastOutput = ['wait'];
  }
}

module.exports = { ConsciousAgent, StepOutput };
