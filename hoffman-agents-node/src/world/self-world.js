const { WorldState } = require('../agent/world-state');

const DEFAULT_GET_STATE = (agent) => ({
  sp: agent.experience.selfToken.stationaryProb,
  pe: agent.meanPredictionError,
  lock: agent.isILocked ? 1 : 0,
  loop: agent.loopScore,
  metaStates: agent.experience.metaTrie.registrySize,
  vocabSize: agent.experience.lexicon.vocabularySize(),
  trieNodes: agent.experience.trie.size(),
});

class SelfWorld {
  constructor(world, agent, getStateFn = null) {
    if (!world || typeof world.step !== 'function') {
      throw new Error('SelfWorld requires a world with a step() method');
    }
    this._inner = world;
    this._agent = agent;
    this._getState = getStateFn || DEFAULT_GET_STATE;
    this._stepCount = 0;
    this.nStates = world.nStates || 1000000;
  }

  step() {
    const worldState = this._inner.step();
    const selfMetrics = this._getState(this._agent);

    const tokens = [];
    for (const [key, val] of Object.entries(selfMetrics)) {
      const str = typeof val === 'number'
        ? `${key}:${Number(val.toFixed(2))}`
        : `${key}:${String(val)}`;
      tokens.push(str);
    }

    worldState.sequences['self'] = tokens;
    worldState.dimensions = { ...worldState.dimensions, ...selfMetrics };

    return worldState;
  }

  get initialState() { return this.step(); }
}

module.exports = { SelfWorld };
