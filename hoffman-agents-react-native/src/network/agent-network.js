const { WorldState } = require('../agent/world-state');
const { ConsciousAgent } = require('../agent/conscious-agent');
const { combine } = require('../combination/operator');

class Connection {
  constructor(strength = 0.5, age = 0) {
    this.strength = strength;
    this.age = age;
  }
}

class Topology {
  constructor({ nAgents, initialConnectionsPerAgent = 5, seed = 42, learningRate = 0.01, pruneThreshold = 0.05, addThreshold = 0.5 } = {}) {
    this._nAgents = nAgents;
    this._learningRate = learningRate;
    this._pruneThreshold = pruneThreshold;
    this._addThreshold = addThreshold;
    this._rng = seedRandom(seed);
    this._connections = new Map();
    this._agentIds = [];
    this._buildInitialTopology(initialConnectionsPerAgent);
  }

  _buildInitialTopology(initialPerAgent) {
    for (let i = 0; i < this._nAgents; i++) {
      const candidates = [];
      for (let j = 0; j < this._nAgents; j++) if (j !== i) candidates.push(j);
      const targets = candidates.length <= initialPerAgent ? candidates : shuffle(candidates, this._rng).slice(0, initialPerAgent);
      for (const j of targets) {
        const key = `${i},${j}`;
        if (!this._connections.has(key)) {
          this._connections.set(key, new Connection(0.3 + this._rng() * 0.4));
        }
      }
    }
  }

  setAgentIds(ids) { this._agentIds = ids; }

  getConnections(agentIdx) {
    const result = [];
    for (const [key, conn] of this._connections) {
      const [i, j] = key.split(',').map(Number);
      if (i === agentIdx && conn.strength > 0) result.push(j);
    }
    return result;
  }

  updateConnection(i, j, delta) {
    const key = `${i},${j}`;
    if (!this._connections.has(key)) this._connections.set(key, new Connection(0));
    const conn = this._connections.get(key);
    conn.strength += this._learningRate * delta;
    conn.strength = Math.max(0, Math.min(1, conn.strength));
    conn.age++;
  }

  pruneWeakConnections() {
    let count = 0;
    for (const [key, conn] of this._connections) {
      if (conn.strength < this._pruneThreshold) {
        this._connections.delete(key);
        count++;
      }
    }
    return count;
  }

  get connectionCount() { return this._connections.size; }
}

class InteractionCycle {
  constructor({ topology, metaObservationInterval = 20, topologyUpdateInterval = 10, predictionWindow = 10 } = {}) {
    this._topology = topology;
    this._metaObservationInterval = metaObservationInterval;
    this._topologyUpdateInterval = topologyUpdateInterval;
    this._predictionWindow = predictionWindow;
  }

  step(agents, agentIds, generation = 0) {
    const outputs = {};

    for (let idx = 0; idx < agentIds.length; idx++) {
      const agentId = agentIds[idx];
      const agent = agents[agentId];
      if (!agent) continue;

      let world;
      if (agent.constituentIds.size > 0) {
        world = this._buildCombinedWorld(agent, agents);
      } else {
        world = this._buildWorldFromTopology(idx, agent, agents, agentIds);
      }
      const out = agent.step(world);
      outputs[agentId] = out.sequence;
    }

    this._updateTopology(agents, agentIds, generation);
    return outputs;
  }

  _buildCombinedWorld(agent, allAgents) {
    const sequences = {};
    for (const cid of agent.constituentIds) {
      const other = allAgents[cid];
      if (other && cid !== agent.agentId) sequences[cid] = other.getOutput();
    }
    return new WorldState(sequences);
  }

  _buildWorldFromTopology(idx, agent, allAgents, agentIds) {
    const connected = this._topology.getConnections(idx);
    const sequences = {};
    for (const otherIdx of connected) {
      if (otherIdx >= agentIds.length) continue;
      const otherId = agentIds[otherIdx];
      if (otherId === agent.agentId) continue;
      const other = allAgents[otherId];
      if (other) sequences[otherId] = other.getOutput();
    }
    return new WorldState(sequences);
  }

  _updateTopology(agents, agentIds, generation) {
    if (this._topologyUpdateInterval === 0 || generation % this._topologyUpdateInterval !== 0) return;

    for (let i = 0; i < agentIds.length; i++) {
      const agentI = agents[agentIds[i]];
      if (!agentI) continue;
      for (let j = 0; j < agentIds.length; j++) {
        if (i === j) continue;
        const predError = agentI.experience.traceBuffer.predictionErrorMean(this._predictionWindow);
        const reduction = predError < 0.5 ? 0.1 - predError : -0.05;
        this._topology.updateConnection(i, j, reduction);
      }
    }
    this._topology.pruneWeakConnections();
  }
}

class AgentNetwork {
  constructor({ nAgents = 20, initialConnections = 5, metaObservationInterval = 20, seed = 42 } = {}) {
    this._nAgents = nAgents;
    this._agents = {};
    this._agentIds = [];
    this._generation = 0;
    this._outputHistory = [];

    this._topology = new Topology({ nAgents, initialConnectionsPerAgent: initialConnections, seed });
    this._cycle = new InteractionCycle({ topology: this._topology, metaObservationInterval });
    this._initAgents(seed);
    this.combinationLog = [];
  }

  _initAgents(seed) {
    for (let i = 0; i < this._nAgents; i++) {
      const agentId = `CA_${String(i).padStart(3, '0')}`;
      const agentSeed = seed + i * 1000;
      let s = agentSeed;
      const agentRng = () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return (s >>> 0) / 0x7FFFFFFF; };
      this._agents[agentId] = new ConsciousAgent({ agentId, rng: agentRng });
      this._agentIds.push(agentId);
    }
    this._topology.setAgentIds(this._agentIds);
  }

  step() {
    const outputs = this._cycle.step(this._agents, this._agentIds, this._generation);
    this._outputHistory.push(outputs);
    const state = { generation: this._generation, outputs };
    this._generation++;
    return state;
  }

  run(nGenerations) {
    const states = [];
    for (let i = 0; i < nGenerations; i++) states.push(this.step());
    return states;
  }

  getAgent(agentId) { return this._agents[agentId] || null; }
  get agents() { return { ...this._agents }; }
  get agentList() { return this._agentIds.map(id => this._agents[id]).filter(Boolean); }
  get agentIds() { return [...this._agentIds]; }
  get generation() { return this._generation; }

  stepAll(worldState) {
    return this._agentIds
      .map(id => this._agents[id])
      .filter(Boolean)
      .map(agent => agent.step(worldState));
  }

  avgPredictionError() {
    if (this._agentIds.length === 0) return 0;
    let sum = 0, count = 0;
    for (const aid of this._agentIds) {
      const ag = this._agents[aid];
      if (ag) { sum += ag.experience.traceBuffer.predictionErrorMean(10); count++; }
    }
    return count > 0 ? sum / count : 0;
  }

  getMetrics() {
    const stepResults = this.agentList.map(a => a.metrics);
    if (stepResults.length === 0) {
      return {
        agentCount: 0, meanPredictionError: 0, predictionErrorVariance: 0,
        iLockRate: 0, meanLoopDepth: 0, dominantTokenRatio: 0,
      };
    }
    const errors = stepResults.map(r => r.predictionError);
    const meanPE = errors.reduce((s, v) => s + v, 0) / errors.length;
    const varPE = errors.reduce((s, v) => s + (v - meanPE) ** 2, 0) / errors.length;
    const iLockRate = stepResults.filter(r => r.iLocked).length / stepResults.length;
    const meanLoopDepth = stepResults.reduce((s, r) => s + r.loopDepth, 0) / stepResults.length;
    const tokenCounts = {};
    for (const r of stepResults) {
      for (const t of r.outputTokens) tokenCounts[t] = (tokenCounts[t] || 0) + 1;
    }
    const maxCount = Math.max(...Object.values(tokenCounts), 0);
    const totalTokens = Object.values(tokenCounts).reduce((s, v) => s + v, 0);
    const dominantTokenRatio = totalTokens > 0 ? maxCount / totalTokens : 0;
    return {
      agentCount: stepResults.length, meanPredictionError: meanPE,
      predictionErrorVariance: varPE, iLockRate,
      meanLoopDepth, dominantTokenRatio,
    };
  }

  getAgentMetrics(agentId) {
    const agent = this._agents[agentId];
    return agent ? agent.metrics : null;
  }
}

function seedRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) & 0x7FFFFFFF;
    return s / 0x7FFFFFFF;
  };
}

function shuffle(arr, rng) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

module.exports = { AgentNetwork, Topology, InteractionCycle };
