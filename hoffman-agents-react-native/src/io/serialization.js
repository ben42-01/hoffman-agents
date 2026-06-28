let fs = null, path = null;
try {
  fs = require('fs');
  path = require('path');
} catch (_) {
  /* React Native — no fs/path available */
}
const crypto = require('../crypto');
const { ExperienceTrie } = require('../core/experience-trie');
const { MetaTrie } = require('../core/meta-trie');
const { SelfTokenState } = require('../core/self-token');
const { TraceBuffer, TraceEvent } = require('../core/trace-buffer');
const { ExperienceLexicon } = require('../core/experience-lexicon');
const { ConsciousAgent } = require('../agent/conscious-agent');
const { ExperienceSpace } = require('../agent/experience-space');

function serializeToJson(agent) {
  return JSON.stringify({
    caVersion: '1.0',
    agentId: agent.agentId,
    generation: agent.generation,
    step: agent.stepCount,
    components: {
      traceBuffer: serializeTraceBuffer(agent.experience.traceBuffer),
      experienceTrie: serializeTrie(agent.experience.trie),
      metaTrie: serializeMetaTrie(agent.experience.metaTrie),
      selfToken: serializeSelfToken(agent.experience.selfToken),
      lexicon: agent.experience.lexicon.toDict(),
    },
    config: {
      traceBufferLength: agent.experience.traceBuffer.maxlen,
      trieMaxDepth: agent.experience.trie.maxDepth,
      metaObservationInterval: agent.metaObservationInterval,
      lockThreshold: agent.experience.selfToken.lockThreshold,
      lockConsecutiveRequired: agent.experience.selfToken.lockConsecutiveRequired,
      protectionRadius: agent.experience.selfToken.protectionRadius,
    },
    metadata: {
      cycleLevel: agent.cycleLevel,
      constituentIds: [...agent.constituentIds].sort(),
      leafConstituentIds: [...agent.leafConstituentIds].sort(),
      combined: agent._combined,
    },
  }, null, 2);
}

function serialize(agent, filePath) {
  const json = serializeToJson(agent);
  if (fs) {
    const tmpPath = filePath + '.tmp.' + crypto.randomUUID();
    fs.writeFileSync(tmpPath, json);
    fs.renameSync(tmpPath, filePath);
  }
  return json;
}

function deserializeFromJson(json) {
  const data = typeof json === 'string' ? JSON.parse(json) : json;
  const comp = data.components || {};

  const exp = new ExperienceSpace({
    traceBuffer: deserializeTraceBuffer(comp.traceBuffer || { maxlen: 50, events: [] }),
    trie: deserializeTrie(comp.experienceTrie || { maxDepth: 10, nodes: [] }),
    metaTrie: deserializeMetaTrie(comp.metaTrie || {}),
    selfToken: deserializeSelfToken(comp.selfToken || {}),
    lexicon: ExperienceLexicon.fromDict(comp.lexicon || { entries: [] }),
  });

  const meta = data.metadata || {};
  const agent = new ConsciousAgent({
    agentId: data.agentId || 'unknown',
    experience: exp,
    generation: data.generation || 0,
    stepCount: data.step || 0,
    metaObservationInterval: (data.config || {}).metaObservationInterval || 20,
    constituentIds: new Set(meta.constituentIds || []),
    leafConstituentIds: new Set(meta.leafConstituentIds || []),
    cycleLevel: meta.cycleLevel || 0,
  });
  agent._combined = meta.combined || false;
  return agent;
}

function deserialize(filePath) {
  if (!fs) throw new Error(`Filesystem unavailable (React Native). Use deserializeFromJson() instead.`);
  return deserializeFromJson(fs.readFileSync(filePath, 'utf8'));
}

function clone(agent, newId) {
  const json = serializeToJson(agent);
  const cloned = deserializeFromJson(json);
  cloned.agentId = newId || `${agent.agentId}_clone`;
  return cloned;
}

function fingerprint(agent) {
  const comp = {
    trie: serializeTrie(agent.experience.trie),
    metaTrie: serializeMetaTrie(agent.experience.metaTrie),
    selfToken: serializeSelfToken(agent.experience.selfToken),
  };
  return crypto.createHash('sha256').update(JSON.stringify(comp)).digest('hex');
}

function saveAgent(agent, directory = './souls') {
  fs.mkdirSync(directory, { recursive: true });
  const filePath = path.join(directory, `agent_${agent.agentId}_gen${String(agent.generation).padStart(6, '0')}_step${String(agent.stepCount).padStart(10, '0')}.soul`);
  serialize(agent, filePath);
  return filePath;
}

function loadAgent(filePath) {
  return deserialize(filePath);
}

function loadLatest(soulDir, agentId) {
  const files = fs.readdirSync(soulDir)
    .filter(f => f.startsWith(`agent_${agentId}_`) && f.endsWith('.soul'))
    .sort();
  if (files.length === 0) return null;
  return deserialize(path.join(soulDir, files[files.length - 1]));
}

function serializeTrie(trie) {
  const nodes = [];
  const collect = (node, path) => {
    if (node !== trie.root) {
      nodes.push({
        path: [...path],
        visitCount: node.visitCount,
        predictionErrors: node.predictionErrors,
        meanPredictionError: node.meanPredictionError,
        wordBinding: node.wordBinding,
      });
    }
    for (const [cs, child] of Object.entries(node.children)) {
      collect(child, [...path, parseInt(cs)]);
    }
  };
  collect(trie.root, []);
  return { maxDepth: trie.maxDepth, nodes };
}

function deserializeTrie(data) {
  const trie = new ExperienceTrie(data.maxDepth || 10);
  for (const nd of data.nodes || []) {
    trie.insert(nd.path);
    const node = trie.lookup(nd.path);
    if (node) {
      node.visitCount = nd.visitCount;
      node.predictionErrors = [...nd.predictionErrors];
      node.meanPredictionError = nd.meanPredictionError;
      node.wordBinding = nd.wordBinding || null;
    }
  }
  return trie;
}

function serializeMetaTrie(mt) {
  const registry = {};
  for (const [mid, snap] of mt._registry) {
    registry[String(mid)] = {
      stateIds: [...snap.stateIds],
      meanPredictionError: snap.meanPredictionError,
      timestamp: snap.timestamp,
    };
  }
  const tokenRegistry = {};
  for (const [mid, counts] of mt._tokenRegistry) {
    const obj = {};
    for (const [token, count] of counts) obj[token] = count;
    tokenRegistry[String(mid)] = obj;
  }
  return {
    trie: serializeTrie(mt.trie),
    registry,
    lastMetaState: mt.lastMetaState,
    snapshotWindow: mt._snapshotWindow,
    maxDepth: mt.trie.maxDepth,
    tokenRegistry,
  };
}

function deserializeMetaTrie(data) {
  const mt = new MetaTrie(data.snapshotWindow || 10, data.maxDepth || 10);
  mt._trie = deserializeTrie(data.trie || { nodes: [] });

  for (const [midStr, snapData] of Object.entries(data.registry || {})) {
    const mid = parseInt(midStr);
    mt._registry.set(mid, {
      stateIds: snapData.stateIds,
      meanPredictionError: snapData.meanPredictionError,
      timestamp: snapData.timestamp || 0,
    });
  }

  mt._lastMetaState = data.lastMetaState || null;

  for (const [midStr, counts] of Object.entries(data.tokenRegistry || {})) {
    const mid = parseInt(midStr);
    const map = new Map();
    for (const [token, count] of Object.entries(counts)) map.set(token, count);
    mt._tokenRegistry.set(mid, map);
  }

  return mt;
}

function serializeSelfToken(st) {
  return {
    token: st.token,
    referentMetaStateId: st.referentMetaStateId,
    stationaryProb: st.stationaryProb,
    locked: st.locked,
    lockGeneration: st.lockGeneration,
    lockThreshold: st.lockThreshold,
    consecutiveAboveThreshold: st.consecutiveAboveThreshold,
    lockConsecutiveRequired: st.lockConsecutiveRequired,
    stabilityHistory: [...st.stabilityHistory],
    protectionRadius: st.protectionRadius,
  };
}

function deserializeSelfToken(data) {
  const st = new SelfTokenState({
    token: data.token || 'I',
    referentMetaStateId: data.referentMetaStateId || null,
    stationaryProb: data.stationaryProb || 0,
    locked: data.locked || false,
    lockGeneration: data.lockGeneration || null,
    lockThreshold: data.lockThreshold || 0.25,
    consecutiveAboveThreshold: data.consecutiveAboveThreshold || 0,
    lockConsecutiveRequired: data.lockConsecutiveRequired || 3,
    protectionRadius: data.protectionRadius || 2,
  });
  st.stabilityHistory = data.stabilityHistory || [];
  return st;
}

function serializeTraceBuffer(buf) {
  const events = [];
  for (const e of buf) {
    events.push({
      fromState: e.fromState, toState: e.toState,
      timestamp: e.timestamp, prediction: e.prediction,
      predictionCorrect: e.predictionCorrect, predictionError: e.predictionError,
      token: e.token,
    });
  }
  return { maxlen: buf.maxlen, events };
}

function deserializeTraceBuffer(data) {
  const buf = new TraceBuffer(data.maxlen || 50);
  for (const ed of data.events || []) {
    buf.append(new TraceEvent(ed.fromState, ed.toState, ed.timestamp, ed.prediction,
      ed.predictionCorrect, ed.predictionError, ed.token));
  }
  return buf;
}

module.exports = {
  serialize, serializeToJson, deserialize, deserializeFromJson,
  clone, fingerprint,
  saveAgent, loadAgent, loadLatest,
  cloneAgent: clone,
};
