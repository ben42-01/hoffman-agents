const { ConsciousAgent, StepOutput, Prediction } = require('./agent/conscious-agent');
const { SimpleWorld } = require('./agent/simple-world');
const { WorldState, EnvironmentState, sequenceToStateId } = require('./agent/world-state');
const { ExperienceSpace } = require('./agent/experience-space');
const { World, WorldBuilder, CoinTossWorld, SelfWorld, Normalizer, FeatureSpec, buildWorldFromDataFrame } = require('./world');
const { AgentNetwork, Topology, InteractionCycle } = require('./network');
const { combine, trivialAgent, experienceSpaceDistance, fuse } = require('./combination');
const {
  serialize, deserialize, clone, fingerprint,
  saveAgent, loadAgent, loadLatest, cloneAgent,
} = require('./io');
const { TraceBuffer, TraceEvent } = require('./core/trace-buffer');
const { ExperienceTrie, TrieNode } = require('./core/experience-trie');
const { MetaTrie, MetaStateSnapshot } = require('./core/meta-trie');
const { SelfTokenState } = require('./core/self-token');
const { ExperienceLexicon, LexiconEntry } = require('./core/experience-lexicon');
const {
  strangeLoopScore, computeSelfReferenceScore,
  populationReferenceScore, populationLoopScore,
  firstDepthNGeneration,
} = require('./core/strange-loop');
const { prune, traceDistance, mergeSimilarPaths } = require('./core/trie-compression');
const { inventToken, isInventedToken } = require('./core/token-inventor');
const { SharedMeaningTracker } = require('./meaning');

// Alias for worlds module
const worlds = require('./world');
const io = require('./io');

module.exports = {
  ConsciousAgent, StepOutput, Prediction,
  SimpleWorld, WorldState, EnvironmentState, ExperienceSpace,
  World, WorldBuilder, CoinTossWorld, SelfWorld, buildWorldFromDataFrame,
  Normalizer, FeatureSpec,
  AgentNetwork, Topology, InteractionCycle,
  combine, trivialAgent, experienceSpaceDistance, fuse,
  serialize, deserialize, clone, fingerprint,
  saveAgent, loadAgent, loadLatest, cloneAgent,
  TraceBuffer, TraceEvent, ExperienceTrie, TrieNode,
  MetaTrie, MetaStateSnapshot,
  SelfTokenState, ExperienceLexicon, LexiconEntry,
  strangeLoopScore, computeSelfReferenceScore,
  populationReferenceScore, populationLoopScore,
  firstDepthNGeneration,
  prune, traceDistance, mergeSimilarPaths,
  inventToken, isInventedToken,
  sequenceToStateId,
  SharedMeaningTracker,
  worlds, io,
};
