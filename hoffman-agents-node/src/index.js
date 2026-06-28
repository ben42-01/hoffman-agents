const { ConsciousAgent, StepOutput } = require('./agent/conscious-agent');
const { SimpleWorld } = require('./agent/simple-world');
const { WorldState, sequenceToStateId } = require('./agent/world-state');
const { ExperienceSpace } = require('./agent/experience-space');
const { World, WorldBuilder, CoinTossWorld, SelfWorld } = require('./world');
const { AgentNetwork } = require('./network');
const { combine, trivialAgent, experienceSpaceDistance } = require('./combination');
const {
  serialize, deserialize, clone, fingerprint,
  saveAgent, loadAgent, loadLatest, cloneAgent,
} = require('./io');
const { TraceBuffer, TraceEvent } = require('./core/trace-buffer');
const { ExperienceTrie } = require('./core/experience-trie');
const { MetaTrie } = require('./core/meta-trie');
const { SelfTokenState } = require('./core/self-token');
const { ExperienceLexicon, LexiconEntry } = require('./core/experience-lexicon');
const { strangeLoopScore, computeSelfReferenceScore } = require('./core/strange-loop');
const { SharedMeaningTracker } = require('./meaning');

// Alias for worlds module
const worlds = require('./world');
const io = require('./io');

module.exports = {
  ConsciousAgent, StepOutput,
  SimpleWorld, WorldState, ExperienceSpace,
  World, WorldBuilder, CoinTossWorld, SelfWorld,
  AgentNetwork,
  combine, trivialAgent, experienceSpaceDistance,
  serialize, deserialize, clone, fingerprint,
  saveAgent, loadAgent, loadLatest, cloneAgent,
  TraceBuffer, TraceEvent, ExperienceTrie, MetaTrie,
  SelfTokenState, ExperienceLexicon, LexiconEntry,
  strangeLoopScore, computeSelfReferenceScore,
  SharedMeaningTracker,
  worlds, io,
};
