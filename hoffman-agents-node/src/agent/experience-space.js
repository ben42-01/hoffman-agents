const { ExperienceTrie } = require('../core/experience-trie');
const { MetaTrie } = require('../core/meta-trie');
const { SelfTokenState } = require('../core/self-token');
const { ExperienceLexicon } = require('../core/experience-lexicon');
const { TraceBuffer } = require('../core/trace-buffer');

class ExperienceSpace {
  constructor({
    trie = new ExperienceTrie(10),
    parentTries = null,
    metaTrie = new MetaTrie(10, 10),
    parentMetaTries = null,
    selfToken = new SelfTokenState(),
    lexicon = new ExperienceLexicon(64),
    traceBuffer = new TraceBuffer(50),
    lastWorldStateId = null,
  } = {}) {
    this.trie = trie;
    this.parentTries = parentTries;
    this.metaTrie = metaTrie;
    this.parentMetaTries = parentMetaTries;
    this.selfToken = selfToken;
    this.lexicon = lexicon;
    this.traceBuffer = traceBuffer;
    this.lastWorldStateId = lastWorldStateId;
  }

  get isIdentityStable() { return this.selfToken.isStable(); }
  get isILocked() { return this.isIdentityStable; }
}

const MemorySpace = ExperienceSpace;

module.exports = { ExperienceSpace, MemorySpace };
