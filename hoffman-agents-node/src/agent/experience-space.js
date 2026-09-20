const { ExperienceTrie } = require('../core/experience-trie');
const { MetaTrie } = require('../core/meta-trie');
const { SelfTokenState } = require('../core/self-token');
const { ExperienceLexicon } = require('../core/experience-lexicon');
const { TraceBuffer } = require('../core/trace-buffer');

const MATH_VERSIONS = ['v3', 'legacy'];

class ExperienceSpace {
  constructor({
    trie = new ExperienceTrie(10),
    metaTrie = new MetaTrie(10, 10),
    selfToken = new SelfTokenState(),
    lexicon = new ExperienceLexicon(64),
    traceBuffer = new TraceBuffer(50),
    lastWorldStateId = null,
    mathVersion = null,
  } = {}) {
    this.trie = trie;
    this.metaTrie = metaTrie;
    this.selfToken = selfToken;
    this.lexicon = lexicon;
    this.traceBuffer = traceBuffer;
    this.lastWorldStateId = lastWorldStateId;
    if (mathVersion) this.setMathVersion(mathVersion);
  }

  // 'v3' (default) or 'legacy' (reproduces 2.x dynamics).
  setMathVersion(mathVersion) {
    if (!MATH_VERSIONS.includes(mathVersion)) {
      throw new Error(`Invalid mathVersion "${mathVersion}". Use: ${MATH_VERSIONS.join(', ')}`);
    }
    this.metaTrie.mathVersion = mathVersion;
    this.selfToken.mathVersion = mathVersion;
    return this;
  }

  get mathVersion() { return this.metaTrie.mathVersion; }

  get isIdentityStable() { return this.selfToken.isStable(); }
  get isILocked() { return this.isIdentityStable; }
}

const MemorySpace = ExperienceSpace;

module.exports = { ExperienceSpace, MemorySpace, MATH_VERSIONS };
