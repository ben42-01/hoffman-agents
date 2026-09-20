// Captures golden behaviour of the 2.1.2 math. Run once against 2.1.2 code;
// the legacy test replays it with mathVersion: 'legacy'.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { ConsciousAgent, SimpleWorld } = require('../../src');
const { serialize } = require('../../src/io');
const { mulberry32 } = require('../../src/math/rng');

const runs = [];
for (const seed of [1, 2, 3]) {
  const agent = new ConsciousAgent({ agentId: `CA_legacy_${seed}`, rng: mulberry32(seed), world: new SimpleWorld({ nStates: 50, seed }) });
  const h = crypto.createHash('sha256');
  let lockStep = null;
  for (let i = 0; i < 300; i++) {
    const out = agent.step();
    h.update(out.sequenceStr + '|');
    if (out.iLocked && lockStep === null) lockStep = out.step;
  }
  runs.push({
    seed, lockStep,
    outputHash: h.digest('hex'),
    registrySize: agent.experience.metaTrie.registrySize,
    stationaryProb: agent.experience.selfToken.stationaryProb,
    lexiconSize: agent.experience.lexicon.entryCount,
  });
  if (seed === 1) serialize(agent, path.join(__dirname, 'legacy-2.1.2.soul'));
}
fs.writeFileSync(path.join(__dirname, 'legacy-2.1.2.json'), JSON.stringify(runs, null, 2) + '\n');
console.log(runs);
