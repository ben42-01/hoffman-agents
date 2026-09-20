const { WorldState, SimpleWorld } = require('../../src');
const { mulberry32 } = require('../../src/math/rng');

// Stays in "home" with probability pHome each step, otherwise visits one of n
// other states uniformly at random: a single dominant experiential attractor.
function homeWorld(seed, pHome = 0.9, n = 20) {
  const r = mulberry32(seed + 999);
  return { step: () => WorldState.fromSequence('world', [r() < pHome ? 'home' : `s${Math.floor(r() * n)}`]) };
}

function noiseWorld(seed, nStates = 50) {
  return new SimpleWorld({ nStates, seed });
}

// Attractor until `switchAt`, i.i.d.-like noise afterwards.
function switchingWorld(seed, switchAt) {
  const home = homeWorld(seed, 0.95);
  const noise = noiseWorld(seed);
  let t = 0;
  return { step: () => (++t < switchAt ? home.step() : noise.step()) };
}

module.exports = { homeWorld, noiseWorld, switchingWorld };
