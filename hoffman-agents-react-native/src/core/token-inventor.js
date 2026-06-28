const crypto = require('../crypto');

const CONSONANTS = 'bdfghjklmnprstvwxyz';
const VOWELS = 'aeiou';
const CORE_TOKENS = new Set(['I', 'notice', 'familiar', 'different', 'wait']);

function inventToken() {
  const c1 = CONSONANTS[Math.floor(Math.random() * CONSONANTS.length)];
  const v = VOWELS[Math.floor(Math.random() * VOWELS.length)];
  const c2 = CONSONANTS[Math.floor(Math.random() * CONSONANTS.length)];
  return `${c1}${v}${c2}`;
}

function isInventedToken(token) {
  if (CORE_TOKENS.has(token)) return false;
  if (token.startsWith('p:')) return false;
  if (token.length !== 3) return false;
  return CONSONANTS.includes(token[0]) && VOWELS.includes(token[1]) && CONSONANTS.includes(token[2]);
}

module.exports = { inventToken, isInventedToken };
