// 2.x output-state chain, preserved verbatim for mathVersion: 'legacy'.
function nextState(current, pStable, pLexicon, pExplore, rng) {
  const r = rng();
  if (current === 'lexicon') {
    if (r < 0.7) return 'core';
    if (r < 0.85) return 'lexicon';
    if (r < 0.95) return 'explore';
    return 'idle';
  } else {
    if (r < pStable) return 'core';
    if (r < pStable + pLexicon) return 'lexicon';
    if (r < pStable + pLexicon + pExplore) return 'explore';
    return 'idle';
  }
}

module.exports = { nextState };
